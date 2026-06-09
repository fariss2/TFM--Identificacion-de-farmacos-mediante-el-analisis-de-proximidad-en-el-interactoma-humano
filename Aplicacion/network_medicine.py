import pandas as pd
import os
import numpy as np
from collections import deque
from gene_to_uniprot import convert_gene_list#genes--uniprot
import random

def bfs_multifuente(grafo, origenes):
    """
    Calcula la distancia mínima en pasos desde un conjunto de nodos origen (enfermedad) hacia todos los demás nodos del interactoma.

      params:
        grafo: Grafo no dirigido que representa el interactoma humano 
        origenes: Conjunto o lista de nodos (identificadores UniProt) 
        que conforman la firma molecular o el gen de referencia.

    devuelve:
        dict: Un diccionario donde las claves son todos los identificadores 
        proteicos del grafo y los valores representan su distancia mínima 
        entera respecto al conjunto de entrada.    
    
    """
    dist = {n: float("inf") for n in grafo.nodes()}
    q = deque()
    for s in origenes:
        if s in dist:
            dist[s] = 0
            q.append(s)
    while q:
        u = q.popleft()
        for v in grafo.neighbors(u):
            if dist[v] == float("inf"):
                dist[v] = dist[u] + 1
                q.append(v)
    return dist
def generar_dianas_aleatorias(dianas_reales, grados, bins, tamaño_bin):
    """
    Genera un conjunto de dianas moleculares ficticias mediante un muestreo emparejado por el grado de conectividad topológica

    params:
        dianas_reales: Lista de proteínas diana reales asociadas a un fármaco.
        grados: Diccionario de conectividad del interactoma {nodo: grado}.
        bins:Estructura de contenedores que agrupa las proteínas del interactoma según su grado {bin_id: [lista_de_nodos]}.
        tamaño_bin: Amplitud del intervalo de conectividad para el agrupamiento.

    devuelve:
        list: Colección de nodos muestreados al azar que imitan las propiedades de grado del perfil farmacológico original.
    
    """
    aleatorias = []
    for d in dianas_reales:
        g = grados.get(d)
        if g is None:
            continue
        bin_id = g // tamaño_bin
        if bin_id not in bins or not bins[bin_id]:
            continue
        aleatorias.append(random.choice(bins[bin_id]))
    return aleatorias
def distancia_media_conjunto(dianas, distancias_ref):
    """
    Calcula la distancia mínima promedio (observada)entre las dianas moleculares de un fármaco y el módulo de la enfermedad.

    params:
        dianas: Lista de proteínas sobre las cuales actúa el compuesto terapéutico.
        distancias_ref: Diccionario de distancias mínimas generado previamente por `bfs_multifuente`.

    devuelve:
        Distancia media observada en la red. Devuelve `None` si ninguna diana es resoluble.
    
    """
 
    vals = [distancias_ref.get(d, float("inf")) for d in dianas]
    vals = [v for v in vals if v != float("inf")]
    return sum(vals) / len(vals) if vals else None
def proximidad_estadistica(dianas, distancias_ref, grados, bins, repeticiones=200, tamaño_bin=50):
    """
    Realiza una simulación MC para evaluar la significación estadística de la proximidad observada de un fármaco frente al módulo de la enfermedad.
    params:
        dianas: Conjunto de identificadores proteicos asociados al fármaco a evaluar.
        distancias_ref: Distancias basales calculadas desde el foco clínico.
        grados: Diccionario topológico de conectividad {nodo: grado}.
        bins: Contenedores homogéneos para el remuestreo controlado por grado.
        repeticiones: Número de simulaciones independientes de MC. Por defecto=200.
        tamaño_bin: Rango de tolerancia para agrupar nodos homólogos. Por defecto=50.

    devuelve:
        Tupla con los indicadores estadísticos del compuesto:
            - dist_obs:Distancia real promedia observaeda
            - media: Media de la distribución aleatoria simulada 
            - sd: Desviación estándar de la distribución aleatoria.
            - z: z-score 
            - p: pvalor
    """
    dist_obs = distancia_media_conjunto(dianas, distancias_ref)
    if dist_obs is None:
        return None, None, None, None, None
    dist_aleatorias = []
    for _ in range(repeticiones):
        rand_set = generar_dianas_aleatorias(dianas, grados, bins, tamaño_bin)
        d_rand = distancia_media_conjunto(rand_set, distancias_ref)
        if d_rand is not None:
            dist_aleatorias.append(d_rand)
    if len(dist_aleatorias) < 3:
        return dist_obs, None, None, None, None
    media = float(np.mean(dist_aleatorias))
    sd = float(np.std(dist_aleatorias)) if np.std(dist_aleatorias) > 0 else 1e-9
    z = (dist_obs - media) / sd
    p = (1 + sum(x <= dist_obs for x in dist_aleatorias)) / (1 + len(dist_aleatorias))
    return dist_obs, media, sd, z, p


