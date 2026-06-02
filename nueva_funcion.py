import pandas as pd
import networkx as nx
import os
import numpy as np
from collections import deque
from gene_to_uniprot import convert_gene_list#genes--uniprot

DATA_PATH = r"C:\Users\Nisrin Fariss Lamine\Downloads\tfm"

def cargar_datos():
    print("Cargando BioGRID...")
    df_edges = pd.read_csv(os.path.join(DATA_PATH, "biogrid_edges.csv"))
    G = nx.from_pandas_edgelist(df_edges, source="source", target="target")

    print("Cargando DrugBank limpio...")
    df_drug = pd.read_csv(os.path.join(DATA_PATH, "drugbank_targets_clean.csv"))

    print("Cargando Enfermedades...")
    df_disorders = pd.read_csv(os.path.join(DATA_PATH, "disorder_genes.csv"), sep=";")

    return G, df_drug, df_disorders
def bfs_multifuente(grafo, origenes):
    
    distancias = {nodo: float("inf") for nodo in grafo.nodes()}#dist inf,nodos no visitados
    cola = deque()

    for o in origenes:
        if o in distancias:#si esta en nodos no visitados
            distancias[o] = 0# a simismo 
            cola.append(o)

    while cola:
        actual = cola.popleft()
        for vecino in grafo.neighbors(actual):
            if distancias[vecino] == float("inf"):
                distancias[vecino] = distancias[actual] + 1
                cola.append(vecino)

    return distancias#diccionario de nodo y su distancia minima desde  laenfermedad al resto de nodos

def construir_bins_por_grado(grafo, tamaño_bin=50):
    
    grados = {nodo: grafo.degree(nodo) for nodo in grafo.nodes()}#conexion de cada nodo
    bins = {}
    for proteina, grado in grados.items():
        bin_id = grado // tamaño_bin #ver a q bin corresponde, eneteros
        bins.setdefault(bin_id, []).append(proteina)
        
    return grados, bins
def seleccionar_enfermedad_valida(df_disorders, G):
    for enf in df_disorders["disorder"].unique():

        genes_str = df_disorders[df_disorders["disorder"] == enf]["gene_symb"].values[0]
        genes_simbolos = [g.strip() for g in genes_str.split(",") if g.strip()]

        conversion = convert_gene_list(genes_simbolos)

        genes_uniprot = [
            u
            for lista in conversion.values()
            for u in lista
        ]

        genes_presentes = [
            u for u in genes_uniprot
            if u in G.nodes()
        ]

        if len(genes_presentes) >= 1:
            return enf, genes_presentes

    return None, []
def generar_dianas_aleatorias(dianas_reales, grados, bins, tamaño_bin=50):
    
    aleatorias = []

    for diana in dianas_reales:

        grado = grados.get(diana)#grado de esa diana

        if grado is None:# si no esta en el grafo se ignora
            continue

        bin_id = grado // tamaño_bin

        if bin_id not in bins:#bin inexistente se ignora
            continue
        aleatorias.append(random.choice(bins[bin_id]))#selecciona una proteína aleatoria del mismo bin


    return aleatorias

def distancia_media_conjunto(dianas, distancias):
    
    #la distancia de cada diana desde el diccionario de distancias
    valores = [distancias.get(d, float("inf")) for d in dianas]

    # no alcanzables(infinito)
    valores = [v for v in valores if v != float("inf")]

    return sum(valores) / len(valores) if valores else None

def proximidad_estadistica(dianas, distancias_ref, grados, bins, repeticiones=200):
    dist_obs = distancia_media_conjunto(dianas, distancias_ref)#dist real

    if dist_obs is None:
        return None, None, None, None, None

    dist_aleatorias = []

    for _ in range(repeticiones): #simulaciones =

        d_rand = generar_dianas_aleatorias(dianas, grados, bins)

        dist_rand = distancia_media_conjunto(d_rand, distancias_ref)
        if dist_rand is not None:
            dist_aleatorias.append(dist_rand)

    if len(dist_aleatorias) < 3: #en caso de pocas observaciones
        return dist_obs, None, None, None, None

    media = np.mean(dist_aleatorias)
    desviacion = np.std(dist_aleatorias) if np.std(dist_aleatorias) > 0 else 1e-9#q no divida entre 0

    z = (dist_obs - media) / desviacion#  cuántas desviaciones está la real respecto a la media
    p = (1 + sum(x <= dist_obs for x in dist_aleatorias)) / (1 + len(dist_aleatorias))

    return dist_obs, media, desviacion, z, p

    