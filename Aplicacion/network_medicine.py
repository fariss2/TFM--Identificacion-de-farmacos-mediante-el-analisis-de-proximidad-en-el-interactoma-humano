import pandas as pd
import os
import numpy as np
from collections import deque
from gene_to_uniprot import convert_gene_list#genes--uniprot
import random

def bfs_multifuente(grafo, origenes):
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
    vals = [distancias_ref.get(d, float("inf")) for d in dianas]
    vals = [v for v in vals if v != float("inf")]
    return sum(vals) / len(vals) if vals else None
def proximidad_estadistica(dianas, distancias_ref, grados, bins, repeticiones=200, tamaño_bin=50):
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


