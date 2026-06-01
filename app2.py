import streamlit as st
import pandas as pd
import networkx as nx
import os
import numpy as np
import random
from collections import deque
from pyvis.network import Network
import streamlit.components.v1 as components
from gene_to_uniprot import convert_gene_list

# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------
st.set_page_config(page_title="Reposicionamiento de Fármacos", layout="wide")
st.title("Reposicionamiento de Fármacos mediante Medicina de Redes")

st.info("""
**¿Qué hace este análisis?**


Esta herramienta identifica fármacos potenciales para una enfermedad basándose en su proximidad en la red de proteínas humanas.

Cuanto más cerca están las proteínas diana de un fármaco de las proteínas asociadas a la enfermedad, mayor es su potencial efecto terapéutico.

**Interpretación clínica:**
- No implica indicación aprobada, sino hipótesis de reposicionamiento.
- Prioriza candidatos para investigación

""")

st.sidebar.info("""
Aplicación basada en Medicina de Redes

Permite:
- Reposicionamiento de fármacos
- Análisis de posibles dianas terapéuticas
- Exploración del interactoma humano

No evalúa:
- Eficacia clínica directa
- Farmacocinética
- Toxicidad
- Ensayos clínicos

""")
#DATA_PATH = r"C:\Users\Nisrin Fariss Lamine\Downloads\tfm"

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
BIN_TAM =  50
N_MC = 200 
# ---------------------------------------------------------
# FUNCIONES
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df_edges = pd.read_csv(os.path.join(DATA_PATH, "biogrid_edges.csv"))
    G = nx.from_pandas_edgelist(df_edges, source='source', target='target')

    df_drug = pd.read_csv(os.path.join(DATA_PATH, "drugbank_targets_clean.csv"))

    df_disorders = pd.read_csv(
        os.path.join(DATA_PATH, "disorder_genes.csv"),
        sep=";"
    )

    return G, df_drug, df_disorders


G, df_drug, df_disorders = load_data()

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
@st.cache_data 
def construir_bins_por_grado(_grafo, tamaño_bin):
    grados = {n: _grafo.degree(n) for n in _grafo.nodes()} 
    bins = {} 
    for prot, g in grados.items(): 
        bin_id = g // tamaño_bin 
        bins.setdefault(bin_id, []).append(prot) 
    return grados, bins
grados, bins_grado = construir_bins_por_grado(G, BIN_TAM)
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
def visualize_network(G, disease_nodes, drug_targets, max_neighbors=20):
    net = Network(height="600px", width="100%", bgcolor="#ffffff")
    html_path = os.path.join(os.getcwd(), "network.html")
    net.set_options("""
    var options = {
      "nodes": {"shape": "dot","size": 15,"font": {"size": 16}},
      "edges": {"smooth": false},
      "physics": {
        "enabled": true,
        "stabilization": {"enabled": true, "iterations": 150},
        "barnesHut": {
            "gravitationalConstant": -1500,
            "centralGravity": 0.2,
            "springLength": 110
        },
        "timestep": 0.3,
        "minVelocity": 1.0
      }
    }
    """)

    disease_nodes = [n for n in disease_nodes if n in G]
    drug_targets = [n for n in drug_targets if n in G]

    nodes_to_show = set(disease_nodes + drug_targets)

    for n in list(nodes_to_show):
        if n in G:
            neighbors = list(G.neighbors(n))[:max_neighbors]
            nodes_to_show.update(neighbors)

    subG = G.subgraph(nodes_to_show)

    for node in subG.nodes():
        if node in disease_nodes:
            color = "#e74c3c"#rojo
        elif node in drug_targets:
            color = "#2980b9"#azul
        else:
            color = "#bdc3c7"#gris

        net.add_node(node, label=node, color=color)

    for u, v in subG.edges():
        net.add_edge(u, v)

    net.save_graph(html_path)

    return html_path
    
@st.cache_data
def map_drug_targets(df):
    return df.groupby("DrugBank_ID")["UniProt_ID"].apply(set).to_dict()
drug_targets = map_drug_targets(df_drug)

# ---------------------------------------------------------
# PESTAÑAS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Búsqueda por proteína",
    "Búsqueda por fármaco",
    "Búsqueda por Enfermedad ",
    "Información"
])


# ---------------------------------------------------------
# TAB 1 
# ---------------------------------------------------------
with tab1:
    st.subheader("Proteínas o genes asociados a la enfermedad")
    user_input = st.text_area("Símbolo de interés", height=120)
    if user_input:
        raw_items = [p.strip() for p in user_input.split(",") if p.strip()]
        def es_uniprot(x):
            return len(x) == 6 and x[0].isalpha() and x[1].isdigit()
        symbols = [x for x in raw_items if not es_uniprot(x)]
        uniprots = [x for x in raw_items if es_uniprot(x)]
        #conversion
        mapping = convert_gene_list(symbols) if symbols else {}
        converted_uniprots = [mapping[s] for s in symbols if mapping.get(s) is not None]
        disease_proteins = set(uniprots + converted_uniprots)
        if symbols:
            st.write("### Conversión símbolo → UniProt")
            for s in symbols:
                if mapping.get(s) is None:
                    st.error(f"{s}: NO encontrado")
                else:
                    st.success(f"{s} → {mapping[s]}")
        #filtro
        disease_proteins = disease_proteins.intersection(G.nodes())
        if len(disease_proteins) == 0:
            st.warning("Ninguna proteína coincide con el interactoma.")
            st.stop()
        with st.spinner("Calculando distancias desde el conjunto de enfermedad..."):
            dist_ref = bfs_multifuente(G, disease_proteins)
        resultados = []
        with st.spinner("Evaluando fármacos..."):
            for drug, targets in drug_targets.items():
                valids = list(targets.intersection(G.nodes()))
                if not valids:
                    continue
                d_obs, mu, sd, z, p = proximidad_estadistica(
                    valids, dist_ref, grados, bins_grado, repeticiones=N_MC, tamaño_bin=BIN_TAM
                )
                if d_obs is None:
                    continue
                resultados.append((drug, d_obs, z, p))
        ranking_est = pd.DataFrame(resultados, columns=["DrugBank_ID","Proximidad","Zscore","Pvalor"]).merge(
            df_drug[["DrugBank_ID","Drug_Name"]].drop_duplicates(),
            on="DrugBank_ID", how="left"
        ).sort_values(["Zscore","Pvalor"], ascending=[True, True])
        # Visualización del mejor por Z-score
        top_drug = ranking_est.iloc[0]["DrugBank_ID"]
        top_targets = drug_targets[top_drug]
        st.subheader("Visualización del mejor fármaco")
        html_file = visualize_network(G, disease_proteins, top_targets)
        with open(html_file, "r", encoding="utf-8") as f:
            components.html(f.read(), height=600)
        st.subheader("Top 20 fármacos por Z-score")
        st.dataframe(
            ranking_est[["Drug_Name","DrugBank_ID","Proximidad","Zscore","Pvalor"]].head(20),
            use_container_width=True, hide_index=True
        )
        st.download_button(
            "Descargar ranking (Z-score)",
            ranking_est.to_csv(index=False).encode(),
            "ranking_zscore.csv",
            "text/csv"
        )
# ---------------------------------------------------------
# TAB 2
# ---------------------------------------------------------
with tab2:
    st.subheader("Alternativas a un fármaco por proximidad de dianas")
    st.info("""
    Búsqueda de alternativas terapéuticas

    Este módulo identifica fármacos con mecanismos similares basados en sus dianas en la red.

    Útil cuando:
    - Un fármaco no funciona
    - Hay efectos adversos
    - Se buscan alternativas terapéuticas

    No implica equivalencia clínica directa.
    """)

    # Desplegable con lista de fármacos por nombre
    lista_farmacos = sorted(df_drug["Drug_Name"].dropna().unique())
    farmaco_nombre = st.selectbox("Selecciona un fármaco", lista_farmacos, index=None, placeholder="Buscar fármaco...")

    if st.button("Buscar alternativas"):
        if not farmaco_nombre:
            st.error("Selecciona un fármaco del desplegable.")
        else:
            ids_ref = df_drug.loc[df_drug["Drug_Name"] == farmaco_nombre, "DrugBank_ID"].dropna().unique()
            if len(ids_ref) == 0:
                st.error("No se encontró el ID de DrugBank para el fármaco seleccionado.")
            else:
                drug_id_ref = ids_ref[0]
                ref_targets = df_drug.loc[df_drug["Drug_Name"] == farmaco_nombre, "UniProt_ID"].dropna().unique().tolist()
                ref_targets = [t for t in ref_targets if t in G.nodes()]

                if len(ref_targets) == 0:
                    st.warning("El fármaco seleccionado no tiene dianas presentes en el interactoma.")
                else:
                    # distancias desde las dianas del fármaco de referencia
                    with st.spinner("Calculando distancias desde las dianas del fármaco de referencia..."):
                        dist_ref = bfs_multifuente(G, ref_targets)

                    resultados = []
                    with st.spinner("Evaluando fármacos ..."):
                        for drug_id, tg_set in drug_targets.items():
                            # omitir el propio fármaco de referencia en el ranking
                            if drug_id == drug_id_ref:
                                continue
                            dianas_validas = list(tg_set.intersection(G.nodes()))
                            if not dianas_validas:
                                continue

                            d_obs, mu, sd, z, p = proximidad_estadistica(
                                dianas_validas,
                                dist_ref,
                                grados,
                                bins_grado,
                                repeticiones=N_MC,
                                tamaño_bin=BIN_TAM
                            )
                            if d_obs is None:
                                continue
                            resultados.append((drug_id, d_obs, z, p))

                    if len(resultados) == 0:
                        st.warning("No se pudieron evaluar alternativas con los datos disponibles.")
                    else:
                        ranking_alt = pd.DataFrame(resultados, columns=["DrugBank_ID","Proximidad","Zscore","Pvalor"]).merge(
                            df_drug[["DrugBank_ID","Drug_Name"]].drop_duplicates(),
                            on="DrugBank_ID", how="left"
                        ).sort_values(["Zscore","Pvalor"], ascending=[True, True])

                        top_drug = ranking_alt.iloc[0]["DrugBank_ID"]
                        top_targets = drug_targets[top_drug]

                        st.subheader(f"Visualización — mejor candidato frente a {farmaco_nombre}")
                        html_file = visualize_network(G, ref_targets, top_targets)
                        with open(html_file, "r", encoding="utf-8") as f:
                            components.html(f.read(), height=600)

                        st.subheader("Top 20 alternativas por Z-score")
                        st.dataframe(
                            ranking_alt[["Drug_Name","DrugBank_ID","Proximidad","Zscore","Pvalor"]].head(20),
                            use_container_width=True, hide_index=True
                        )

                        st.download_button(
                            "Descargar alternativas (Z-score)",
                            ranking_alt.to_csv(index=False).encode(),
                            "alternativas_zscore.csv",
                            "text/csv"
                        )
                        
# ---------------------------------------------------------
# TAB 3
# ---------------------------------------------------------
with tab3:
    st.subheader("Explorar genes de una enfermedad")
    disorder = st.selectbox("Selecciona enfermedad", df_disorders["disorder"].dropna().unique())
    genes_str = df_disorders[df_disorders["disorder"] == disorder]["gene_symb"].values[0]
    genes_list = [g.strip() for g in genes_str.split(",") if g.strip()]
    gene = st.selectbox("Selecciona un gen", genes_list)
    mapping = convert_gene_list([gene])
    uniprot = mapping.get(gene)
    if uniprot is None:
        st.error("No se pudo convertir el gen a UniProt.")
    elif uniprot not in G.nodes():
        st.warning("El gen no está presente en el interactoma.")
    else:
        st.success(f"{gene} → {uniprot}")
        if st.button("Evaluar reposicionamiento para este gen"):
            with st.spinner("Calculando distancias desde el gen..."):
                dist_ref = bfs_multifuente(G, [uniprot])
            resultados = []
            with st.spinner("Evaluando fármacos..."):
                for d, tg in drug_targets.items():
                    valids = list(tg.intersection(G.nodes()))
                    if not valids:
                        continue
                    d_obs, mu, sd, z, p = proximidad_estadistica(
                        valids, dist_ref, grados, bins_grado, repeticiones=N_MC, tamaño_bin=BIN_TAM
                    )
                    if d_obs is None:
                        continue
                    resultados.append((d, d_obs, z, p))
            ranking_gene = pd.DataFrame(resultados, columns=["DrugBank_ID","Proximidad","Zscore","Pvalor"]).merge(
                df_drug[["DrugBank_ID","Drug_Name"]].drop_duplicates(),
                on="DrugBank_ID", how="left"
            ).sort_values(["Zscore","Pvalor"], ascending=[True, True])
            # Visualización del mejor por Z-score
            best = ranking_gene.iloc[0]
            st.subheader("Visualización del mejor candidato")
            html_file = visualize_network(G, [uniprot], drug_targets[best["DrugBank_ID"]])
            with open(html_file, "r", encoding="utf-8") as f:
                components.html(f.read(), height=600)
            st.subheader("Top 20 fármacos para este gen por Z-score")
            st.dataframe(
                ranking_gene[["Drug_Name","DrugBank_ID","Proximidad","Zscore","Pvalor"]].head(20),
                use_container_width=True, hide_index=True
            )
            st.download_button(
                "Descargar ranking por gen (Z-score)",
                ranking_gene.to_csv(index=False).encode(),
                "ranking_gen_zscore.csv",
                "text/csv"
            )
            
                
# ---------------------------------------------------------
# TAB 4
# ---------------------------------------------------------
with tab4: 
    st.subheader("¿Cómo funciona esta herramienta?")
    st.write("""
    Esta herramienta utiliza principios de **Medicina de Redes**:

    - Las enfermedades afectan módulos del interactoma.
    - Los fármacos son candidatos si sus dianas están **cerca** del módulo.
    - La proximidad se calcula mediante BFS multifuente.
    - También identifica:
        - Alternativas a un fármaco
        - Fármacos que actúan sobre vecinos
        - Conversión de símbolos → UniProt

    Bases de datos:
    - BioGRID → interacciones proteína-proteína  
    - DrugBank → dianas farmacológicas  
    - MyGene.info → conversión de genes  
    - Conjunto de datos aportados en la asignatura Análisis de Datos de Alta Dimensión y Medicina de Redes.

    
    **No implica equivalencia clínica directa.**

    """)