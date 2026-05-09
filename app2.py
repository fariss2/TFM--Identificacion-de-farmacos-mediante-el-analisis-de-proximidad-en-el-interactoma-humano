
import streamlit as st
import pandas as pd
import networkx as nx
import os
from collections import deque
from pyvis.network import Network
import streamlit.components.v1 as components

from drug_similarity import rank_similar_drugs
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
- No implica indicación aprobada, sino hipótesis de reposicionamiento

""")

st.sidebar.info("""
Aplicación basada en Medicina de Redes

Permite:
- Reposicionamiento de fármacos
- Análisis de dianas terapéuticas
- Exploración del interactoma humano
""")
DATA_PATH = r"C:\Users\Nisrin Fariss Lamine\Downloads\tfm"

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


def pagerank_propio(G, alpha=0.85, max_iter=100, tol=1e-06):
    # Lista de nodos
    nodes = list(G.nodes())
    N = len(nodes)

    # PageRank inicial
    pr = {n: 1.0 / N for n in nodes}

    # Grado de salida (en grafo no dirigido es el grado normal)
    out_degree = {n: len(list(G.neighbors(n))) for n in nodes}

    # Evitar divisiones por cero
    for n in nodes:
        if out_degree[n] == 0:
            out_degree[n] = 1

    # Iteración principal
    for _ in range(max_iter):
        new_pr = {}
        diff = 0

        for n in nodes:
            rank_sum = 0
            for nbr in G.neighbors(n):
                rank_sum += pr[nbr] / out_degree[nbr]

            new_pr[n] = (1 - alpha) / N + alpha * rank_sum

            diff += abs(new_pr[n] - pr[n])

        pr = new_pr

        # Criterio de convergencia
        if diff < tol:
            break

    return pr

def multi_source_bfs(G, sources):
    """ BFS desde múltiples nodos (más rápido que buscar uno a uno) """
    dist = {n: float("inf") for n in G.nodes()}
    queue = deque()

    for s in sources:
        dist[s] = 0
        queue.append(s)

    while queue:
        v = queue.popleft()
        for neigh in G.neighbors(v):
            if dist[neigh] == float("inf"):
                dist[neigh] = dist[v] + 1
                queue.append(neigh)

    return dist


def drugs_targeting_proteins(proteins, df_drug):
    """ Fármacos que actúan sobre un conjunto de proteínas """
    df = df_drug[df_drug["UniProt_ID"].isin(proteins)]
    return (
        df.groupby("DrugBank_ID")
        .agg({"Drug_Name": "first", "UniProt_ID": set})
        .reset_index()
        .rename(columns={"UniProt_ID": "Targets_in_neighbors"})
    )
def visualize_network(G, disease_nodes, drug_targets, max_neighbors=20):

    net = Network(height="600px", width="100%", bgcolor="#ffffff")
    html_path = os.path.join(os.getcwd(), "network.html")

    # Mejorar física y visual
    net.set_options("""
    var options = {
      "nodes": {
        "shape": "dot",
        "size": 15,
        "font": {"size": 16}
      },
      "edges": {
        "smooth": false
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.3,
          "springLength": 95
        },
        "minVelocity": 0.75
      }
    }
    """)

    # Nodos a mostrar
    nodes_to_show = set()

    # Enfermedad
    nodes_to_show.update(disease_nodes)

    # Dianas del fármaco top
    nodes_to_show.update(drug_targets)

    # Vecinos limitados
    for n in list(nodes_to_show):
        neighbors = list(G.neighbors(n))[:max_neighbors]
        nodes_to_show.update(neighbors)

    subG = G.subgraph(nodes_to_show)

    # Añadir nodos
    for node in subG.nodes():
        if node in disease_nodes:
            color = "#e74c3c"  # rojo
        elif node in drug_targets:
            color = "#2980b9"  # azul
        else:
            color = "#bdc3c7"  # gris

        net.add_node(node, label=node, color=color)

    # Añadir aristas
    for u, v in subG.edges():
        net.add_edge(u, v)

    net.save_graph(html_path)
    return html_path

pagerank_scores = pagerank_propio(G)


# ---------------------------------------------------------
# PESTAÑAS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Búsqueda por proteína",
    "Búsqueda por fármaco",
    "Búsqueda por Enfermedad ",
    "Conversión a UniProt",
    "Información"
])


# ---------------------------------------------------------
# TAB 1 
# ---------------------------------------------------------
with tab1:
    st.markdown("### Proteínas o genes asociados a la enfermedad")

    user_input = st.text_area(
        "Simbolo de interés.",
        height=120
    )

    if user_input:

        raw_items = [p.strip() for p in user_input.split(",") if p.strip()]

        def is_uniprot(x):
            return len(x) == 6 and x[0].isalpha() and x[1].isdigit()

        symbols = [x for x in raw_items if not is_uniprot(x)]
        uniprots = [x for x in raw_items if is_uniprot(x)]

        # Conversión
        mapping = convert_gene_list(symbols) if symbols else {}

        converted_uniprots = [
            mapping[s] for s in symbols if mapping[s] is not None
        ]

        disease_proteins = set(uniprots + converted_uniprots)

        # Mostrar conversión
        if symbols:
            st.write("### Conversión símbolo → UniProt")
            for s in symbols:
                if mapping[s] is None:
                    st.error(f"{s}: NO encontrado")
                else:
                    st.success(f"{s} → {mapping[s]}")

        # Filtrar por red
        disease_proteins = disease_proteins.intersection(G.nodes())

        st.write(f"Proteínas válidas: **{len(disease_proteins)}**")
        if len(disease_proteins) == 0:
            st.warning("Ninguna proteína coincide con el interactoma.")
            st.stop()

        # Distancias
        with st.spinner("Calculando distancias..."):
            dist_to_disease = multi_source_bfs(G, disease_proteins)

        # Targets de fármacos
        drug_targets = (
            df_drug.groupby("DrugBank_ID")["UniProt_ID"]
            .apply(set)
            .to_dict()
        )

        def proximidad(targets):
            dists = [
                dist_to_disease.get(t, float("inf"))
                for t in targets
            ]
            dists = [d for d in dists if d != float("inf")]
            if len(dists) == 0:
                return None
            return sum(dists) / len(dists)

        # Ranking
        with st.spinner("Evaluando fármacos..."):
            results = []
            for drug, targets in drug_targets.items():
                valid_targets = targets.intersection(G.nodes())
                if len(valid_targets) == 0:
                    continue
                score = proximidad(valid_targets)
                if score is not None:
                    results.append((drug, score))

        ranking = pd.DataFrame(results, columns=["DrugBank_ID", "Proximidad"])
        ranking = ranking.sort_values("Proximidad")
 

        def drug_pagerank(targets):
            values = [pagerank_scores[t] for t in targets if t in pagerank_scores]
            return sum(values) / len(values) if values else 0

        ranking["PageRank"] = ranking["DrugBank_ID"].apply(
            lambda d: drug_pagerank(drug_targets[d])
        )

        
        ranking["Combinación"] = (#por docs 
            ranking["Proximidad"].rank(method="dense") * 0.7 +
            ranking["PageRank"].rank(method="dense", ascending=False) * 0.3
        )

       
        


        # Añadir nombres
        ranking = ranking.merge(
            df_drug[['DrugBank_ID', 'Drug_Name']].drop_duplicates(),
            on="DrugBank_ID",
            how="left"
        )
        
        st.info("""
        **Visualización de red:**

        - 🔴 Proteínas de la enfermedad  
        - 🔵 Dianas del fármaco  
        """)
        # VISUALIZACIÓN
        top_drug = ranking.iloc[0]["DrugBank_ID"]
        top_targets = drug_targets[top_drug]

        st.subheader("Visualización del fármaco mejor posicionado")
        html_file = visualize_network(G, disease_proteins, top_targets)

        with open(html_file, "r", encoding="utf-8") as f:
            components.html(f.read(), height=600)
        
        st.info("""
        **Interpretación del ranking:**

        - Cada fila representa un fármaco, los primeros fármacos son los más relevantes

        Este resultado no sustituye evidencia clínica.
        """)
        
        # TABLA
        st.subheader("Top 20 fármacos (Proximidad + PageRank)")

        st.dataframe(
            ranking[["Drug_Name", "DrugBank_ID", "Proximidad", "PageRank", "Combinación"]]
            .sort_values("Combinación")
            .head(20),
            use_container_width=True,
            hide_index=True
        )

        best = ranking.sort_values("Combinación").iloc[0]

        st.markdown("## Interpretación clínica del modelo combinado")

        st.success(f"""
        **Fármaco prioritario (modelo combinado): _{best['Drug_Name']}_**

        **Proximidad:** {best['Proximidad']:.2f}  
        **PageRank:** {best['PageRank']:.5f}

        ### Por qué este fármaco es relevante:
        - Está cerca de los genes implicados en la enfermedad  
        - Sus dianas se sitúan en proteínas **centrales del interactoma**, lo que implica mayor impacto regulatorio  
        - La combinación de métricas reduce falsos positivos e identifica fármacos:
        
            – Cercanos a la enfermedad (efecto directo)
            
            – Influyentes en la red biológica (efecto sistémico).

        Este resultado fortalece la priorización para análisis preclínicos.
        """)

        st.subheader("Análisis rápido de vecinos")
        st.info("""
        **Análisis de vecindario:**
        
        Explora qué fármacos actúan sobre proteínas cercanas a una proteína de interés.
        Útil para identificar dianas indirectas.
        """)
      

        protein_single = st.text_input("Selecciona una proteína para analizar su vecindario (UniProt ID)")

        if st.button("Buscar fármacos que actúan sobre los vecinos", key="neighbors_from_tab1"):
            if protein_single not in G.nodes():
                st.error("La proteína no está en el interactoma.")
            else:
                neighbors = list(G.neighbors(protein_single))
                st.write(f"Vecinos encontrados: {len(neighbors)}")

                result = drugs_targeting_proteins(neighbors, df_drug)

                if result.empty:
                    st.warning("No se encontraron fármacos para esos vecinos.")
                else:
                    result["Num_targets"] = result["Targets_in_neighbors"].apply(len)
                    result = result.sort_values("Num_targets", ascending=False)
                    st.dataframe(result, use_container_width=True, hide_index=True)


        st.download_button(
            "Descargar ranking",
            ranking.to_csv(index=False).encode(),
            "drug_ranking.csv",
            "text/csv"
        )


# ---------------------------------------------------------
# TAB 2
# ---------------------------------------------------------
with tab2:
    st.subheader("Alternativas a un fármaco")
    st.info("""
    **Búsqueda de alternativas terapéuticas**

    Este módulo identifica fármacos con mecanismos similares basados en sus dianas en la red.

    Útil cuando:
    - Un fármaco no funciona
    - Hay efectos adversos
    - Se buscan alternativas terapéuticas

    No implica equivalencia clínica directa.
    """)
    
    drug_list = sorted(df_drug["Drug_Name"].dropna().unique())

    drug_query = st.selectbox(
        "Selecciona un fármaco",
        drug_list,
        index=None,
        placeholder="Buscar fármaco..."
    )


    if st.button("Buscar alternativas"):
        try:
            drug_id, targets, ranking = rank_similar_drugs(G, df_drug, drug_query)

            st.write(f"ID del fármaco: **{drug_id}**")
            st.write(f"Dianas: {', '.join(targets)}")

            st.dataframe(
                ranking[["Drug_Name", "DrugBank_ID", "Proximidad"]].head(20),
                use_container_width=True,hide_index=True
            )

            st.download_button(
                "Descargar alternativas",
                ranking.to_csv(index=False).encode(),
                "drug_alternatives.csv",
                "text/csv"
            )

        except ValueError as e:
            st.error(str(e))



# ---------------------------------------------------------
# TAB 3
# ---------------------------------------------------------
with tab3:
    st.subheader("Explorar genes de una enfermedad")

    disorder = st.selectbox(
        "Selecciona enfermedad",
        df_disorders["disorder"].dropna().unique()
    )

    # Obtener genes asociados a la enfermedad
    genes_str = df_disorders[df_disorders["disorder"] == disorder]["gene_symb"].values[0]
    genes_list = [g.strip() for g in genes_str.split(",") if g.strip()]

    gene = st.selectbox("Selecciona un gen", genes_list)

    st.info(f"Gen seleccionado: **{gene}**")

    # Conversión símbolo → UniProt
    mapping = convert_gene_list([gene])
    uniprot = mapping.get(gene)

    # Validaciones
    if uniprot is None:
        st.error("No se pudo convertir el gen seleccionado a UniProt.")
    elif uniprot not in G.nodes():
        st.warning("El gen no está presente en el interactoma.")
    else:
        st.success(f"{gene} → {uniprot}")

        # ==========================================
        #   BOTÓN: REPOSICIONAMIENTO PARA ESTE GEN
        # ==========================================
        if st.button("Evaluar reposicionamiento para este gen"):
            
            # Distancia desde el gen (BFS)
            with st.spinner("Calculando proximidad del gen…"):
                dist_single = multi_source_bfs(G, [uniprot])

            # Targets de fármacos
            drug_targets = (
                df_drug.groupby("DrugBank_ID")["UniProt_ID"]
                .apply(set)
                .to_dict()
            )

            def proximity_single(targets):
                d = [dist_single.get(t, float("inf")) for t in targets]
                d = [x for x in d if x != float("inf")]
                return sum(d) / len(d) if d else None

            # Ranking por proximidad
            results = []
            for d, targets in drug_targets.items():
                valid_targets = targets.intersection(G.nodes())
                if len(valid_targets) == 0:
                    continue
                score = proximity_single(valid_targets)
                if score is not None:
                    results.append((d, score))

            ranking_gene = pd.DataFrame(results, columns=["DrugBank_ID", "Proximidad"])

            ranking_gene = ranking_gene.merge(
                df_drug[["DrugBank_ID", "Drug_Name"]].drop_duplicates(),
                on="DrugBank_ID"
            )

            # ============================
            #   PageRank por fármaco
            # ============================
            def drug_pagerank(targets):
                values = [pagerank_scores[t] for t in targets if t in pagerank_scores]
                return sum(values) / len(values) if values else 0

            ranking_gene["PageRank"] = ranking_gene["DrugBank_ID"].apply(
                lambda d: drug_pagerank(drug_targets[d])
            )

            # ============================
            #   Score combinado
            # ============================

           
            ranking_gene["Combinación"] = (
                ranking_gene["Proximidad"].rank(method="dense") * 0.7 +
                ranking_gene["PageRank"].rank(method="dense", ascending=False) * 0.3
            )

            # Seleccionar mejor fármaco
            best = ranking_gene.sort_values("Combinación").iloc[0]

            # ============================
            #   INTERPRETACIÓN CLÍNICA
            # ============================
            st.markdown("## Interpretación clínica")
            st.success(f"""
            **Fármaco prioritario para el gen {gene}: _{best['Drug_Name']}_**

            - Proximidad: **{best['Proximidad']:.2f}**  
            - PageRank: **{best['PageRank']:.4f}**  
            - Score combinado: **{best['Combinación']:.2f}**

            El fármaco actúa sobre dianas próximas al gen y situadas en zonas centrales del interactoma.  
            Esto refuerza la hipótesis de reposicionamiento.
            """)

            # ============================
            #   VISUALIZACIÓN DE RED
            # ============================
            st.subheader("Visualización del fármaco prioritario")

            html_file = visualize_network(
                G,
                [uniprot],
                drug_targets[best["DrugBank_ID"]]
            )

            with open(html_file, "r", encoding="utf-8") as f:
                components.html(f.read(), height=600)

            # ============================
            #   TABLA DE RESULTADOS
            # ============================
            st.subheader("Top 20 fármacos para este gen")
            st.dataframe(
                ranking_gene.sort_values("Combinación").head(20),
                use_container_width=True,hide_index=True
            )
# ---------------------------------------------------------
# TAB 4
# ---------------------------------------------------------
with tab4:
    st.subheader("Conversión de símbolos → UniProt")
    st.info("""
    **Conversión de identificadores**

    Convierte genes a identificadores UniProt usados en la red.

    Necesario para integrar datos biológicos.
    """)

    genes = st.text_input("Introduce gen")

    if st.button("Convertir símbolos"):
        lst = [g.strip().upper() for g in genes.split(",")]
        mapping = convert_gene_list(lst)

        st.write("### Resultados:")
        for g, u in mapping.items():
            if u is None:
                st.error(f"{g}: No encontrado")
            else:
                st.success(f"{g} → {u}")
# ---------------------------------------------------------
# TAB 5
# ---------------------------------------------------------
with tab5:
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
    - BioGRID 
    - DrugBank 
    - MyGene.info
    
    **No implica equivalencia clínica directa.**

    """)

