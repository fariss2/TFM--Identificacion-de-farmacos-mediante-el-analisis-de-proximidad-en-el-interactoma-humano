
import streamlit as st
import pandas as pd
import networkx as nx
import os
from collections import deque

from drug_similarity import rank_similar_drugs
from gene_to_uniprot import convert_gene_list

st.set_page_config(page_title="Drug Repurposing", layout="wide")
st.title("Reposicionamiento de fármacos mediante Medicina de Redes")

# CARGA DE DATOS
DATA_PATH = r"C:\Users\Nisrin Fariss Lamine\Downloads\tfm"

@st.cache_data
def load_data():
    df_edges = pd.read_csv(os.path.join(DATA_PATH, "biogrid_edges.csv"))
    G = nx.from_pandas_edgelist(df_edges, source='source', target='target')

    df_drug = pd.read_csv(os.path.join(DATA_PATH, "drugbank_targets_clean.csv"))
    return G, df_drug

G, df_drug = load_data()

st.sidebar.header("Información general")

# FUNCIONES 
def multi_source_bfs(G, sources):
    """ BFS desde múltiples proteínas a la vez (más rápido) """
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
        .agg({
            "Drug_Name": "first",
            "UniProt_ID": set
        })
        .reset_index()
        .rename(columns={"UniProt_ID": "Targets_in_neighbors"})
    )

#PESTAÑAS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Ranking de fármacos (proximidad)",
    " Alternativas a un fármaco",
    " Conversión símbolo → UniProt",
    " Fármacos sobre vecinos",
    " Información y metodología"
])

#— PROXIMIDAD ENFERMEDAD → FÁRMACOS
with tab1:
    st.subheader("Introduce proteínas asociadas a la enfermedad")

    user_input = st.text_area(
        "Puedes introducir símbolos (TP53) o UniProt (P04637)",
        height=120
    )

    if user_input:

        # Separación y detección 
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

        # Mostrar conversiones 
        if symbols:
            st.write("### Conversión símbolo → UniProt:")
            for s in symbols:
                if mapping[s] is None:
                    st.error(f"{s}: NO encontrado")
                else:
                    st.success(f"{s} → {mapping[s]}")

        #Filtrar por red
        disease_proteins = disease_proteins.intersection(G.nodes())
        st.write(f"Proteínas válidas: **{len(disease_proteins)}**")

        if len(disease_proteins) == 0:
            st.warning("Ninguna coincidencia.")
            st.stop()

        # Distancias 
        with st.spinner("Calculando distancias en la red..."):
            dist_to_disease = multi_source_bfs(G, disease_proteins)

        #  Targets de fármacos 
        drug_targets = (
            df_drug
            .groupby('DrugBank_ID')['UniProt_ID']
            .apply(set)
            .to_dict()
        )

        # Función proximidad 
        def proximity(targets):
            dists = [
                dist_to_disease.get(t, float("inf"))
                for t in targets
                if t in G.nodes()
            ]
            if len(dists) == 0:
                return None
            return sum(dists) / len(dists)

        #  Calcular ranking
        with st.spinner("Calculando proximidad de fármacos..."):
            results = []
            for drug, targets in drug_targets.items():
                valid_targets = targets.intersection(G.nodes())
                if len(valid_targets) == 0:
                    continue
                score = proximity(valid_targets)
                if score is not None:
                    results.append((drug, score))

        ranking = pd.DataFrame(results, columns=["DrugBank_ID", "Proximity"])
        ranking = ranking.sort_values("Proximity")

        # Añadir nombres
        drug_names = df_drug[['DrugBank_ID', 'Drug_Name']].drop_duplicates()
        ranking = ranking.merge(drug_names, on="DrugBank_ID", how="left")

        #OUTPUT
        st.subheader("Top fármacos candidatos")
        st.dataframe(
            ranking[['Drug_Name', 'DrugBank_ID', 'Proximity']].head(20),
            use_container_width=True
        )

        # Descarga
        csv = ranking.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar datos",
            data=csv,
            file_name="drug_ranking.csv",
            mime="text/csv"
        )

#— ALTERNATIVAS A UN FÁRMACO
with tab2:
    st.subheader("Buscar alternativas basadas en red")

    drug_query = st.text_input(
        "Introduce el nombre del fármaco"
    )

    if st.button("Buscar alternativas"):
        try:
            drug_id, targets, ranking = rank_similar_drugs(G, df_drug, drug_query)

            st.write(f"Fármaco: **{drug_query}** (ID: {drug_id})")
            st.write(f"Dianas → {', '.join(targets)}")

            st.subheader("Fármacos similares")

            st.dataframe(
                ranking[["Drug_Name", "DrugBank_ID", "Proximity"]].head(20),
                use_container_width=True
            )

            csv2 = ranking.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Descargar ranking de alternativas",
                csv2,
                "drug_alternatives.csv",
                "text/csv"
            )

        except ValueError as e:
            st.error(str(e))

#CONVERSIÓN SÍMBOLO → UNIPROT
with tab3:
    st.subheader("Conversión de símbolos de genes a UniProt")

    user_genes = st.text_input("Ejemplo: TP53, EGFR, BRCA1...")

    if st.button("Convertir"):
        if user_genes.strip():
            genes = [g.strip().upper() for g in user_genes.split(",")]
            mapping = convert_gene_list(genes)

            st.write("### Resultados:")
            for g, u in mapping.items():
                if u is None:
                    st.error(f"{g}: No encontrado")
                else:
                    st.success(f"{g} → {u}")

#— FÁRMACOS SOBRE VECINOS
with tab4:
    st.subheader("Fármacos que actúan sobre vecinos de una proteína")

    protein_input = st.text_input("Introduce UniProt (ej: P04637)")

    if st.button("Buscar vecinos"):
        if protein_input not in G.nodes():
            st.error("Proteína no encontrada en el interactoma.")
        else:
            neighbors = list(G.neighbors(protein_input))

            st.write(f"Vecinos directos: **{len(neighbors)}**")
            st.write(neighbors[:50])

            result = drugs_targeting_proteins(neighbors, df_drug)

            st.subheader("Fármacos que actúan sobre los vecinos")

            if result.empty:
                st.warning("No se encontraron fármacos.")
            else:
                result["Num_targets"] = result["Targets_in_neighbors"].apply(len)
                result = result.sort_values("Num_targets", ascending=False)

                st.dataframe(
                    result[["Drug_Name", "DrugBank_ID", "Num_targets", "Targets_in_neighbors"]],
                    use_container_width=True
                )

#— INFORMACIÓN Y METODOLOGÍA
with tab5:
    st.subheader("¿Cómo funciona esta herramienta?")
    st.write("""
    Esta aplicación utiliza principios de **Medicina de Redes**:

    - Las enfermedades alteran módulos completos del interactoma.
    - Las proteínas asociadas a la enfermedad definen una región de interés.
    - Un fármaco es candidato si sus dianas están **cerca de ese módulo**.
    - La proximidad se estima como distancia mínima promedio.
    - También se identifican:
        - Alternativas a un fármaco (targets cercanos en red)
        - Fármacos que actúan sobre vecinos de proteínas clave
        - Conversión de nombres clínicos → UniProt

    Las bases de datos usadas son:
    - **BioGRID** para el interactoma
    - **DrugBank** para relaciones fármaco-diana
    - **DisGeNet** proximamente...
    - **MyGene.info** para convertir símbolos a UniProt
    -

    """)

