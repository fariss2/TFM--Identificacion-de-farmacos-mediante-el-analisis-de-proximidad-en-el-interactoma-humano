import streamlit as st
import pandas as pd
import networkx as nx
import os
import numpy as np
import random
from collections import deque
from pyvis.network import Network
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import plotly.express as px  
from gene_to_uniprot import convert_gene_list
from network_medicine import bfs_multifuente,proximidad_estadistica, generar_dianas_aleatorias,distancia_media_conjunto
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

st.sidebar.image(
    "portada.png", 
    use_container_width=True 
)

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

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
BIN_TAM =  50
# Modificado por Nacho: se aumenta el numero de simulaciones Monte Carlo para estabilizar z-score y p-valor.
N_MC = 1000 
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

@st.cache_data 
def construir_bins_por_grado(_grafo, tamaño_bin):
    grados = {n: _grafo.degree(n) for n in _grafo.nodes()} 
    bins = {} 
    for prot, g in grados.items(): 
        bin_id = g // tamaño_bin 
        bins.setdefault(bin_id, []).append(prot) 
    return grados, bins
grados, bins_grado = construir_bins_por_grado(G, BIN_TAM)
@st.cache_data
def map_drug_targets(df):
    return df.groupby("DrugBank_ID")["UniProt_ID"].apply(set).to_dict()
drug_targets = map_drug_targets(df_drug)
# ---------------------------------------------------------
# VISUALIZACION
# ---------------------------------------------------------
def visualize_context_network(
    G,
    context_nodes,
    drug_targets,
    context_label,
    drug_label,
    max_neighbors=12,
):
    net = Network(height="650px", width="100%", bgcolor="#ffffff")
    html_path = os.path.join(os.getcwd(), "network_context.html")
    net.set_options("""
    var options = {
      "nodes": {
        "font": {"size": 15},
        "borderWidth": 2,
        "shadow": false
      },
      "edges": {
        "smooth": false,
        "color": {"inherit": false}
      },
      "physics": {
        "enabled": true,
        "stabilization": {"enabled": true, "iterations": 200},
        "barnesHut": {
          "gravitationalConstant": -2200,
          "centralGravity": 0.25,
          "springLength": 125,
          "springConstant": 0.04
        },
        "timestep": 0.3,
        "minVelocity": 1.0
      }
    }
    """)

    context_nodes = [n for n in context_nodes if n in G]
    drug_targets = [n for n in drug_targets if n in G]
    context_set = set(context_nodes)
    target_set = set(drug_targets)

    nodes_to_show = context_set | target_set
    for node in list(nodes_to_show):
        neighbors = list(G.neighbors(node))[:max_neighbors]
        nodes_to_show.update(neighbors)

    context_id = "__context_node__"
    drug_id = "__drug_node__"

    net.add_node(
        context_id,
        label=context_label,
        title=f"{context_label}\nOrigen de referencia",
        color="#e74c3c",
        shape="diamond",
        size=34,
        physics=True,
    )
    net.add_node(
        drug_id,
        label=drug_label,
        title=f"{drug_label}\nFarmaco candidato",
        color="#2980b9",
        shape="star",
        size=34,
        physics=True,
    )

    for node in nodes_to_show:
        is_context = node in context_set
        is_target = node in target_set

        if is_context and is_target:
            color = "#8e44ad"
            role = "proteina de referencia y diana del farmaco"
        elif is_context:
            color = "#e74c3c"
            role = "proteina de referencia"
        elif is_target:
            color = "#2980b9"
            role = "diana del farmaco"
        else:
            color = "#bdc3c7"
            role = "vecino en el interactoma"

        net.add_node(
            node,
            label=node,
            title=(
                f"{node}\n"
                f"Rol: {role}\n"
                f"Grado en interactoma: {G.degree(node)}"
            ),
            color=color,
            shape="dot",
            size=18 if is_context or is_target else 11,
        )

    subG = G.subgraph(nodes_to_show)
    for u, v in subG.edges():
        net.add_edge(u, v, color="#95a5a6", width=1)

    for node in context_nodes:
        if node in nodes_to_show:
            net.add_edge(context_id, node, color="#e74c3c", dashes=True, width=2)

    for node in drug_targets:
        if node in nodes_to_show:
            net.add_edge(drug_id, node, color="#2980b9", dashes=True, width=2)

    net.save_graph(html_path)

    return html_path
# ---------------------------------------------------------
# PESTAÑAS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Búsqueda por Enfermedad",
    "Búsqueda por Fármaco",
    "Búsqueda por Proteína ",
    "Información"
])


# ---------------------------------------------------------
# TAB 1 
# ---------------------------------------------------------

with tab1:
    st.subheader("Explorar genes de una enfermedad")
    disorder = st.selectbox("Selecciona enfermedad", df_disorders["disorder"].dropna().unique())
    genes_str = df_disorders[df_disorders["disorder"] == disorder]["gene_symb"].values[0]
    genes_list = [g.strip() for g in genes_str.split(",") if g.strip()]
    gene = st.selectbox("Selecciona un gen", genes_list)
    mapping = convert_gene_list([gene])
    # Modificado por Nacho: la conversion puede devolver varias accesiones UniProt para un mismo gen
    uniprots = mapping.get(gene, [])
    uniprots_validos = [u for u in uniprots if u in G.nodes()]
    uniprot = ", ".join(uniprots_validos)
    if not uniprots:
        st.error("No se pudo convertir el gen a UniProt.")
    elif not uniprots_validos:
        st.warning("El gen no está presente en el interactoma.")
    else:
        st.success(f"{gene} → {uniprot}")
        if st.button("Evaluar reposicionamiento para este gen"):
            with st.spinner("Calculando distancias desde el gen..."):
                # Modificado por Nacho: BFS multifuente desde todas las accesiones validas del gen
                dist_ref = bfs_multifuente(G, uniprots_validos)
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
                    resultados.append((d, d_obs, mu,sd, z, p))
            ranking_gene = pd.DataFrame(resultados, columns=["DrugBank_ID","Proximidad","Media nula", "Desviación nula","Zscore","Pvalor"]).merge(
                df_drug[["DrugBank_ID","Drug_Name"]].drop_duplicates(),
                on="DrugBank_ID", how="left"
            ).sort_values(["Zscore","Pvalor"], ascending=[True, True])

            

                        

            # Visualización del mejor por Z-score
            # Modificado por Nacho: se evita acceder a la primera fila si no hay farmacos evaluables
            if ranking_gene.empty:
                st.warning("No se encontraron farmacos evaluables para este gen.")
                st.stop()
            best = ranking_gene.iloc[0]
            
            st.subheader("Top 20 fármacos para este gen por Z-score")
            st.dataframe(
                ranking_gene[["Drug_Name","DrugBank_ID","Proximidad","Media nula","Desviación nula","Zscore","Pvalor"]].head(20),
                use_container_width=True, hide_index=True
            )

            st.subheader("Visualización del candidato más priorizado por la red")
            # Modificado por Nacho: se usa la nueva red contextual para conectar gen y farmaco candidato
            html_file = visualize_context_network(
                G,
                uniprots_validos,
                drug_targets[best["DrugBank_ID"]],
                f"{gene}",
                f"{best['Drug_Name']} ({best['DrugBank_ID']})",
            )
            with open(html_file, "r", encoding="utf-8") as f:
                components.html(f.read(), height=600)

                
            st.subheader("Dispersión: Z-score vs P-valor")
            
            top20_ids = set(ranking_gene.nsmallest(20, "Zscore")["DrugBank_ID"])
            
            # columna para la clasificación 
            ranking_gene['Clasificación'] = 'Otros fármacos'
            ranking_gene.loc[ranking_gene['DrugBank_ID'].isin(top20_ids), 'Clasificación'] = 'Top 20 fármacos'
            
            #  -log10(Pvalor) para el eje X para q no se solapen cerca del cero
            ranking_gene["logP"] = -np.log10(ranking_gene["Pvalor"])
            
            fig_px = px.scatter(
                ranking_gene,
                x="logP",
                y="Zscore",
                color="Clasificación",
                color_discrete_map={
                    'Top 20 fármacos': 'red',
                    'Otros fármacos': 'gray'
                },
                #  raton
                hover_name="Drug_Name", 
                hover_data={
                    'DrugBank_ID': True,  
                    'Zscore': ':.2f',    
                    'Pvalor': ':.4f',     
                    'logP': False,        
                    'Clasificación': False 
                },
                labels={
                    "logP": "-log10(P-valor)",
                    "Zscore": "Z-score"
                },
                template="plotly_white"
            )
            
            fig_px.add_hline(y=0, line_dash="dash", line_color="black")
            
            st.plotly_chart(fig_px, use_container_width=True)

            
            st.download_button(
                "Descargar ranking por gen (Z-score)",
                ranking_gene.to_csv(index=False).encode(),
                "ranking_gen_zscore.csv",
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
                            resultados.append((drug_id, d_obs,mu,sd, z, p))
                        

                    if len(resultados) == 0:
                        st.warning("No se pudieron evaluar alternativas con los datos disponibles.")
                    else:
                        ranking_alt = pd.DataFrame(resultados, columns=["DrugBank_ID","Proximidad","Media nula","Desviación nula","Zscore","Pvalor"]).merge(
                            df_drug[["DrugBank_ID","Drug_Name"]].drop_duplicates(),
                            on="DrugBank_ID", how="left"
                        ).sort_values(["Zscore","Pvalor"], ascending=[True, True])

                        top_drug = ranking_alt.iloc[0]["DrugBank_ID"]
                        top_targets = drug_targets[top_drug]
                        # Modificado por Nacho: se recupera el nombre para etiquetar el nodo del farmaco candidato
                        top_drug_name = ranking_alt.iloc[0]["Drug_Name"]
                        


                        

                        st.subheader("Top 20 alternativas por Z-score")
                        st.dataframe(
                            ranking_alt[["Drug_Name","DrugBank_ID","Proximidad","Media nula","Desviación nula", "Zscore","Pvalor"]].head(20),
                            use_container_width=True, hide_index=True
                        )


                        st.subheader(f"Visualización — candidato  frente a {farmaco_nombre}")
                        # Modificado por Nacho: se visualiza el farmaco de referencia y el candidato como nodos propios
                        html_file = visualize_context_network(
                            G,
                            ref_targets,
                            top_targets,
                            f"{farmaco_nombre} ({drug_id_ref})",
                            f"{top_drug_name} ({top_drug})",
                        )
                        with open(html_file, "r", encoding="utf-8") as f:
                            components.html(f.read(), height=600)


                            
                        st.subheader("Dispersión: Z-score vs P-valor")
            
                        top20_ids = set(ranking_alt.nsmallest(20, "Zscore")["DrugBank_ID"])
                        
                        # columna para la clasificación 
                        ranking_alt['Clasificación'] = 'Otros fármacos'
                        ranking_alt.loc[ranking_alt['DrugBank_ID'].isin(top20_ids), 'Clasificación'] = 'Top 20 fármacos'
                        
                        #  -log10(Pvalor) para el eje X para q no se solapen cerca del cero
                        ranking_alt["logP"] = -np.log10(ranking_alt["Pvalor"])
                        
                        fig_px = px.scatter(
                            ranking_alt,
                            x="logP",
                            y="Zscore",
                            color="Clasificación",
                            color_discrete_map={
                                'Top 20 fármacos': 'red',
                                'Otros fármacos': 'gray'
                            },
                            #  raton
                            hover_name="Drug_Name", 
                            hover_data={
                                'DrugBank_ID': True,  
                                'Zscore': ':.2f',    
                                'Pvalor': ':.4f',     
                                'logP': False,        
                                'Clasificación': False 
                            },
                            labels={
                                "logP": "-log10(P-valor)",
                                "Zscore": "Z-score"
                            },
                            template="plotly_white"
                        )
                        
                        fig_px.add_hline(y=0, line_dash="dash", line_color="black")
                        
                        st.plotly_chart(fig_px, use_container_width=True)
                        

                        st.download_button(
                            "Descargar alternativas ",
                            ranking_alt.to_csv(index=False).encode(),
                            "alternativas_zscore.csv",
                            "text/csv"
                        )
                        
# ---------------------------------------------------------
# TAB 3
# ---------------------------------------------------------
          
with tab3:
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
        #converted_uniprots = [mapping[s] for s in symbols if mapping.get(s) is not None]
        converted_uniprots = []
        for s in symbols:
            vals = mapping.get(s)
            if vals is None:
                continue
            if isinstance(vals, list):
                converted_uniprots.extend(vals)
            else:
                converted_uniprots.append(vals)
                
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
                resultados.append((drug, d_obs,mu,sd, z, p))
        ranking_est = pd.DataFrame(resultados, columns=["DrugBank_ID","Proximidad","Media nula","Desviación nula","Zscore","Pvalor"]).merge(
            df_drug[["DrugBank_ID","Drug_Name"]].drop_duplicates(),
            on="DrugBank_ID", how="left"
        ).sort_values(["Zscore","Pvalor"], ascending=[True, True])
        # Visualización del mejor por Z-score
        top_drug = ranking_est.iloc[0]["DrugBank_ID"]
        top_targets = drug_targets[top_drug]
        # Modificado por Nacho: se recupera el nombre para etiquetar el nodo del farmaco candidato.
        top_drug_name = ranking_est.iloc[0]["Drug_Name"]
        


        


        
        with open(html_file, "r", encoding="utf-8") as f:
            components.html(f.read(), height=600)
        st.subheader("Top 20 fármacos por Z-score")
        st.dataframe(
            ranking_est[["Drug_Name","DrugBank_ID","Proximidad","Media nula","Desviación nula","Zscore","Pvalor"]].head(20),
            use_container_width=True, hide_index=True
        )
        
        st.subheader("Visualización del mejor fármaco")
        #se usa la nueva red contextual en lugar de mostrar solo proteinas
        html_file = visualize_context_network(
            G,
            disease_proteins,
            top_targets,
            "Proteinas de enfermedad",
            f"{top_drug_name} ({top_drug})",
        )

        

        st.subheader("Dispersión: Z-score vs P-valor")
            
        top20_ids = set(ranking_gene.nsmallest(20, "Zscore")["DrugBank_ID"])
            
            # columna para la clasificación 
        ranking_est['Clasificación'] = 'Otros fármacos'
        ranking_est.loc[ranking_est['DrugBank_ID'].isin(top20_ids), 'Clasificación'] = 'Top 20 fármacos'
            
            #  -log10(Pvalor) para el eje X para q no se solapen cerca del cero
        ranking_est["logP"] = -np.log10(ranking_est["Pvalor"])
            
        fig_px = px.scatter(
            ranking_est,
            x="logP",
            y="Zscore",
            color="Clasificación",
            color_discrete_map={
                'Top 20 fármacos': 'red',
                'Otros fármacos': 'gray'
                },
                #  raton
            hover_name="Drug_Name", 
            hover_data={
                'DrugBank_ID': True,  
                'Zscore': ':.2f',    
                'Pvalor': ':.4f',     
                'logP': False,        
                'Clasificación': False 
                },
            labels={
                "logP": "-log10(P-valor)",
                "Zscore": "Z-score"
                },
            template="plotly_white"
            )
            
        fig_px.add_hline(y=0, line_dash="dash", line_color="black")
            
        st.plotly_chart(fig_px, use_container_width=True)

        
        st.download_button(
            "Descargar ranking ",
            ranking_est.to_csv(index=False).encode(),
            "ranking_zscore.csv",
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
    - La proximidad y su significancia estadística ($Z$-score) se calculan mediante simulación Monte Carlo ajustada por grado.
    - También identifica:
        - Alternativas a un fármaco
        - Fármacos que actúan sobre vecinos
        - Conversión de símbolos → UniProt

    """)
    
    # ---------------------------------------------------------
    # NUEVA SECCIÓN: CONTROL Y CALIDAD DE DATOS
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("Resumen de Calidad y Métricas de los Datos")
    
    # cobertura
    nodos_red = G.number_of_nodes()
    aristas_red = G.number_of_edges()
    
    dianas_totales_db = df_drug["UniProt_ID"].nunique()
    dianas_en_red = len([t for t in df_drug["UniProt_ID"].dropna().unique() if t in G])
    cobertura_dianas = (dianas_en_red / dianas_totales_db) * 100 if dianas_totales_db > 0 else 0
    
    farmacos_totales = df_drug["DrugBank_ID"].nunique()
    # fármacos con al menos 1 diana mapeada en el interactoma
    farmacos_con_diana_en_red = sum(1 for tg in drug_targets.values() if len(tg.intersection(G.nodes())) > 0)
    
    todos_los_genes = set()
    for lista in df_disorders["gene_symb"].dropna():
        for g in lista.split(","):
            if g.strip():
                todos_los_genes.add(g.strip())
                
    enfermedades_totales = df_disorders["disorder"].nunique()

    # metricas formato tabla/métricas estructuradas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Nodos del Interactoma (Proteínas)", value=f"{nodos_red:,}")
        st.metric(label="Interacciones (Aristas BioGRID)", value=f"{aristas_red:,}")
    with col2:
        st.metric(label="Fármacos únicos (DrugBank)", value=f"{farmacos_totales:,}")
        st.metric(label="Fármacos Evaluables en Red", value=f"{farmacos_con_diana_en_red:,}")
    with col3:
        st.metric(label="Enfermedades cargadas", value=f"{enfermedades_totales:,}")
        st.metric(label="Total Genes Únicos de Enfermedad", value=f"{len(todos_los_genes):,}")

    st.markdown("#### Cobertura biológica en la red:")
    
    #  DF resumen 
    df_metricas = pd.DataFrame({
        "Indicador de Calidad": [
            "Dianas terapéuticas totales en DrugBank",
            "Dianas terapéuticas cubiertas por el Interactoma",
            "Porcentaje de cobertura de dianas",
            "Densidad de la red (Interactoma)",
            "Número medio de interacciones por proteína (Grado medio)"
        ],
        "Valor": [
            f"{dianas_totales_db:,}",
            f"{dianas_en_red:,}",
            f"{cobertura_dianas:.2f}%",
            f"{nx.density(G):.5f}",
            f"{np.mean([d for n, d in G.degree()]):.2f}"
        ]
    })
    st.table(df_metricas)
    
    st.markdown("---")
    st.write("""
    **Bases de datos utilizadas:**
    - **BioGRID:** Red de interacciones proteína-proteína humanas (interactoma).
    - **DrugBank:** Registro de fármacos y sus correspondientes dianas proteicas mapeadas a identificadores UniProt.
    - **MyGene.info:** Módulo de conversión dinámica de Símbolos Genéticos a accesiones UniProt.
    - *Datos provistos en el marco de la asignatura Análisis de Datos de Alta Dimensión y Medicina de Redes.*
    """)
