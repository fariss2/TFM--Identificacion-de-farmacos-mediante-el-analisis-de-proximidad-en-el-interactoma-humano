import streamlit as st
import pandas as pd
import networkx as nx
import os
import numpy as np
import random
from collections import deque
from pyvis.network import Network
import streamlit.components.v1 as components
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

Cuanto más significativamente cerca están las proteínas diana de un fármaco de las proteínas asociadas a la enfermedad, mayor es su potencial efecto terapéutico.

**Interpretación clínica:**
- No implica indicación aprobada, sino hipótesis de reposicionamiento.
- Prioriza candidatos para investigación

""")

st.sidebar.image(
    "Aplicacion/portada.png", 
    width='stretch' 
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

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")#ruta del conjunto de datos 
BIN_TAM =  50 #tamaño del bin
# numero de simulaciones Monte Carlo para estabilizar z-score y p-valor.
N_MC = 500 
# ---------------------------------------------------------
# FUNCIONES
# ---------------------------------------------------------
@st.cache_data
def load_data():
    """
    Carga los archivos de datos para el análisis, las interacciones biológicas (BioGRID), los dianas de los fármacos
    (DrugBank) y los genes asociados a enfermedades.
    Devuelve:
        Una tupla que contiene:
            - G: Grafo que representa el interactoma humano.
            - df_drug: DataFrame con la información de fármacos y sus dianas.
            - df_disorders: DataFrame con genes asociados a patologías.
    
    
    """
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
    """
    Agrupa los nodos del grafo en'bins'basados en su grado. fundamental para análisis estadísticos

    parametros:
        _grafo:El grafo o interactoma de NetworkX.
        tamaño_bin: El rango o tamaño del intervalo de grados para cada contenedor.

    Devuelve:
        Una tupla que contiene:
            - grados: Diccionario de mapeo {nodo: grado_del_nodo}.
            - bins:Diccionario donde la clave es el ID del bin y el valor es una lista de proteínas que pertenecen a ese rango de grado
    
    """
    grados = {n: _grafo.degree(n) for n in _grafo.nodes()} 
    bins = {} 
    for prot, g in grados.items(): 
        bin_id = g // tamaño_bin 
        bins.setdefault(bin_id, []).append(prot) 
    return grados, bins
grados, bins_grado = construir_bins_por_grado(G, BIN_TAM)
@st.cache_data
drug_targets =df_drug.groupby("DrugBank_ID")["UniProt_ID"].apply(set).to_dict()#cada fármaco con el conjunto de sus proteínas diana
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


    """
    Genera una red interactiva en HTML para visualizar el entorno local entre un conjunto de nodos de contexto (enfermedad) y las dianas de un fármaco.

    parametros:
        G: Grafo completo del interactoma
        context_nodes: Lista de nodos (proteínas/genes) asociados al contexto o enfermedad.
        drug_targets: Lista de nodos diana del fármaco a evaluar.
        context_label: Nombre o etiqueta de la enfermedad para el nodo principal virtual.
        drug_label: Nombre del fármaco para el nodo principal virtual.
        max_neighbors: Número máximo de vecinos directos que se mostrarán por cada nodo para evitar la saturación visual. Por defecto es 12.

    devuelve:
        str: Ruta absoluta del archivo HTML generado (`network_context.html`).
    
    """
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
    with st.info(""):
        col_texto, col_imagen = st.columns([3, 2], vertical_alignment="center")
        
        with col_texto:
            st.markdown("""
            **Módulo de Reposicionamiento basado en Genes de la Enfermedad**

            Este módulo permite evaluar la cercanía topológica de los fármacos del catálogo frente a los genes o factores moleculares específicos asociados a una patología seleccionada.

            **Útil cuando:**
            - Deseas priorizar compuestos dirigidos a las bases genéticas de una patología.
            - Analizas una enfermedad compleja a partir de sus dianas moleculares descritas.
            - Buscas fármacos eficaces capaces de interactuar indirectamente sobre el interactoma local del gen de interés.
            """)
            
        with col_imagen:
            st.image(
                "Aplicacion/repo_enf.jpg", 
                width='stretch'
            )
      
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

            

                        

            # se evita acceder a la primera fila si no hay farmacos evaluables
            if ranking_gene.empty:
                st.warning("No se encontraron farmacos evaluables para este gen.")
                st.stop()
            best = ranking_gene.iloc[0]
            
            st.subheader("Top 20 fármacos para este gen por Z-score")
            st.dataframe(
                ranking_gene[["Drug_Name","DrugBank_ID","Proximidad","Media nula","Desviación nula","Zscore","Pvalor"]].head(20),
                width='stretch', hide_index=True
            )

            # Visualización del mejor por Z-score

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
            st.markdown(f"""
**Guía de Interpretación de la Red de Proximidad:**

* **Estrella Azul (Fármaco Candidato - {best['Drug_Name']}):** Representa el fármaco priorizado por el algoritmo de medicina de redes. Las líneas discontinuas azules conectan con sus **dianas proteicas directas** en el interactoma.

* **Diamante Rojo (Gen de Referencia - {gene}):** Representa el gen seleccionado asociado a la enfermedad. Las conexiones indican las proteínas de referencia utilizadas como punto de partida del análisis.

* **Nodos Morados (Superposición):** Proteínas que actúan simultáneamente como dianas del fármaco candidato y están asociadas al gen de interés. Sugieren un **posible mecanismo de acción directo o compartido**.

* **Nodos Grises (Vecinos del Interactoma):** Proteínas cercanas en la red de interacción que actúan como intermediarias. Reflejan la **proximidad topológica indirecta**, clave en medicina de redes.

""")

                
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
            
            st.plotly_chart(fig_px,width='stretch')

            
            st.download_button(
                "Descargar ranking por gen",
                ranking_gene.to_csv(index=False).encode(),
                "ranking_gen_zscore.csv",
                "text/csv"
            )
            

# ---------------------------------------------------------
# TAB 2
# ---------------------------------------------------------
with tab2:
    st.subheader("Alternativas a un fármaco por proximidad de dianas")
    with st.info(""):
        col_texto, col_imagen = st.columns([3, 2], vertical_alignment="center")
        
        with col_texto:
            st.markdown("""
            **Búsqueda de alternativas terapéuticas**

            Este módulo identifica fármacos con mecanismos similares basados en sus dianas en la red.

            **Útil cuando:**
            - Un fármaco no funciona
            - Hay efectos adversos
            - Se buscan alternativas terapéuticas

            **No implica equivalencia clínica directa.**
            """)
            
        with col_imagen:
            st.image(
                "Aplicacion/farmacos.jpg", 
                width='stretch'
            )
   

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
                        top_drug_name = ranking_alt.iloc[0]["Drug_Name"]
                        


                        

                        st.subheader("Top 20 alternativas por Z-score")
                        st.dataframe(
                            ranking_alt[["Drug_Name","DrugBank_ID","Proximidad","Media nula","Desviación nula", "Zscore","Pvalor"]].head(20),
                            width='stretch', hide_index=True
                        )


                        st.subheader(f"Visualización — candidato  frente a {farmaco_nombre}")
                        html_file = visualize_context_network(
                            G,
                            ref_targets,
                            top_targets,
                            f"{farmaco_nombre} ({drug_id_ref})",
                            f"{top_drug_name} ({top_drug})",
                        )
                        with open(html_file, "r", encoding="utf-8") as f:
                            components.html(f.read(), height=600)

                        st.markdown(f"""
                        **Guía de Interpretación de la Red de Proximidad:**
                        * **Estrella Azul (Fármaco Candidato - {top_drug_name}):** Representa al fármaco alternativo priorizado por el algoritmo. Las líneas discontinuas azules muestran sus dianas proteicas directas en el mapa.
                        * **Diamante Rojo (Referencia - {farmaco_nombre}):** Representa el fármaco de base seleccionado del cual estás buscando alternativas. Las líneas discontinuas rojas conectan directamente con sus proteínas asociadas.
                        * **Nodos Morados (Superposición):** Proteínas clave que funcionan simultáneamente como dianas de **{top_drug_name}** y de **{farmaco_nombre}**. Indican un solapamiento directo de mecanismos de acción en el interactoma.
                        * **Nodos Grises (Vecinos del Interactoma):** Proteínas del interactoma humano (BioGRID) que actúan como puentes topológicos intermediarios. Facilitan la interconexión física y funcional entre ambos perfiles terapéuticos.
                        """)
                


                            
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
                        
                        st.plotly_chart(fig_px, width='stretch')
                        

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

    with st.info(""):
        col_texto, col_imagen = st.columns([3, 2], vertical_alignment="center")
        
        with col_texto:
            st.markdown("""
            **Análisis de dianas o proteínas individuales**

            Este módulo te permite evaluar el potencial de reposicionamiento de fármacos enfocándote en proteínas aisladas de interés o en genes específicos relacionado con una patología.

            **Útil cuando:**
            - Deseas estudiar el impacto de una única proteína clave.
            - Analizas un biomarcador específico o un receptor aislado.
            - Buscas fármacos cuyas dianas se encuentren topológicamente cerca de esta proteína en el interactoma.
            """)
            
        with col_imagen:
            st.image("Aplicacion/prot.png", width='stretch')
            
    user_input = st.text_area("Símbolo de interés", height=120)

    if user_input:
        raw_items = [p.strip() for p in user_input.split(",") if p.strip()]

        def es_uniprot(x):
            return len(x) == 6 and x[0].isalpha() and x[1].isdigit()

        symbols = [x for x in raw_items if not es_uniprot(x)]
        uniprots = [x for x in raw_items if es_uniprot(x)]

        mapping = convert_gene_list(symbols) if symbols else {}

        converted_uniprots = []
        for s in symbols:
            vals = mapping.get(s)
            if vals is None:
                continue
            
            if not isinstance(vals, list):
                vals = [vals]
                
            # Filtro Swiss-Prot Canónico
            swiss_prot_ids = [str(v).split('-')[0] for v in vals if len(str(v).split('-')[0]) == 6]
            if swiss_prot_ids:
                converted_uniprots.append(swiss_prot_ids[0])

        disease_proteins = set(uniprots + converted_uniprots)

        # conversión limpia en pantalla antes de evaluar
        if symbols:
            st.write("### Conversión símbolo → UniProt")
            for s in symbols:
                ind_vals = mapping.get(s)
                if ind_vals is None:
                    st.error(f"{s}: NO encontrado")
                else:
                    if not isinstance(ind_vals, list):
                        ind_vals = [ind_vals]
                    sp_ids = [str(v).split('-')[0] for v in ind_vals if len(str(v).split('-')[0]) == 6]
                    if sp_ids:
                        st.success(f"{s} → {sp_ids[0]}")
                    else:
                        st.warning(f"{s} → No se encontró un ID Swiss-Prot canónico equivalente.")

        # Filtrar por grafo
        disease_proteins = disease_proteins.intersection(G.nodes())

        if len(disease_proteins) == 0:
            st.warning("Ninguna proteína coincide con el interactoma.")
            st.stop()

        if st.button("Evaluar gen/proteína"):
            
            @st.cache_data(show_spinner=False)
            def compute_distances_cached(nodes_tuple):
                return bfs_multifuente(G, set(nodes_tuple))

            with st.spinner("Calculando distancias..."):
                dist_ref = compute_distances_cached(tuple(sorted(disease_proteins)))

            # Monte Carlo dinámico
            N_MC_DYNAMIC = 200 if len(disease_proteins) > 5 else 500

            @st.cache_data(show_spinner=False)
            def compute_ranking_cached(nodes_tuple):
                resultados = []
                for drug, targets in drug_targets.items():
                    valids = list(targets.intersection(G.nodes()))
                    if not valids:
                        continue

                    d_obs, mu, sd, z, p = proximidad_estadistica(
                        valids,
                        dist_ref,
                        grados,
                        bins_grado,
                        repeticiones=N_MC_DYNAMIC,
                        tamaño_bin=BIN_TAM
                    )

                    if d_obs is None:
                        continue

                    resultados.append((drug, d_obs, mu, sd, z, p))

                return resultados

            with st.spinner("Evaluando fármacos..."):
                resultados = compute_ranking_cached(tuple(sorted(disease_proteins)))

            ranking_est = pd.DataFrame(
                resultados,
                columns=["DrugBank_ID","Proximidad","Media nula","Desviación nula","Zscore","Pvalor"]
            ).merge(
                df_drug[["DrugBank_ID","Drug_Name"]].drop_duplicates(),
                on="DrugBank_ID", how="left"
            ).sort_values(["Zscore","Pvalor"], ascending=[True, True])

            if ranking_est.empty:
                st.warning("No se encontraron fármacos evaluables.")
                st.stop()

            st.subheader("Top 20 fármacos por Z-score")
            st.dataframe(
                ranking_est[["Drug_Name","DrugBank_ID","Proximidad","Media nula","Desviación nula","Zscore","Pvalor"]].head(20),
                width=True, hide_index=True
            )

            st.subheader("Visualización del fármaco candidato")

            top = ranking_est.iloc[0]
            top_targets = drug_targets[top["DrugBank_ID"]]

            # Reducir tamaño de entrada a la red visualizada
            disease_subset = list(disease_proteins)[:5]

            html_file = visualize_context_network(
                G,
                disease_subset,
                top_targets,
                "Proteínas input",
                f"{top['Drug_Name']} ({top['DrugBank_ID']})",
            )

            with open(html_file, "r", encoding="utf-8") as f:
                components.html(f.read(), height=600)
                
            st.markdown(f"""
**Guía de Interpretación de la Red de Proximidad:**

* **Estrella Azul (Fármaco Candidato - {top['Drug_Name']}):** Representa el fármaco priorizado por el algoritmo. Las líneas discontinuas azules conectan con sus **dianas proteicas directas**, mostrando sobre qué elementos del interactoma actúa.

* **Diamante Rojo (Proteínas de Entrada- {str(user_input).strip()}):** Representa el conjunto de proteínas o genes introducidos por el usuario. Constituyen el contexto biológico de referencia del análisis.

* **Nodos Morados (Superposición):** Proteínas que coinciden entre las dianas del fármaco candidato y el conjunto de proteínas de entrada. Indican una **interacción directa potencial**, sugiriendo un mecanismo de acción más específico.

* **Nodos Grises (Vecinos del Interactoma):** Proteínas cercanas en la red de interacción que no son dianas directas pero actúan como intermediarias. Reflejan la **proximidad funcional indirecta**, clave en estrategias de reposicionamiento.
""")

            st.subheader("Dispersión: Z-score vs P-valor")

            top20_ids = set(ranking_est.nsmallest(20, "Zscore")["DrugBank_ID"])

            ranking_est['Clasificación'] = 'Otros fármacos'
            ranking_est.loc[ranking_est['DrugBank_ID'].isin(top20_ids), 'Clasificación'] = 'Top 20 fármacos'

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
                hover_name="Drug_Name",
                template="plotly_white"
            )

            fig_px.add_hline(y=0, line_dash="dash", line_color="black")

            st.plotly_chart(fig_px,width='stretch')

            st.download_button(
                "Descargar ranking",
                ranking_est.to_csv(index=False).encode(),
                "ranking_zscore.csv",
                "text/csv"
            )
            
# ---------------------------------------------------------
# TAB 4
# ---------------------------------------------------------
with tab4: 
    st.subheader(""" ¿Cómo funciona esta herramienta?""")
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
    # CONTROL Y CALIDAD DE DATOS
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader(""" Resumen de Calidad y Métricas de los Datos""")
    
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
    ### Bases de datos utilizadas:
    - **BioGRID:** Red de interacciones proteína-proteína humanas (interactoma).
    - **DrugBank:** Registro de fármacos y sus correspondientes dianas proteicas mapeadas a identificadores UniProt.
    - **MyGene.info:** Módulo de conversión dinámica de Símbolos Genéticos a accesiones UniProt.
    - *Datos provistos en el marco de la asignatura Análisis de Datos de Alta Dimensión y Medicina de Redes.*
    """)
    st.markdown("---")

    st.subheader("Información Académica")
    
    st.markdown("""
    #### Trabajo de Fin de Máster (TFM)
    Esta aplicación ha sido desarrollada como parte de un **Trabajo de Fin de Máster (TFM)** en la **Universidad de Burgos (UBU)**.
    
    * **Titulación:** Máster Universitario en Ingeniería Biomédica.
    * **Autora:** Nisrin Fariss Lamine.
    * **Tutores de TFM:** 
      * José Ignacio Santos Martín 
      * Virginia Ahedo García 
    
    ---
    """)
