import networkx as nx
import pandas as pd


# Construye un diccionario donde cada fármaco (DrugBank_ID)
# se asocia con el conjunto de proteínas (UniProt_ID) que lo representan
def get_drug_targets(df_drug):
    return (
        df_drug.groupby("DrugBank_ID")["UniProt_ID"]  # agrupa por fármaco
        .apply(set)                                    # convierte targets a set (sin duplicados)
        .to_dict()                                     # pasa a diccionario Python
    )


# Dado el nombre de un fármaco, devuelve su ID y sus targets proteicos
def get_drug_by_name(df_drug, drug_name):
    # filtra ignorando mayúsculas/minúsculas
    df = df_drug[df_drug["Drug_Name"].str.lower() == drug_name.lower()]
    
    # si no existe el fármaco, devuelve valores vacíos
    if df.empty:
        return None, set()

    # obtiene el ID del primer resultado encontrado
    drug_id = df["DrugBank_ID"].iloc[0]

    # extrae los targets (proteínas) asociados, eliminando valores nulos
    targets = set(df["UniProt_ID"].dropna())

    return drug_id, targets


# Calcula la distancia más corta entre dos nodos en un grafo
def shortest_distance(a, b, G):
    try:
        # usa algoritmo de caminos más cortos de NetworkX
        return nx.shortest_path_length(G, a, b)
    except Exception:
        # si no hay camino o error, devuelve None
        return None


# Calcula la proximidad media entre dos conjuntos de targets (proteínas)
def drug_to_drug_proximity(G, targets_A, targets_B):
    dists = []

    # compara cada proteína del fármaco A con cada proteína del B
    for tA in targets_A:
        for tB in targets_B:
            d = shortest_distance(tA, tB, G)

            # si existe conexión en el grafo, se guarda la distancia
            if d is not None:
                dists.append(d)

    # si no hay ninguna conexión, no se puede calcular proximidad
    if len(dists) == 0:
        return None

    # devuelve la media de distancias 
    return sum(dists) / len(dists)


# Función principal: ordena fármacos según similitud con otro fármaco
def rank_similar_drugs(G, df_drug, drug_name):

    # diccionario: DrugBank_ID -> set de targets
    drug_targets_dict = get_drug_targets(df_drug)

    # obtiene ID y targets del fármaco de referencia
    drugA_id, targets_A = get_drug_by_name(df_drug, drug_name)

    # si no existe el fármaco, lanza error
    if drugA_id is None:
        raise ValueError(f"Fármaco '{drug_name}' no encontrado en DrugBank")

    results = []

    # compara el fármaco A con todos los demás fármacos
    for drugB_id, targets_B in drug_targets_dict.items():

        # evita compararse consigo mismo
        if drugB_id == drugA_id:
            continue

        # calcula proximidad entre ambos fármacos
        score = drug_to_drug_proximity(G, targets_A, targets_B)

        # guarda resultado si es válido
        if score is not None:
            results.append((drugB_id, score))

    # convierte resultados a DataFrame
    ranking = pd.DataFrame(results, columns=["DrugBank_ID", "Proximity"])

    # ordena por proximidad (menor distancia = más similar)
    ranking = ranking.sort_values("Proximity")

    # añade nombres de los fármacos
    names = df_drug[["DrugBank_ID", "Drug_Name"]].drop_duplicates()
    ranking = ranking.merge(names, on="DrugBank_ID", how="left")

    return drugA_id, targets_A, ranking