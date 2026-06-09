import requests
#conversion gene a codigo uniptot para su uso posterior 
   
def symbol_to_uniprot(symbol):

    """
    Consulta la API de MyGene.info para mapear un símbolo genético humano a sus identificadores UniProt correspondientes (Swiss-Prot).

    realiza una petición HTTP GET a los servidores de MyGene.info, aplicando filtros específicos para la especie Homo sapiens. 
    Procesa la respuesta JSON para extraer tanto los registros curados manualmente (Swiss-Prot)

    parametros:
        symbol: Símbolo oficial del gen de interés (ej: 'TP53').

    devuelve:
        list: Lista de cadenas de caracteres que contienen las accesiones UniProt de destino asociadas al gen.
    
    
    
    """
    url = "https://mygene.info/v3/query"

    params = {
        "q": symbol,
        "fields": "uniprot",
        "species": "human"

    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    if not data.get("hits"):
        return []   

    uniprot = data["hits"][0].get("uniprot", {})

    resultados = []

    if isinstance(uniprot, dict):
        for key in ["Swiss-Prot", "TrEMBL"]:
            val = uniprot.get(key)

            if isinstance(val, list):
                resultados.extend(val)
            elif isinstance(val, str):
                resultados.append(val)

    return resultados
    

def convert_gene_list(symbols):


    """
    conversión masiva de una lista de símbolos genéticos a sus respectivas equivalencias en identificadores de la base de datos UniProt.

    parametros:
        symbols: Lista o colección indexada de cadenas de caracteres con los Gene Symbols a procesar (['APP', 'TP53', 'TNF']).

    deveuelve:
        dict: Un diccionario estructurado con mapeo clave-valor, donde:
              - Clave: El símbolo genético original de entrada 
              - Valor: Una lista con las accesiones UniProt válidas recuperadas para dicho gen.
    
    """
    return {s: symbol_to_uniprot(s) for s in symbols}
