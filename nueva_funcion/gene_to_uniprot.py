

import requests
#conversion gene a codigo uniptot para su uso posterior 
def symbol_to_uniprot(symbol):
    url = "https://mygene.info/v3/query"

    params = {
        "q": symbol,
        "fields": "uniprot"
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    if not data.get("hits"):
        return None

    uniprot = data["hits"][0].get("uniprot", {})

    if isinstance(uniprot, dict):
        return uniprot.get("Swiss-Prot") or uniprot.get("TrEMBL")

    return None


def convert_gene_list(symbols):
    return {s: symbol_to_uniprot(s) for s in symbols}
