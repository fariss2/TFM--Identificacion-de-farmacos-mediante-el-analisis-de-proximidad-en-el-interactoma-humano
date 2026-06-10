# Reposicionamiento de Fármacos mediante Medicina de Redes
<p align="center">
  <img src="https://raw.githubusercontent.com/fariss2/TFM--Identificacion-de-farmacos-mediante-el-analisis-de-proximidad-en-el-interactoma-humano/main/Aplicacion/portada.png" width="700"/>
</p>

Aplicación interactiva desarrollada con **Streamlit** para identificar **fármacos candidatos** mediante análisis de **medicina de redes**.

Esta herramienta evalúa la proximidad entre proteínas asociadas a enfermedades y las dianas de los fármacos dentro del **interactoma humano**, permitiendo generar hipótesis de reposicionamiento terapéutico.

---

## Características Principales

La aplicación se divide en 4 módulos o pestañas interactivas:
1. **Búsqueda por Enfermedad:** Evalúa el reposicionamiento de fármacos del catálogo frente a los genes específicos de una patología seleccionada.
2. **Búsqueda por Fármaco:** Encuentra alternativas terapéuticas (fármacos con mecanismos similares) basándose en la cercanía de sus dianas en la red.
3. **Búsqueda por Proteína/Gen:** Permite introducir manualmente símbolos genéticos o códigos UniProt para estudiar biomarcadores o receptores aislados.
4. **Calidad y Métricas:** Resumen estadístico de la cobertura biológica del interactoma y las bases de datos utilizadas.

### Características Técnicas:
* **Visualización de Redes :**  Visualización interactiva de redes proteína-fármaco.
* **Análisis Estadístico:** Gráficos de dispersión interactivos de Plotly.
* **Conversión Dinámica:** Conversión automática de símbolos de genes a accesiones UniProt.

---

## Fundamento teórico

La aplicación se basa en principios de **Medicina de Redes**:

- Las enfermedades afectan módulos del interactoma
- Los fármacos actúan sobre proteínas diana
- La proximidad entre ambos indica potencial terapéutico
- La significancia se calcula mediante simulaciones Monte Carlo

---
## Librerías y Datos Utilizados

* **Frontend/App:** Streamlit
* **Análisis de Redes:** NetworkX, PyVis
* **Procesamiento de Datos y Gráficos:** Pandas, NumPy, Plotly Express
* **APIs de Biología:** MyGene.info (`gene_to_uniprot`)

### Bases de Datos Biológicas (incluidas en `/data`)
* **BioGRID:** Red de interacciones proteína-proteína humanas (interactoma).
* **DrugBank:** Registro de fármacos y sus correspondientes dianas proteicas mapeadas a UniProt.
* **Disorder Genes:** Mapeo de genes asociados a patologías.[MultiplexDiseasome](https://github.com/manlius/MultiplexDiseasome) (Halu et al., 2019)


---


## Estructura del proyecto

```

├── app.py                      # Aplicación principal de Streamlit
├── gene_to_uniprot.py          # Módulo para la conversión de símbolos a UniProt
├── network_medicine.py         # Algoritmos de red 
├── network_context.html        # Red interactiva generada (autogenerado)
├── requirements.txt            # Dependencias del proyecto
├── Aplicacion/                 # Recursos visuales de la interfaz
│   ├── farmacos.jpg
│   ├── portada.png
│   ├── prot.png
│   └── repo_enf.jpg
└── data/                       # Conjuntos de datos 
    ├── biogrid_edges.csv
    ├── biogrid_nodes.csv
    ├── disorder_genes.csv
    └── drugbank_targets_clean.csv  

```


----

### Instalar dependencias:
    "pip install -r requirements.txt"
### Ejecucion en local 
    "python -m streamlit run app.py"

## Información Académica
Este software forma parte de un Trabajo de Fin de Máster (TFM) realizado en la Universidad de Burgos (UBU) para la titulación del Máster Universitario en Ingeniería Biomédica.

**Autora**: Nisrin Fariss Lamine

**Tutores**: José Ignacio Santos Martín - Virginia Ahedo García
