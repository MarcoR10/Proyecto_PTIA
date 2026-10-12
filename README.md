# Detector de anomalías en consumo de agua

Proyecto del curso **Principios de Inteligencia Artificial (PTIA)**, Escuela Colombiana de Ingeniería Julio Garavito. Grupo 02: Esteban Valente Arenas y Marco Antonio Alvarez.

Prototipo de interfaz que ayuda a un operador a identificar posibles fugas en medidores de agua de edificios. Un modelo **Random Forest** (entrenado con **SMOTEENN** solo sobre el conjunto de entrenamiento) asigna un score de riesgo diario a cada edificio, y una aplicación **Streamlit** muestra las alertas, su explicación y permite registrar la decisión del operador.

> **Importante:** el dataset usado (BDG2) no trae etiquetas de fuga. Sobre el consumo real se **inyectan fugas sintéticas** (enfoque semisintético). Las métricas miden la detección de fugas simuladas, no de fugas reales.

---

## 1. Inicio rápido (ejecutar la aplicación)

Este repositorio **ya incluye los datos y el modelo entrenado**, así que no hace falta descargar nada más ni entrenar para ver la aplicación funcionando.

**Requisito:** Python **3.10 o superior** instalado.

```powershell
# 1. Obtener el proyecto
git clone <URL-DEL-REPOSITORIO>
cd <NOMBRE-DE-LA-CARPETA>
```

> Sin Git: en la página del repositorio usen **Code → Download ZIP**, descomprímanlo y abran la terminal dentro de esa carpeta (la que contiene `app.py`).

```powershell
# 2. Crear y activar el entorno virtual (solo la primera vez)
py -m venv .venv
.venv\Scripts\activate

# 3. Instalar dependencias
# python -m pip install --upgrade pip
# python -m pip install -r requirements.txt

# 4. Ejecutar la aplicación
streamlit run app.py
```

Se abre el navegador en `http://localhost:8501`. Si no se abre solo, copien esa dirección. La primera vez, Streamlit pide un correo opcional: dejen el campo vacío y presionen **Enter**. Para detener la app, `Ctrl + C` en la terminal.

Cuando el entorno está activo, la línea de la terminal empieza con `(.venv)`. Si usan **PyCharm**, pueden abrir la carpeta como proyecto, crear el entorno en *Settings → Project → Python Interpreter* y correr los comandos 3 y 4 en la pestaña **Terminal**.

En macOS o Linux, el paso 2 cambia a `python3 -m venv .venv` y `source .venv/bin/activate`.

## 2. Qué incluye el repositorio

| Archivo | Para qué sirve |
|---|---|
| `data/water.csv` (o `water.txt`) | Medidores de agua de BDG2 (lecturas horarias en litros). Es la entrada del entrenamiento. |
| `data/consumo_procesado.csv` | Consumo diario por edificio con las fugas inyectadas. La app lo usa para las gráficas. |
| `data/predicciones_test.csv` | Score del modelo para el conjunto de prueba. La app lo usa para las alertas. |
| `models/metricas.json` | Métricas, umbral e importancia de variables. La app lo usa para la vista de métricas. |
| `models/modelo_rf.joblib` | Random Forest entrenado. La app **no lo necesita**, solo sirve para reutilizar el modelo. |

Para ejecutar la aplicación bastan `data/consumo_procesado.csv`, `data/predicciones_test.csv` y `models/metricas.json`.

## 3. Cómo usar la aplicación

| Vista | Qué muestra |
|---|---|
| **Panel principal** | Alertas generadas, pendientes de revisión y edificios afectados, ordenadas por score de riesgo. |
| **Detalle de alerta** | Consumo diario y caudal nocturno de los últimos 45 días, la explicación de la alerta y los botones **Confirmar fuga** / **Falso positivo**. |
| **Retroalimentación** | Decisiones registradas por el operador en la sesión. |
| **Métricas del modelo** | Recall, Precision, F1, PR-AUC, matriz de confusión y comparación con una regla fija, por día y por episodio de fuga. |

**Funciones simuladas:** la retroalimentación del operador se guarda solo durante la sesión y **no reentrena el modelo**.

## 4. Reentrenar el modelo (opcional)

Solo es necesario si quieren reproducir los resultados o cambiar el código. **Sobrescribe** los archivos generados de `data/` y `models/`.

```powershell
# python -m src.train
```

Este comando:

1. Carga y limpia los medidores de agua de `data/water.csv` (descarta los de baja cobertura o casi siempre en cero) y los agrega a nivel diario.
2. Inyecta las fugas sintéticas.
3. Construye las variables usando solo información pasada.
4. Hace la partición **temporal** Train / Validation / Test (60 / 20 / 20) y aplica SMOTEENN **solo a Train**.
5. Entrena el Random Forest, elige el umbral de alerta en validación y calcula las métricas, incluida una regla fija de comparación.
6. Guarda los resultados en `data/` y `models/`.

Las semillas están fijas, pero los números pueden variar ligeramente si cambian las versiones de las librerías. Después de reentrenar, vuelvan a abrir la aplicación para ver los resultados nuevos.

**Si falta `data/water.csv`:** descárguenlo del repositorio de *Building Data Genome Project 2* en GitHub (`buds-lab/building-data-genome-project-2`, carpeta `data/meters/raw/`, archivo `water.csv`) y cópienlo en `data/`. Debe tener una columna `timestamp` y una columna por edificio.

## 5. Estructura del proyecto

```
Proyecto_PTIA/
├── app.py                 # Interfaz Streamlit
├── requirements.txt
├── README.md
├── data/                  # Datos de entrada y archivos generados
├── models/                # Modelo entrenado y métricas
└── src/
    ├── data_loader.py     # Carga, limpieza y agregación diaria
    ├── inject_leaks.py    # Inyección de fugas sintéticas (solo evaluación)
    ├── features.py        # Variables con información pasada
    └── train.py           # Partición temporal, SMOTEENN, Random Forest y métricas
```

## 6. Solución de problemas

**`Faltan datos o modelo` al abrir la app.** Faltan archivos en `data/` o `models/`, o se ejecutó `streamlit run app.py` desde otra carpeta. Verifiquen la tabla de la sección 2 y ejecuten el comando desde la raíz del proyecto. Si los archivos no están, corran `python -m src.train`.

**`streamlit` no se reconoce como comando.** Verifiquen que el entorno esté activo (`(.venv)` al inicio de la línea) o ejecuten `python -m streamlit run app.py`.

**`Fatal error in launcher: Unable to create process using ...` al instalar.** El entorno virtual se creó en una ruta que luego se renombró o movió. Borren la carpeta `.venv`, créenla de nuevo (paso 2 del inicio rápido) y no la muevan después. Usar `python -m pip` en lugar de `pip` evita este tipo de fallo.

**PowerShell no deja activar el entorno** (scripts deshabilitados). Ejecuten una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, o usen CMD en lugar de PowerShell.

**Error al cargar `modelo_rf.joblib`.** El modelo se guardó con una versión específica de scikit-learn. Si usan otra, reentrenen con `python -m src.train`. La aplicación no depende de este archivo.

**`No encontré el archivo de agua` al reentrenar.** Verifiquen que `water.csv` (o `water.txt`) esté dentro de `data/`.

## 7. Limitaciones

- Las fugas son simuladas sobre consumo real, así que no hay evidencia de desempeño con fugas reales.
- BDG2 contiene edificios no residenciales, no hogares.
- Se trabaja a nivel diario y sin presión ni topología de red; no se localiza la fuga.
- Las fugas más pequeñas quedan por debajo del ruido normal del caudal nocturno y son difíciles de detectar.

## 8. Fuentes y créditos

- **Datos:** Building Data Genome Project 2 (BDG2), medidores de agua. Se incluye una copia del archivo para facilitar la ejecución; la referencia completa y la licencia están en la fuente original y en el informe.
- **Librerías:** pandas, NumPy, scikit-learn, imbalanced-learn, Streamlit y matplotlib.
- **Uso de IA:** el diseño y buena parte del código se desarrollaron con la asistencia de Claude (Anthropic). Este uso debe declararse en el informe según la política del curso.
