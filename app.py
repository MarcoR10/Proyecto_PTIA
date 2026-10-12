"""Prototipo interactivo - Detector de anomalías en consumo de agua (PTIA).
Ejecutar con:  streamlit run app.py
Antes correr (una vez):  python -m src.train  (lee data/water.csv de BDG2)
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Detector de fugas", page_icon="💧", layout="wide")

DATA = Path("data")
MODELS = Path("models")


@st.cache_data
def cargar():
    pred = pd.read_csv(DATA / "predicciones_test.csv", parse_dates=["fecha"])
    hist = pd.read_csv(DATA / "consumo_procesado.csv", parse_dates=["fecha"])
    met = json.loads((MODELS / "metricas.json").read_text())
    return pred, hist, met


if not (DATA / "predicciones_test.csv").exists():
    st.error("Faltan datos o modelo. Ejecuta: `python -m src.train` (necesita `data/water.csv` de BDG2).")
    st.stop()

pred, hist, met = cargar()
umbral = met["test"]["umbral"]

# Estado de la sesión: retroalimentación del operador
if "feedback" not in st.session_state:
    st.session_state.feedback = {}  # (usuario, fecha) -> "Fuga confirmada" / "Falso positivo"

alertas = pred[pred["score"] >= umbral].sort_values("score", ascending=False).copy()
alertas["clave"] = alertas["usuario_id"].astype(str) + " | " + alertas["fecha"].dt.strftime("%Y-%m-%d")

st.sidebar.title("💧 Detector de fugas")
vista = st.sidebar.radio("Vista", ["Panel principal", "Detalle de alerta", "Retroalimentación", "Métricas del modelo"])
st.sidebar.caption(f"Umbral de alerta: {umbral:.2f}")
st.sidebar.caption("Datos: medidores de agua de BDG2 (consumo real). Las fugas son inyectadas (semisintético).")

# ---------------------------------------------------------------- Panel
if vista == "Panel principal":
    st.title("Panel principal")
    pendientes = [k for k in alertas["clave"] if k not in st.session_state.feedback]
    c1, c2, c3 = st.columns(3)
    c1.metric("Alertas generadas", len(alertas))
    c2.metric("Pendientes de revisión", len(pendientes))
    c3.metric("Edificios con alerta", alertas["usuario_id"].nunique())

    st.subheader("Alertas ordenadas por riesgo")
    tabla = alertas[["usuario_id", "fecha", "consumo_dia", "ratio_consumo", "ratio_noche", "score"]].copy()
    tabla["estado"] = alertas["clave"].map(st.session_state.feedback).fillna("Pendiente")
    tabla["fecha"] = tabla["fecha"].dt.date
    tabla = tabla.rename(columns={"usuario_id": "edificio", "consumo_dia": "consumo_dia (L)"})
    st.dataframe(tabla.round(2), width="stretch", hide_index=True)

# ---------------------------------------------------------------- Detalle
elif vista == "Detalle de alerta":
    st.title("Detalle de alerta")
    if alertas.empty:
        st.info("No hay alertas.")
        st.stop()
    clave = st.selectbox("Selecciona una alerta (edificio | fecha)", alertas["clave"].tolist())
    fila = alertas[alertas["clave"] == clave].iloc[0]

    st.metric("Score de riesgo", f"{fila['score']:.2f}")

    h = hist[(hist.usuario_id == fila["usuario_id"]) &
             (hist.fecha <= fila["fecha"]) &
             (hist.fecha > fila["fecha"] - pd.Timedelta(days=45))].set_index("fecha")
    izq, der = st.columns(2)
    izq.subheader("Consumo diario (litros)")
    izq.line_chart(h["consumo_dia"])
    der.subheader("Caudal nocturno promedio (L/h, 02:00-05:00)")
    der.line_chart(h["consumo_nocturno"])

    st.subheader("¿Por qué se generó esta alerta?")
    st.write(f"- El consumo del día es **{fila['ratio_consumo']:.2f}×** su mediana de los últimos 28 días "
             f"(z-score {fila['zscore_consumo']:.1f}).")
    st.write(f"- El caudal nocturno es **{fila['ratio_noche']:.2f}×** su nivel habitual. "
             "Un caudal nocturno elevado es una señal típica de fuga.")
    imp = pd.Series(met["importancias"]).sort_values(ascending=False).head(5)
    st.caption("Variables más influyentes en el modelo (importancia global):")
    st.bar_chart(imp)

    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("✅ Confirmar fuga"):
        st.session_state.feedback[clave] = "Fuga confirmada"
        st.success("Registrado: fuga confirmada")
    if b2.button("❌ Falso positivo"):
        st.session_state.feedback[clave] = "Falso positivo"
        st.warning("Registrado: falso positivo")

# ---------------------------------------------------------------- Feedback
elif vista == "Retroalimentación":
    st.title("Retroalimentación del operador")
    if not st.session_state.feedback:
        st.info("Aún no has revisado alertas. Ve a 'Detalle de alerta'.")
    else:
        df_fb = pd.DataFrame(st.session_state.feedback.items(), columns=["alerta (edificio | fecha)", "decisión"])
        st.dataframe(df_fb, width="stretch", hide_index=True)
        st.caption("Estas etiquetas podrían usarse para reentrenar el modelo (canal de retroalimentación).")

# ---------------------------------------------------------------- Métricas
else:
    st.title("Métricas del modelo (conjunto de test)")
    t = met["test"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall", f"{t['recall']:.3f}")
    c2.metric("Precision", f"{t['precision']:.3f}")
    c3.metric("F1", f"{t['f1']:.3f}")
    c4.metric("PR-AUC", f"{t['pr_auc']:.3f}")
    (tn, fp), (fn, tp) = t["matriz_confusion"]
    st.subheader("Matriz de confusión")
    st.dataframe(pd.DataFrame([[tn, fp], [fn, tp]],
                              index=["Real: normal", "Real: fuga"],
                              columns=["Pred: normal", "Pred: fuga"]))
    b = met["baseline_regla"]
    st.subheader("Comparación con una regla fija")
    st.dataframe(pd.DataFrame(
        [["Regla fija (" + b["regla"] + ")", b["recall"], b["precision"], b["f1"]],
         ["Random Forest", t["recall"], t["precision"], t["f1"]]],
        columns=["Método", "Recall", "Precision", "F1"]).round(3), hide_index=True)
    ev, evb = t["eventos"], b["eventos"]
    st.subheader("Detección por episodio de fuga")
    st.dataframe(pd.DataFrame(
        [["Regla fija", f"{evb['episodios_detectados']}/{evb['n_episodios']}", evb["recall_episodio"],
          evb["retraso_medio_dias"], evb["falsas_alarmas_dias"]],
         ["Random Forest", f"{ev['episodios_detectados']}/{ev['n_episodios']}", ev["recall_episodio"],
          ev["retraso_medio_dias"], ev["falsas_alarmas_dias"]]],
        columns=["Método", "Episodios detectados", "Recall por episodio", "Retraso medio (días)", "Días con falsa alarma"]
    ).round(2), hide_index=True)
    st.caption("Consumo real de BDG2 con fugas inyectadas: las métricas miden detección de fugas simuladas. "
               "Partición temporal Train/Val/Test; SMOTEENN solo en train; umbral elegido en validación.")
