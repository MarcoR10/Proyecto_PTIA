"""Ingeniería de características por edificio usando SOLO información pasada
(shift(1)) para no filtrar información del día evaluado.

v2: la línea base es la MEDIANA de los últimos 28 días (robusta: una fuga de
3-7 días no la contamina) y se agregan variables de persistencia, porque una
fuga real dura varios días seguidos.
"""
import pandas as pd

FEATURES = [
    "consumo_dia",
    "consumo_nocturno",
    "mediana_28d",
    "std_28d",
    "noche_mediana_28d",
    "ratio_consumo",
    "zscore_consumo",
    "ratio_noche",
    "ratio_noche_lag1",
    "ratio_noche_lag2",
    "ratio_noche_media3",
    "fraccion_noche",
    "dia_semana",
]


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["usuario_id", "fecha"]).copy()
    g = df.groupby("usuario_id")

    def pasado(col, f):  # estadístico móvil de los 28 días PREVIOS (sin el día actual)
        return g[col].transform(lambda s: f(s.shift(1).rolling(28, min_periods=14)))

    df["mediana_28d"] = pasado("consumo_dia", lambda r: r.median())
    df["std_28d"] = pasado("consumo_dia", lambda r: r.std())
    df["noche_mediana_28d"] = pasado("consumo_nocturno", lambda r: r.median())

    df["ratio_consumo"] = (df["consumo_dia"] / df["mediana_28d"].replace(0, 1e-6)).clip(upper=50)
    df["zscore_consumo"] = ((df["consumo_dia"] - df["mediana_28d"]) / df["std_28d"].replace(0, 1e-6)).clip(-50, 50)
    df["ratio_noche"] = (df["consumo_nocturno"] / df["noche_mediana_28d"].replace(0, 1e-6)).clip(upper=50)

    # Persistencia: cómo venía el caudal nocturno los días anteriores
    g = df.groupby("usuario_id")
    df["ratio_noche_lag1"] = g["ratio_noche"].shift(1)
    df["ratio_noche_lag2"] = g["ratio_noche"].shift(2)
    df["ratio_noche_media3"] = g["ratio_noche"].transform(lambda s: s.rolling(3, min_periods=2).mean())

    # Qué fracción del caudal horario medio es caudal nocturno (indicador clásico de fuga)
    df["fraccion_noche"] = (df["consumo_nocturno"] / (df["consumo_dia"] / 24).replace(0, 1e-6)).clip(upper=50)
    df["dia_semana"] = df["fecha"].dt.dayofweek

    return df.dropna(subset=FEATURES).reset_index(drop=True)
