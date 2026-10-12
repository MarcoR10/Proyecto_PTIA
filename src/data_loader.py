"""Carga de los medidores de AGUA del dataset BDG2 (Building Data Genome Project 2).

Formato de entrada (water.csv): una fila por hora, una columna por edificio,
valores en litros. Salida: tabla diaria en formato largo con las columnas
usuario_id (edificio), fecha, consumo_dia (L), consumo_nocturno (L/h).
"""
from pathlib import Path

import numpy as np
import pandas as pd

RUTAS_POSIBLES = ["data/water.csv", "data/water.txt", "data/meters/raw/water.csv"]


def _buscar_archivo(ruta=None):
    candidatas = [ruta] if ruta else RUTAS_POSIBLES
    for r in candidatas:
        if r and Path(r).exists():
            return r
    raise FileNotFoundError(f"No encontré el archivo de agua. Probé: {candidatas}")


def cargar_agua(ruta=None, cobertura_min=0.90, max_dias_cero=0.30, verbose=True):
    ruta = _buscar_archivo(ruta)
    wide = pd.read_csv(ruta, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    n_ini = wide.shape[1]

    wide = wide.mask(wide < 0)  # lecturas negativas = error de medición -> faltante

    # 1) Filtrar edificios con poca cobertura horaria
    cobertura = wide.notna().mean()
    wide = wide.loc[:, cobertura >= cobertura_min]

    # 2) Agregación diaria. Un día solo es válido si tiene las 24 lecturas.
    dia = wide.resample("D").sum(min_count=24)
    noche_h = wide.between_time("02:00", "04:59")
    noche = noche_h.resample("D").mean()
    noche = noche.mask(noche_h.resample("D").count() < 3)
    noche = noche.where(dia.notna())

    # 3) Filtrar edificios que casi siempre marcan 0 (medidor inactivo)
    frac_cero = (dia == 0).sum() / dia.notna().sum().replace(0, np.nan)
    ok = frac_cero[frac_cero <= max_dias_cero].index
    dia, noche = dia[ok], noche[ok]

    largo = (
        dia.reset_index().melt(id_vars="timestamp", var_name="usuario_id", value_name="consumo_dia")
        .merge(
            noche.reset_index().melt(id_vars="timestamp", var_name="usuario_id", value_name="consumo_nocturno"),
            on=["timestamp", "usuario_id"],
        )
        .rename(columns={"timestamp": "fecha"})
        .sort_values(["usuario_id", "fecha"])
        .reset_index(drop=True)
    )
    if verbose:
        print(f"Edificios: {n_ini} en el archivo -> {len(ok)} tras filtros "
              f"| días válidos: {largo.consumo_dia.notna().sum()} de {len(largo)}")
    return largo
