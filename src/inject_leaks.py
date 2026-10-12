"""Inyección de fugas SINTÉTICAS sobre consumo real (enfoque semisintético).

BDG2 no trae etiquetas de fuga. Se simulan episodios de 3 a 7 días en los que
se suma un caudal constante L (L/h): una fuga corre las 24 horas, así que
aumenta el consumo diario en 24*L y el caudal nocturno en L. Se agregan además
picos legítimos (solo consumo diurno) que NO son fuga.

Esto debe declararse en el informe: las métricas miden la detección de fugas
simuladas sobre consumo real, no de fugas reales.
"""
import numpy as np
import pandas as pd


def inyectar_fugas(df, fugas_pct=0.02, picos_pct=0.03, seed=42, dias_historial=30):
    rng = np.random.default_rng(seed)
    out = []

    for uid, g in df.groupby("usuario_id", sort=False):
        g = g.sort_values("fecha").reset_index(drop=True)
        dia = g["consumo_dia"].to_numpy(dtype=float).copy()
        noche = g["consumo_nocturno"].to_numpy(dtype=float).copy()
        valido = ~np.isnan(dia) & ~np.isnan(noche)
        n = len(g)
        fuga = np.zeros(n, dtype=int)

        if valido.sum() > 2 * dias_historial:
            base_h = np.nanmedian(dia) / 24.0  # litros/hora promedio del edificio

            # Picos legítimos (visitas, eventos, riego): suben solo el consumo diurno
            for d in rng.choice(np.where(valido)[0], size=int(valido.sum() * picos_pct), replace=False):
                dia[d] += base_h * 24 * rng.uniform(0.10, 0.50)

            # Episodios de fuga
            objetivo = int(valido.sum() * fugas_pct)
            marcados, intentos = 0, 0
            while marcados < objetivo and intentos < 500:
                intentos += 1
                dur = int(rng.integers(3, 8))
                ini = int(rng.integers(dias_historial, max(dias_historial + 1, n - dur)))
                ventana = slice(ini, ini + dur)
                if ini + dur > n or not valido[ventana].all() or fuga[ventana].any():
                    continue
                L = base_h * rng.uniform(0.03, 0.25)   # caudal de fuga (L/h)
                dia[ventana] += L * 24
                noche[ventana] += L
                fuga[ventana] = 1
                marcados += dur

        g["consumo_dia"], g["consumo_nocturno"], g["es_fuga"] = dia, noche, fuga
        out.append(g)

    return pd.concat(out, ignore_index=True)
