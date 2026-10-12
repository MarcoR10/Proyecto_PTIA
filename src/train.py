"""Entrenamiento: partición temporal Train/Val/Test, SMOTEENN SOLO sobre train,
Random Forest y métricas apropiadas para desbalance."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score)

from src.data_loader import cargar_agua
from src.inject_leaks import inyectar_fugas
from src.features import FEATURES, construir_features

try:
    from imblearn.combine import SMOTEENN
except ImportError:  # permite probar sin imbalanced-learn
    SMOTEENN = None


def particion_temporal(df, p_train=0.6, p_val=0.2):
    fechas = np.sort(df["fecha"].unique())
    c1 = fechas[int(len(fechas) * p_train)]
    c2 = fechas[int(len(fechas) * (p_train + p_val))]
    return df[df.fecha < c1], df[(df.fecha >= c1) & (df.fecha < c2)], df[df.fecha >= c2]


def metricas(y, proba, umbral):
    pred = (proba >= umbral).astype(int)
    return {
        "umbral": float(umbral),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, proba)),
        "matriz_confusion": confusion_matrix(y, pred).tolist(),
    }


def metricas_eventos(df, pred):
    """Métricas por EPISODIO de fuga: un episodio se detecta si se alerta al menos
    un día dentro de él. Mide también el retraso (días) hasta la primera alerta."""
    d = df[["usuario_id", "fecha", "es_fuga"]].copy()
    d["pred"] = np.asarray(pred)
    d = d.sort_values(["usuario_id", "fecha"])
    # id de episodio: cambia cuando cambia el estado de fuga o el edificio
    cambio = (d["es_fuga"] != d["es_fuga"].shift()) | (d["usuario_id"] != d["usuario_id"].shift())
    d["ep"] = cambio.cumsum()
    eps = d[d["es_fuga"] == 1].groupby("ep")
    n = eps.ngroups
    detectados, retrasos = 0, []
    for _, e in eps:
        hit = np.flatnonzero(e["pred"].to_numpy() == 1)
        if len(hit):
            detectados += 1
            retrasos.append(int(hit[0]))
    normales = d[d["es_fuga"] == 0]
    return {
        "n_episodios": int(n),
        "episodios_detectados": int(detectados),
        "recall_episodio": float(detectados / n) if n else 0.0,
        "retraso_medio_dias": float(np.mean(retrasos)) if retrasos else None,
        "falsas_alarmas_dias": int(normales["pred"].sum()),
        "tasa_falsas_alarmas": float(normales["pred"].mean()) if len(normales) else 0.0,
    }


def elegir_umbral(y_val, proba_val):
    """Umbral que maximiza F1 en VALIDACIÓN (nunca en test)."""
    mejores = max(np.linspace(0.05, 0.95, 91),
                  key=lambda u: f1_score(y_val, (proba_val >= u).astype(int), zero_division=0))
    return float(mejores)


def main():
    crudo = inyectar_fugas(cargar_agua())
    crudo.to_csv("data/consumo_procesado.csv", index=False)  # historial para la app
    df = construir_features(crudo)
    train, val, test = particion_temporal(df)
    print(f"Train {len(train)} | Val {len(val)} | Test {len(test)}")
    print(f"% fugas -> train {train.es_fuga.mean():.2%} | val {val.es_fuga.mean():.2%} | test {test.es_fuga.mean():.2%}")

    X_tr, y_tr = train[FEATURES], train["es_fuga"]
    if SMOTEENN is not None:
        X_tr, y_tr = SMOTEENN(random_state=42).fit_resample(X_tr, y_tr)  # solo train
        print(f"Tras SMOTEENN (solo train): {len(X_tr)} filas, {y_tr.mean():.1%} fugas")
    else:
        print("imbalanced-learn no instalado: entrenando sin remuestreo")

    modelo = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    modelo.fit(X_tr, y_tr)

    umbral = elegir_umbral(val["es_fuga"], modelo.predict_proba(val[FEATURES])[:, 1])
    # Línea base: regla fija sobre el caudal nocturno (sin aprendizaje)
    u_regla = max(np.linspace(1.05, 5, 80),
                  key=lambda u: f1_score(val["es_fuga"], (val["ratio_noche"] >= u).astype(int), zero_division=0))
    pred_regla = (test["ratio_noche"] >= u_regla).astype(int)
    baseline = {
        "regla": f"ratio_noche >= {u_regla:.2f}",
        "recall": float(recall_score(test["es_fuga"], pred_regla, zero_division=0)),
        "precision": float(precision_score(test["es_fuga"], pred_regla, zero_division=0)),
        "f1": float(f1_score(test["es_fuga"], pred_regla, zero_division=0)),
        "eventos": metricas_eventos(test, pred_regla),
    }

    resultado = {
        "baseline_regla": baseline,
        "validacion": metricas(val["es_fuga"], modelo.predict_proba(val[FEATURES])[:, 1], umbral),
        "test": metricas(test["es_fuga"], modelo.predict_proba(test[FEATURES])[:, 1], umbral),
        "importancias": dict(zip(FEATURES, map(float, modelo.feature_importances_))),
    }

    resultado["test"]["eventos"] = metricas_eventos(test, modelo.predict_proba(test[FEATURES])[:, 1] >= umbral)

    test = test.copy()
    test["score"] = modelo.predict_proba(test[FEATURES])[:, 1]
    test.to_csv("data/predicciones_test.csv", index=False)
    joblib.dump(modelo, "models/modelo_rf.joblib")
    with open("models/metricas.json", "w") as f:
        json.dump(resultado, f, indent=2)

    t = resultado["test"]
    b = resultado["baseline_regla"]
    print(f"BASELINE regla ({b['regla']}) -> recall {b['recall']:.3f} | precision {b['precision']:.3f} | F1 {b['f1']:.3f}")
    ev, evb = resultado["test"]["eventos"], b["eventos"]
    print(f"EPISODIOS test -> modelo {ev['episodios_detectados']}/{ev['n_episodios']} detectados "
          f"(recall {ev['recall_episodio']:.2f}, retraso medio {ev['retraso_medio_dias']} días) | "
          f"regla {evb['episodios_detectados']}/{evb['n_episodios']} (recall {evb['recall_episodio']:.2f})")
    print(f"TEST -> recall {t['recall']:.3f} | precision {t['precision']:.3f} | "
          f"F1 {t['f1']:.3f} | PR-AUC {t['pr_auc']:.3f} | umbral {umbral:.2f}")


if __name__ == "__main__":
    main()
