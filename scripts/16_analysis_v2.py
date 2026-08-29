"""Analisis de la segunda ronda. Brazo de compensacion y replicacion.

Dos preguntas.
  1. Que aporta reparar el canal frente a excluirlo, y que cuesta llevar el
     termino de sesgo siempre activo en lugar de dispararlo.
  2. Coinciden las dos rondas en los diez brazos comunes. Es una replicacion
     sobre semillas independientes. Ver D21.

Resumenes por mediana y rango intercuartilico, mas fraccion de repeticiones
con mejora. La media no se usa sola. Ver D22.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "primary")

ARMS = ["EKF_nom", "CovMatch", "SageHusa", "Huber", "Propuesto",
        "Compensado", "Comp_siempre"]


def summary_cell(df, scen, sev, arms, metric="pos_rmse", ref="EKF_nom"):
    sub = df[(df.escenario == scen) & (df.severidad == sev)]
    base = sub[sub.estimador == ref].sort_values("repeticion")[metric].to_numpy()
    rows = []
    for a in arms:
        v = sub[sub.estimador == a].sort_values("repeticion")[metric].to_numpy()
        if v.size == 0:
            continue
        row = {"brazo": a, "mediana": np.median(v),
               "q1": np.percentile(v, 25), "q3": np.percentile(v, 75),
               "cv": v.std(ddof=1) / v.mean() if v.mean() else np.nan}
        if a != ref and v.size == base.size:
            row["mejora_frac"] = float((v < base).mean())
            row["mejora_mediana_pct"] = float(100 * np.median(v / base - 1.0))
            row["p"] = float(wilcoxon(v, base).pvalue)
        rows.append(row)
    return pd.DataFrame(rows)


def replication(v1, v2):
    """Acuerdo entre rondas sobre los brazos comunes."""
    common = sorted(set(v1.estimador) & set(v2.estimador))
    a = (v1[v1.estimador.isin(common)]
         .groupby(["estimador", "escenario", "severidad"])["pos_rmse"].mean())
    b = (v2[v2.estimador.isin(common)]
         .groupby(["estimador", "escenario", "severidad"])["pos_rmse"].mean())
    j = pd.concat([a.rename("v1"), b.rename("v2")], axis=1).dropna()
    j["dif_rel"] = j["v2"] / j["v1"] - 1.0
    return common, j


def main():
    v1 = pd.read_csv(os.path.join(RES, "confirmatory_runs.csv"))
    v2 = pd.read_csv(os.path.join(RES, "confirmatory_v2_runs.csv"))
    report = {}

    print("=" * 76)
    print("Escenario de sesgo, severidad uno. RMSE de posicion, segunda ronda")
    print("=" * 76)
    s = summary_cell(v2, "S2_bias", 1.0, ARMS)
    print(s.round(4).to_string(index=False))
    s.to_csv(os.path.join(ROOT, "tables", "v2_bias_sev1.csv"), index=False)
    report["S2_bias_sev1"] = s.round(5).to_dict("records")

    print()
    print("=" * 76)
    print("Escenario sin degradacion. Precio de cada mecanismo")
    print("=" * 76)
    n = summary_cell(v2, "S0_nominal", 0.0, ARMS)
    print(n.round(4).to_string(index=False))
    n.to_csv(os.path.join(ROOT, "tables", "v2_nominal.csv"), index=False)
    report["S0_nominal"] = n.round(5).to_dict("records")

    print()
    print("=" * 76)
    print("Replicacion entre rondas, brazos comunes, semillas independientes")
    print("=" * 76)
    common, j = replication(v1, v2)
    print("brazos comunes %d, celdas comparadas %d" % (len(common), len(j)))
    agg = j.groupby(level=0)["dif_rel"].agg(
        mediana="median", p10=lambda x: np.percentile(x, 10),
        p90=lambda x: np.percentile(x, 90),
        maxabs=lambda x: np.max(np.abs(x)))
    print((100 * agg).round(2).to_string())
    print("Valores en por ciento. Diferencia relativa de la segunda ronda")
    print("respecto de la primera.")
    j.to_csv(os.path.join(ROOT, "tables", "replication_v1_v2.csv"))
    report["replicacion"] = {
        "n_brazos": len(common), "n_celdas": int(len(j)),
        "max_abs_dif_rel_pct": round(float(100 * j["dif_rel"].abs().max()), 3),
        "mediana_abs_dif_rel_pct": round(
            float(100 * j["dif_rel"].abs().median()), 3)}

    with open(os.path.join(RES, "analysis_v2.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=float)
    print()
    print("escrito results/primary/analysis_v2.json y tablas en tables/")


if __name__ == "__main__":
    main()
