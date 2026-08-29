"""Frontera de adaptacion y costo computacional relativo.

Dos analisis que se derivan del mismo archivo confirmatorio y no requieren
volver a simular.

Frontera de adaptacion. Para cada comparador y cada modo de degradacion se
localiza la severidad en la que el estimador deja de ser consistente. El
criterio es que el NEES promedio salga de la banda teorica al noventa y cinco
por ciento. Como el NEES de cada repeticion se distribuye como chi cuadrado
con seis grados de libertad, el promedio sobre N repeticiones tiene banda
superior igual al cuantil de una chi cuadrado con seis por N grados de
libertad dividido por N. El umbral se interpola de forma lineal entre las
severidades muestreadas y se declara ausente si el estimador nunca sale de la
banda.

Costo computacional. Se deriva del tiempo de pared registrado por corrida.
Advertencia obligatoria, mide una implementacion vectorizada sobre
repeticiones en Python, de modo que es indicativo del costo algoritmico
relativo y no de un tiempo de ejecucion embebido.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "primary")
TAB = os.path.join(ROOT, "tables")

MAIN = ["EKF_nom", "EKF_mis", "CovMatch", "SageHusa", "Huber", "Propuesto"]
DOF = 6
N_STEPS = 12000


def nees_band(n_rep, level=0.95):
    lo = chi2.ppf((1.0 - level) / 2.0, DOF * n_rep) / n_rep
    hi = chi2.ppf(1.0 - (1.0 - level) / 2.0, DOF * n_rep) / n_rep
    return lo, hi


def frontier(df, hi):
    """Severidad en la que el NEES promedio abandona la banda de consistencia."""
    rows = []
    for (scen, est), g in df[df.estimador.isin(MAIN)].groupby(
            ["escenario", "estimador"]):
        m = g.groupby("severidad")["nees"].mean().sort_index()
        sev = m.index.to_numpy(dtype=float)
        val = m.to_numpy()
        thr = np.nan
        for i in range(1, sev.size):
            if val[i - 1] <= hi < val[i]:
                frac = (hi - val[i - 1]) / (val[i] - val[i - 1])
                thr = sev[i - 1] + frac * (sev[i] - sev[i - 1])
                break
        if np.isnan(thr) and val[0] > hi:
            thr = 0.0
        rows.append({"escenario": scen, "estimador": est,
                     "severidad_umbral": thr,
                     "nees_sev_max": float(val[-1])})
    return pd.DataFrame(rows)


def cost(df):
    """Costo relativo por paso, derivado del tiempo de pared por corrida."""
    g = (df[df.estimador.isin(MAIN)]
         .groupby(["estimador", "escenario", "severidad"])["segundos_corrida"]
         .first().reset_index())
    agg = g.groupby("estimador")["segundos_corrida"].agg(["mean", "std"])
    base = agg.loc["EKF_nom", "mean"]
    agg["relativo"] = agg["mean"] / base
    agg["us_por_paso"] = agg["mean"] / N_STEPS * 1.0e6
    return agg.sort_values("mean")


def main():
    df = pd.read_csv(os.path.join(RES, "confirmatory_runs.csv"))
    n_rep = int(df.groupby(["estimador", "escenario", "severidad"]).size().max())
    lo, hi = nees_band(n_rep)
    os.makedirs(TAB, exist_ok=True)

    print("banda de consistencia al noventa y cinco por ciento para el NEES")
    print("promedio sobre %d repeticiones, de %.3f a %.3f" % (n_rep, lo, hi))
    print()
    print("=" * 74)
    print("Frontera de adaptacion. Severidad en la que se pierde consistencia")
    print("Un valor ausente significa que el estimador nunca sale de la banda")
    print("=" * 74)
    fr = frontier(df, hi)
    piv = fr.pivot(index="estimador", columns="escenario",
                   values="severidad_umbral").reindex(MAIN)
    print(piv.round(3).to_string(na_rep="  ---"))
    fr.to_csv(os.path.join(TAB, "adaptation_frontier.csv"), index=False)

    print()
    print("NEES en severidad maxima, para dimensionar la magnitud del fallo")
    piv2 = fr.pivot(index="estimador", columns="escenario",
                    values="nees_sev_max").reindex(MAIN)
    print(piv2.round(1).to_string())

    print()
    print("=" * 74)
    print("Costo computacional relativo")
    print("=" * 74)
    ct = cost(df)
    print(ct.round(3).to_string())
    ct.to_csv(os.path.join(TAB, "computational_cost.csv"))
    print()
    print("Advertencia. Mide una implementacion vectorizada en Python. Es")
    print("indicativo del costo algoritmico relativo, no de tiempo embebido.")

    with open(os.path.join(RES, "frontier_cost.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"n_rep": n_rep, "banda_nees": [lo, hi],
                   "frontera": fr.to_dict("records"),
                   "costo_relativo": ct["relativo"].round(3).to_dict()},
                  fh, indent=2, ensure_ascii=False)
    print("escrito tables/adaptation_frontier.csv y "
          "tables/computational_cost.csv")


if __name__ == "__main__":
    main()
