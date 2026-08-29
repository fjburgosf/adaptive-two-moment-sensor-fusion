"""Contraste de H1, H2 y H3 sobre el estudio confirmatorio.

Parte de results/primary/confirmatory_runs.csv y no vuelve a simular. Toda
tabla y figura del manuscrito debe derivar de este mismo archivo.

Disciplina estadistica.
  Comparaciones pareadas sobre las mismas repeticiones, que es valido porque
  el NoiseBook garantiza identica realizacion aleatoria entre estimadores.
  Prueba de Wilcoxon de rangos con signo, que no supone normalidad.
  Correccion de Holm sobre el numero total de pruebas de cada familia, con el
  numero registrado de forma explicita.

Salida, results/primary/hypotheses.json y tablas en tables/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "primary")
TAB = os.path.join(ROOT, "tables")

MAIN = ["EKF_nom", "EKF_mis", "CovMatch", "SageHusa", "Huber", "Propuesto"]
NEES_TARGET = 6.0


def holm(pvals):
    """Correccion de Holm. Devuelve los p ajustados en el orden de entrada."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def cell_means(df, metric, estimators=MAIN):
    """Media por estimador dentro de cada celda de escenario y severidad."""
    sub = df[df.estimador.isin(estimators)]
    return sub.groupby(["escenario", "severidad", "estimador"])[metric].mean()


def h1(df):
    """H1. Conservacion del ordenamiento entre lazo abierto y lazo cerrado.

    Para cada celda se ordenan los seis estimadores por error de estimacion de
    posicion y por tres metricas de lazo cerrado. Una correlacion cercana a
    mas uno significa que el ordenamiento se conserva, es decir que H1 no se
    sostiene en esa celda.
    """
    targets = {"track_rmse": "error de seguimiento medio",
               "track_max": "excursion maxima",
               "recovery": "tiempo de recuperacion"}
    rows = []
    for (scen, sev), grp in df[df.estimador.isin(MAIN)].groupby(
            ["escenario", "severidad"]):
        m = grp.groupby("estimador").mean(numeric_only=True)
        m = m.reindex(MAIN)
        for tgt, _ in targets.items():
            rho = spearmanr(m["pos_rmse"], m[tgt]).statistic
            tau = kendalltau(m["pos_rmse"], m[tgt]).statistic
            rows.append({"escenario": scen, "severidad": sev, "objetivo": tgt,
                         "spearman": rho, "kendall": tau})
    out = pd.DataFrame(rows)
    summary = (out.groupby("objetivo")["spearman"]
                  .agg(["mean", "median", "min", "max", "count"]))
    return out, summary


def h2(df):
    """H2. La consistencia predice el lazo cerrado mejor que el RMSE.

    En cada celda se calcula la correlacion de ordenamientos entre el error de
    seguimiento y dos predictores rivales, el error de estimacion y la
    desviacion de consistencia medida como el logaritmo del cociente entre el
    NEES observado y su valor esperado. Despues se comparan las magnitudes de
    ambas correlaciones con una prueba pareada sobre las celdas.
    """
    rows = []
    for (scen, sev), grp in df[df.estimador.isin(MAIN)].groupby(
            ["escenario", "severidad"]):
        m = grp.groupby("estimador").mean(numeric_only=True).reindex(MAIN)
        incons = np.abs(np.log(m["nees"] / NEES_TARGET))
        rows.append({
            "escenario": scen, "severidad": sev,
            "rho_rmse": spearmanr(m["pos_rmse"], m["track_rmse"]).statistic,
            "rho_incons": spearmanr(incons, m["track_rmse"]).statistic,
        })
    out = pd.DataFrame(rows).dropna()
    a = out["rho_incons"].abs()
    b = out["rho_rmse"].abs()
    stat, p = wilcoxon(a, b)
    return out, {"n_celdas": int(len(out)),
                 "media_abs_rho_inconsistencia": float(a.mean()),
                 "media_abs_rho_rmse": float(b.mean()),
                 "wilcoxon_p": float(p),
                 "lectura": ("H2 se sostiene si la consistencia correlaciona "
                             "mas que el RMSE y la prueba pareada lo respalda")}


def h3(df):
    """H3. El tiempo minimo de permanencia reduce la conmutacion espuria.

    Se compara el esquema completo contra la ablacion sin permanencia, sobre
    las mismas repeticiones, en el numero de conmutaciones y en el error.
    """
    rows = []
    pvals, keys = [], []
    for (scen, sev), grp in df.groupby(["escenario", "severidad"]):
        have = set(grp.estimador.unique())
        if not {"Propuesto", "Abl_sin_permanencia"} <= have:
            continue
        for ch in ("gnss", "mag"):
            a = grp[grp.estimador == "Propuesto"].sort_values(
                "repeticion")["sw_" + ch].to_numpy()
            b = grp[grp.estimador == "Abl_sin_permanencia"].sort_values(
                "repeticion")["sw_" + ch].to_numpy()
            if np.allclose(a, b):
                p = 1.0
            else:
                p = wilcoxon(a, b).pvalue
            pvals.append(p)
            keys.append((scen, sev, ch))
            rows.append({"escenario": scen, "severidad": sev, "canal": ch,
                         "conmut_propuesto": float(a.mean()),
                         "conmut_sin_permanencia": float(b.mean()),
                         "p": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"].to_numpy())
    return out, {"n_pruebas": len(out)}


def head_to_head(df, severity=1.0):
    """Comparacion pareada del esquema propuesto contra cada comparador."""
    rows, pvals = [], []
    for scen in sorted(df.escenario.unique()):
        for metric in ("pos_rmse", "track_rmse", "track_max"):
            base = df[(df.escenario == scen) & (df.severidad == severity)]
            if base.empty:
                continue
            prop = base[base.estimador == "Propuesto"].sort_values(
                "repeticion")[metric].to_numpy()
            for other in MAIN:
                if other == "Propuesto":
                    continue
                oth = base[base.estimador == other].sort_values(
                    "repeticion")[metric].to_numpy()
                if oth.size != prop.size or np.allclose(prop, oth):
                    continue
                p = wilcoxon(prop, oth).pvalue
                delta = float(np.median(prop - oth))
                rows.append({"escenario": scen, "metrica": metric,
                             "contra": other, "mediana_diff": delta,
                             "mejor": "Propuesto" if delta < 0 else other,
                             "p": p})
                pvals.append(p)
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"].to_numpy())
        out["significativo"] = out["p_holm"] < 0.05
    return out


def main():
    df = pd.read_csv(os.path.join(RES, "confirmatory_runs.csv"))
    os.makedirs(TAB, exist_ok=True)
    report = {}

    print("=" * 74)
    print("H1. Conservacion del ordenamiento, seis estimadores por celda")
    print("=" * 74)
    d1, s1 = h1(df)
    print(s1.round(3).to_string())
    d1.to_csv(os.path.join(TAB, "h1_rank_correlations.csv"), index=False)
    report["H1"] = {k: {kk: (None if pd.isna(vv) else round(float(vv), 4))
                        for kk, vv in v.items()}
                    for k, v in s1.to_dict("index").items()}
    print()
    print("por escenario, objetivo error de seguimiento medio")
    piv = d1[d1.objetivo == "track_rmse"].pivot(
        index="escenario", columns="severidad", values="spearman")
    print(piv.round(2).to_string())

    print()
    print("=" * 74)
    print("H2. Consistencia frente a RMSE como predictor del lazo cerrado")
    print("=" * 74)
    d2, s2 = h2(df)
    for k, v in s2.items():
        print("  %-30s %s" % (k, v if isinstance(v, str) else round(v, 4)))
    d2.to_csv(os.path.join(TAB, "h2_predictors.csv"), index=False)
    report["H2"] = s2

    print()
    print("=" * 74)
    print("H3. Tiempo minimo de permanencia y conmutacion espuria")
    print("=" * 74)
    d3, s3 = h3(df)
    if len(d3):
        agg = d3.groupby("canal")[["conmut_propuesto",
                                   "conmut_sin_permanencia"]].mean()
        print(agg.round(2).to_string())
        print("pruebas %d, significativas tras Holm %d"
              % (len(d3), int((d3.p_holm < 0.05).sum())))
        d3.to_csv(os.path.join(TAB, "h3_switching.csv"), index=False)
        report["H3"] = {"n_pruebas": len(d3),
                        "n_significativas": int((d3.p_holm < 0.05).sum()),
                        "media_propuesto": float(agg["conmut_propuesto"].mean()),
                        "media_sin_permanencia": float(
                            agg["conmut_sin_permanencia"].mean())}

    print()
    print("=" * 74)
    print("Comparacion pareada, severidad uno")
    print("=" * 74)
    hh = head_to_head(df)
    hh.to_csv(os.path.join(TAB, "head_to_head_sev1.csv"), index=False)
    for metric in ("pos_rmse", "track_rmse", "track_max"):
        sub = hh[hh.metrica == metric]
        won = sub[(sub.mejor == "Propuesto") & sub.significativo]
        lost = sub[(sub.mejor != "Propuesto") & sub.significativo]
        print("%-12s pruebas %2d  gana %2d  pierde %2d  empata %2d"
              % (metric, len(sub), len(won), len(lost),
                 len(sub) - len(won) - len(lost)))
    report["head_to_head"] = {
        "n_pruebas": len(hh),
        "n_significativas": int(hh.significativo.sum()) if len(hh) else 0}

    with open(os.path.join(RES, "hypotheses.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print()
    print("escrito results/primary/hypotheses.json y tablas en tables/")


if __name__ == "__main__":
    main()
