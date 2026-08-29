"""Sensibilidades editoriales derivadas de los resultados confirmatorios.

No vuelve a simular. Evalua dos decisiones de resumen.

1. Excluye el escenario nominal y las severidades cero para comprobar que las
   condiciones nominales repetidas no gobiernan las correlaciones de H1 y H2.
2. Exige una permanencia final de cinco segundos dentro de la banda para
   clasificar una corrida como recuperada.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "primary")
TAB = os.path.join(ROOT, "tables")
MAIN = ["EKF_nom", "EKF_mis", "CovMatch", "SageHusa", "Huber", "Propuesto"]
HORIZON = 100.0
FINAL_DWELL = 5.0


def rank_corr(a, b):
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.rank().corr(b.rank()))


def main():
    h1 = pd.read_csv(os.path.join(TAB, "h1_rank_correlations.csv"))
    h2 = pd.read_csv(os.path.join(TAB, "h2_predictors.csv"))
    keep1 = h1[(h1.escenario != "S0_nominal") & (h1.severidad > 0)]
    keep2 = h2[(h2.escenario != "S0_nominal") & (h2.severidad > 0)]

    report = {
        "degraded_only_h1": keep1.groupby("objetivo")["spearman"].mean().to_dict(),
        "degraded_only_h2": {
            "rmse_abs_mean": float(keep2.rho_rmse.abs().mean()),
            "inconsistency_abs_mean": float(keep2.rho_incons.abs().mean()),
            "n_cells": int(len(keep2)),
        },
    }

    data = pd.read_csv(os.path.join(RES, "confirmatory_runs.csv"))
    data = data[data.estimador.isin(MAIN)].copy()
    data["recovery_failure_5s"] = (
        data.recovery > (HORIZON - FINAL_DWELL)
    ).astype(float)

    rows = []
    for (scenario, severity), group in data.groupby(["escenario", "severidad"]):
        means = (group.groupby("estimador")[["pos_rmse", "recovery_failure_5s"]]
                 .mean().reindex(MAIN))
        rho = rank_corr(means.pos_rmse, means.recovery_failure_5s)
        rows.append({"escenario": scenario, "severidad": severity,
                     "rho_rmse_failure": rho})
    recovery = pd.DataFrame(rows)
    recovery.to_csv(os.path.join(TAB, "recovery_sensitivity.csv"), index=False)
    degraded = recovery[(recovery.escenario != "S0_nominal")
                        & (recovery.severidad > 0)]
    report["recovery_5s"] = {
        "mean_rho_all_valid_cells": float(recovery.rho_rmse_failure.mean()),
        "valid_cells_all": int(recovery.rho_rmse_failure.notna().sum()),
        "mean_rho_degraded": float(degraded.rho_rmse_failure.mean()),
        "valid_cells_degraded": int(degraded.rho_rmse_failure.notna().sum()),
    }

    bias = data[(data.escenario == "S2_bias") & (data.severidad == 1.0)]
    bias_summary = (bias.groupby("estimador")
                    .agg(failure_fraction=("recovery_failure_5s", "mean"),
                         median_recovery=("recovery", "median"))
                    .reset_index())
    bias_summary.to_csv(os.path.join(TAB, "recovery_bias_sev1.csv"), index=False)
    report["bias_severity_one"] = bias_summary.to_dict("records")

    path = os.path.join(RES, "sensitivity_review.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
