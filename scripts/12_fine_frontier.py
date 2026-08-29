"""Rejilla fina de severidad para localizar la frontera de consistencia.

Motivo. La rejilla del estudio confirmatorio, con paso de 0.25, es demasiado
gruesa en los escenarios donde la consistencia se pierde de inmediato. Entre
severidad cero y 0.25 el NEES pasa de seis a mas de mil, de modo que
interpolar de forma lineal en ese tramo produce un umbral sin significado.

Este barrido usa una rejilla logaritmica cerca de cero sobre los dos
escenarios afectados. Las severidades son nuevas y no se solapan con las del
estudio confirmatorio, de modo que no se reusa ni se re-analiza el mismo
conjunto. La sal es la misma, confirmatorio-v1, porque el mecanismo ya estaba
bloqueado antes de esta corrida y no se ajusta nada con ella.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from estimhyb import batch_runner, registry  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FINE = (0.02, 0.04, 0.07, 0.12, 0.18, 0.25)
SCENARIOS = ("S2_bias", "S5_combined")
MAIN = ["EKF_nom", "CovMatch", "SageHusa", "Huber", "Propuesto"]


def main():
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    hi = chi2.ppf(0.975, 6 * n_rep) / n_rep
    ests = {k: v for k, v in registry.estimators().items() if k in MAIN}
    rows = []
    for scen in SCENARIOS:
        for sev in FINE:
            tag = "%s_fine_%.3f" % (scen, sev)
            book = batch_runner.make_book(tag, n_rep,
                                          salt=batch_runner.SALT_CONFIRM)
            deg = registry.degradation_for(scen, sev)
            for lab, (f, q) in ests.items():
                out = batch_runner.run_batch(f, book, degradation=deg,
                                             q_scale=q)
                s = batch_runner.summarize(out)
                rows.append({"escenario": scen, "severidad": sev,
                             "estimador": lab,
                             "nees": float(s["nees"].mean()),
                             "pos_rmse": float(s["pos_rmse"].mean())})
            print("%s sev %.2f listo" % (scen, sev), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "tables", "fine_frontier.csv"), index=False)
    print()
    print("umbral superior de la banda de consistencia %.3f" % hi)
    for scen in SCENARIOS:
        print()
        print("== %s, NEES promedio" % scen)
        piv = df[df.escenario == scen].pivot(index="estimador",
                                             columns="severidad",
                                             values="nees").reindex(MAIN)
        print(piv.round(1).to_string())
        print("primera severidad con perdida de consistencia")
        for est in MAIN:
            row = piv.loc[est]
            over = row[row > hi]
            first = over.index[0] if len(over) else None
            print("   %-10s %s" % (est, "%.2f" % first if first else "ninguna"))
    with open(os.path.join(ROOT, "results", "primary", "fine_frontier.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"n_rep": n_rep, "banda_superior": float(hi),
                   "severidades": list(FINE),
                   "datos": df.to_dict("records")}, fh, indent=2,
                  ensure_ascii=False)


if __name__ == "__main__":
    main()
