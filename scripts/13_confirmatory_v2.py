"""Segunda ronda confirmatoria. Anade el brazo de compensacion de sesgo.

Por que hace falta una segunda ronda. El brazo de compensacion fue disenado
despues de observar R4, que son datos de la primera ronda. Segun la
disciplina de D15, evaluarlo sobre esas mismas realizaciones seria ajustar al
conjunto de prueba. Por eso esta corrida usa la sal confirmatorio-v2 y
regenera todas las realizaciones.

Se vuelven a correr **todos** los brazos, no solo los nuevos, porque las
comparaciones deben ser pareadas sobre la misma realizacion.

Beneficio adicional. Los diez brazos comunes a las dos rondas quedan
evaluados sobre semillas independientes, lo que da una replicacion gratuita.
El acuerdo entre v1 y v2 en esos brazos es evidencia de estabilidad y se
verifica en scripts/14_replication_check.py.

Uso.
    python scripts/13_confirmatory_v2.py [n_rep]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from estimhyb import batch_runner, registry  # noqa: E402
from estimhyb.sensors import CHANNELS  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "primary")
ABLATION_SCENARIOS = ("S0_nominal", "S2_bias", "S3_outlier")


def rows_from(summary, label, family, scenario, severity, n_rep, seconds):
    base = {"estimador": label, "familia": family, "escenario": scenario,
            "severidad": severity, "segundos_corrida": seconds}
    cols = ["track_rmse", "track_max", "head_rmse", "effort", "nees",
            "pos_rmse", "theta_rmse", "recovery"]
    for c in CHANNELS:
        cols += ["nis_" + c, "acc_" + c, "infl_" + c, "sw_" + c]
    out = []
    for r in range(n_rep):
        row = dict(base)
        row["repeticion"] = r
        for c in cols:
            row[c] = float(summary[c][r])
        out.append(row)
    return out


def main():
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    main_ests = registry.estimators()
    abl_ests = registry.ablations()
    comp_ests = registry.compensating()
    rows = []
    t_start = time.time()
    total = (len(main_ests) * 30 + len(abl_ests) * len(ABLATION_SCENARIOS) * 5
             + len(comp_ests) * 30)
    done = 0

    for scenario in registry.SCENARIOS:
        for sev in registry.SEVERITIES:
            tag = "%s_%.2f" % (scenario, sev)
            book = batch_runner.make_book(tag, n_rep,
                                          salt=batch_runner.SALT_CONFIRM2)
            deg = registry.degradation_for(scenario, sev)
            plan = [(lab, f, q, None, "principal")
                    for lab, (f, q) in main_ests.items()]
            plan += [(lab, f, q, cls, "compensacion")
                     for lab, (f, q, cls) in comp_ests.items()]
            if scenario in ABLATION_SCENARIOS:
                plan += [(lab, f, q, None, "ablacion")
                         for lab, (f, q) in abl_ests.items()]
            for label, factory, q, cls, family in plan:
                t0 = time.time()
                out = batch_runner.run_batch(factory, book, degradation=deg,
                                             q_scale=q, filter_class=cls)
                s = batch_runner.summarize(out)
                dt = time.time() - t0
                rows += rows_from(s, label, family, scenario, sev, n_rep, dt)
                done += 1
                if done % 15 == 0:
                    el = time.time() - t_start
                    print("%4d/%d  %6.1f min  estimado %6.1f min"
                          % (done, total, el / 60.0, el / done * total / 60.0),
                          flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "confirmatory_v2_runs.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    meta = {"fecha": "2026-08-27", "n_rep": n_rep,
            "sal": batch_runner.SALT_CONFIRM2,
            "brazos": list(main_ests) + list(comp_ests) + list(abl_ests),
            "filas": len(df),
            "minutos": round((time.time() - t_start) / 60.0, 2)}
    with open(os.path.join(OUT, "confirmatory_v2_meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print("escrito %s con %d filas en %.1f min"
          % (path, len(df), meta["minutos"]))


if __name__ == "__main__":
    main()
