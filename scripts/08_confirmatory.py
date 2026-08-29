"""Estudio confirmatorio. Barrido completo sobre semillas nuevas.

Condiciones que hacen valida esta corrida, fijadas en docs/DECISIONS.md.
  D12  el protocolo de degradacion esta bloqueado
  D17  el mecanismo propuesto esta bloqueado con sus ocho parametros
  D18  las semillas usan la sal confirmatorio-v1, distinta de la del piloto

Salida. Un archivo tidy con una fila por combinacion de estimador, escenario,
severidad y repeticion. Todo analisis posterior parte de ese archivo y no
vuelve a simular, lo que garantiza que tablas y figuras del manuscrito
provengan de la misma ejecucion.

Uso.
    python scripts/08_confirmatory.py [n_rep]
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from estimhyb import batch_runner, registry  # noqa: E402
from estimhyb.sensors import CHANNELS  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "primary")

# Escenarios en los que tambien se corren las ablaciones. Se eligen antes de
# ver resultados, por criterio de diseno. S0 mide falsa alarma, S2 activa la
# prueba del primer momento, S3 activa la inflacion.
ABLATION_SCENARIOS = ("S0_nominal", "S2_bias", "S3_outlier")


def rows_from(summary, label, family, scenario, severity, n_rep, seconds):
    base = {
        "estimador": label,
        "familia": family,
        "escenario": scenario,
        "severidad": severity,
        "segundos_corrida": seconds,
    }
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
    rows = []
    t_start = time.time()
    total = (len(main_ests) * len(registry.SCENARIOS) * len(registry.SEVERITIES)
             + len(abl_ests) * len(ABLATION_SCENARIOS) * len(registry.SEVERITIES))
    done = 0

    for scenario in registry.SCENARIOS:
        for sev in registry.SEVERITIES:
            tag = "%s_%.2f" % (scenario, sev)
            book = batch_runner.make_book(tag, n_rep,
                                          salt=batch_runner.SALT_CONFIRM)
            deg = registry.degradation_for(scenario, sev)
            plan = [(lab, f, q, "principal") for lab, (f, q) in main_ests.items()]
            if scenario in ABLATION_SCENARIOS:
                plan += [(lab, f, q, "ablacion")
                         for lab, (f, q) in abl_ests.items()]
            for label, factory, q, family in plan:
                t0 = time.time()
                out = batch_runner.run_batch(factory, book, degradation=deg,
                                             q_scale=q)
                s = batch_runner.summarize(out)
                dt = time.time() - t0
                rows += rows_from(s, label, family, scenario, sev, n_rep, dt)
                done += 1
                if done % 10 == 0:
                    el = time.time() - t_start
                    print("%4d/%d  %6.1f min transcurridos  %6.1f min estimados"
                          % (done, total, el / 60.0, el / done * total / 60.0),
                          flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "confirmatory_runs.csv")
    df.to_csv(path, index=False, encoding="utf-8")
    meta = {
        "fecha": "2026-08-27",
        "n_rep": n_rep,
        "sal": batch_runner.SALT_CONFIRM,
        "estimadores": list(main_ests) + list(abl_ests),
        "escenarios": list(registry.SCENARIOS),
        "severidades": list(registry.SEVERITIES),
        "filas": len(df),
        "minutos": round((time.time() - t_start) / 60.0, 2),
    }
    import json
    with open(os.path.join(OUT, "confirmatory_meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print("escrito %s con %d filas en %.1f min"
          % (path, len(df), meta["minutos"]))


if __name__ == "__main__":
    main()
