"""Sintonizacion del filtro extendido nominal.

Define de forma operacional que significa bien sintonizado. Se busca el
escalar sobre Q que lleva el NEES promedio del escenario sin degradacion a la
dimension del estado. La sintonizacion se hace una sola vez, antes de que
exista ningun esquema adaptativo, y queda fijada para todo el estudio.

El comparador mal sintonizado se define despues como un desplazamiento fijo y
declarado respecto de este valor, no como un valor elegido a conveniencia.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from estimhyb import batch_runner, plant  # noqa: E402
from estimhyb.filters.adapters import NoiseAdapter  # noqa: E402


def evaluate(book, q_scale, warm=20.0):
    out = batch_runner.run_batch(lambda n: NoiseAdapter(n), book,
                                 q_scale=q_scale)
    s = batch_runner.summarize(out, warm=warm)
    return (float(s["nees"].mean()), float(s["nees"].std()),
            float(s["pos_rmse"].mean()), float(s["track_rmse"].mean()))


def main():
    n_rep = 60
    book = batch_runner.make_book("nominal", n_rep)
    target = plant.STATE_DIM
    print("objetivo de NEES %d, repeticiones %d" % (target, n_rep))
    print("%-9s %9s %8s %11s %11s" % ("q_scale", "NEES", "sd", "rmse pos", "track"))
    rows = []
    for q in (0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2):
        nees, sd, rmse, track = evaluate(book, q)
        rows.append((q, nees, sd, rmse, track))
        print("%-9.2f %9.3f %8.3f %11.4f %11.4f" % (q, nees, sd, rmse, track))

    qs = np.array([r[0] for r in rows])
    ns = np.array([r[1] for r in rows])
    # Interpolacion lineal sobre log(q) para localizar el cruce con el objetivo.
    order = np.argsort(ns)
    q_star = float(np.exp(np.interp(target, ns[order], np.log(qs[order]))))
    nees, sd, rmse, track = evaluate(book, q_star)
    print()
    print("q_scale sintonizado %.3f  NEES %.3f (sd %.3f)  rmse pos %.4f"
          % (q_star, nees, sd, rmse))

    cfg = {
        "q_scale_nominal": round(q_star, 4),
        "nees_target": target,
        "nees_achieved": round(nees, 4),
        "n_rep_tuning": n_rep,
        "warm_s": 20.0,
        "note": ("Sintonizado el 2026-08-27 sobre el escenario sin degradacion. "
                 "No se re-sintoniza en ningun escenario degradado."),
    }
    path = os.path.join(os.path.dirname(__file__), "..", "configs",
                        "tuning.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    print("escrito configs/tuning.json")


if __name__ == "__main__":
    main()
