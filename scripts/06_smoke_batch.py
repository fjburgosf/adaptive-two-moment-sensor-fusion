"""Verificacion del ejecutor por lotes.

Comprueba consistencia en el escenario nominal y mide el tiempo de ejecucion
para dimensionar el presupuesto de computo del estudio completo.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from estimhyb import batch_runner, plant  # noqa: E402
from estimhyb.filters.adapters import NoiseAdapter  # noqa: E402
from estimhyb.sensors import CHANNELS  # noqa: E402


def main():
    n_rep = 40
    book = batch_runner.make_book("nominal", n_rep)
    t0 = time.time()
    out = batch_runner.run_batch(lambda n: NoiseAdapter(n), book)
    dur = time.time() - t0
    s = batch_runner.summarize(out)

    print("repeticiones %d, tiempo %.2f s, por repeticion %.3f s"
          % (n_rep, dur, dur / n_rep))
    print("NEES medio %.3f  desviacion entre repeticiones %.3f  (esperado %d)"
          % (s["nees"].mean(), s["nees"].std(), plant.STATE_DIM))
    for c in CHANNELS:
        dof = 2 if c == "gnss" else 1
        print("  NIS %-5s %.3f  (esperado %d)" % (c, s["nis_" + c].mean(), dof))
    print("rmse de posicion %.4f m" % s["pos_rmse"].mean())
    print("error de seguimiento rmse %.4f m" % s["track_rmse"].mean())
    print()
    print("presupuesto, 6 estimadores x 6 escenarios x 5 severidades:")
    print("  %.1f minutos con %d repeticiones" % (dur * 6 * 6 * 5 / 60.0, n_rep))


if __name__ == "__main__":
    main()
