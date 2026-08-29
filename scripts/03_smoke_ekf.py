"""Verificación del filtro extendido nominal.

Comprueba tres cosas en ausencia de degradación.
    1. el filtro converge y el error de estimación es acotado
    2. el NEES promedio se aproxima a la dimensión del estado, que es seis
    3. el NIS promedio de cada canal se aproxima a sus grados de libertad
Si alguna falla, el problema es de implementación y no un hallazgo.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from estimhyb import plant, runner  # noqa: E402
from estimhyb.filters.adapters import NoiseAdapter  # noqa: E402
from estimhyb.filters.ekf import ExtendedKalmanFilter  # noqa: E402
from estimhyb.sensors import CHANNELS  # noqa: E402


def factory(x0, P0, p, dt):
    return ExtendedKalmanFilter(x0, P0, p, dt, NoiseAdapter(), name="ekf_nom")


def main():
    reps = 12
    warm = 20.0
    nees, track, est = [], [], []
    nis = {c: [] for c in CHANNELS}
    for r in range(reps):
        out = runner.run_scenario(factory, scenario_id="nominal", replicate=r)
        m = out["t"] >= warm
        nees.append(out["nees"][m].mean())
        track.append(out["track_err"][m].mean())
        est.append(np.abs(out["est_err"][m]).mean(axis=0))
        for c in CHANNELS:
            v = out["nis_" + c][m]
            nis[c].append(np.nanmean(v))

    print("repeticiones %d, descarte inicial %.0f s" % (reps, warm))
    print("NEES medio %.3f  (esperado %d, dimension del estado)"
          % (float(np.mean(nees)), plant.STATE_DIM))
    for c in CHANNELS:
        dof = 2 if c == "gnss" else 1
        print("  NIS %-5s %.3f  (esperado %d)" % (c, float(np.mean(nis[c])), dof))
    est = np.array(est).mean(axis=0)
    print("error absoluto medio por estado")
    for i, nm in enumerate(plant.STATE_NAMES):
        print("  %-6s %.5f %s" % (nm, est[i], plant.STATE_UNITS[i]))
    print("error de seguimiento medio %.4f m" % float(np.mean(track)))


if __name__ == "__main__":
    main()
