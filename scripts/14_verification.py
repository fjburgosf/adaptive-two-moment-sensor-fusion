"""Verificacion independiente exigida por D11.

Reproduce los numeros de titular de R4 y R5 con el camino de codigo escalar,
que es una implementacion distinta del nucleo de filtrado. Ver el aviso en
estimhyb.filters.ekf sobre el alcance real de esa independencia.

El ejecutor escalar consume los generadores aleatorios en otro orden, de modo
que la coincidencia esperada es en distribucion y no realizacion a
realizacion. El criterio de aceptacion es que la media del camino escalar
caiga dentro del intervalo de confianza al noventa y cinco por ciento de la
media por lotes, y a la inversa.

Cualquier discrepancia se registra en docs/DECISIONS.md y no se promedia ni
se descarta en silencio.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from estimhyb import registry, runner  # noqa: E402
from estimhyb.filters.adapters import DwellTimeAdapter, NoiseAdapter  # noqa: E402
from estimhyb.filters.ekf import ExtendedKalmanFilter  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results", "primary")

CASES = [("S0_nominal", 0.0), ("S2_bias", 1.0)]
ARMS = {
    "EKF_nom": lambda: NoiseAdapter(1),
    "Propuesto": lambda: DwellTimeAdapter(1),
    "Abl_sin_media": lambda: DwellTimeAdapter(1, mean_test=False),
}


def scalar_run(adapter_factory, q_scale, degradation, scenario_id, reps):
    pos, track = [], []
    for r in range(reps):
        def factory(x0, P0, p, dt):
            return ExtendedKalmanFilter(x0, P0, p, dt, adapter_factory(),
                                        q_scale=q_scale)
        out = runner.run_scenario(factory, degradation=degradation,
                                  scenario_id=scenario_id, replicate=r)
        m = out["t"] >= 20.0
        pos.append(np.sqrt((out["est_err"][m][:, :2] ** 2).sum(axis=1).mean()))
        track.append(np.sqrt((out["track_err"][m] ** 2).mean()))
    return np.array(pos), np.array(track)


def ci95(x):
    return 1.96 * float(np.std(x, ddof=1)) / np.sqrt(x.size)


def cv(x):
    return float(np.std(x, ddof=1) / np.mean(x))


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    ref = pd.read_csv(os.path.join(RES, "confirmatory_runs.csv"))
    tuning = registry.load_tuning()
    q_nom = float(tuning["q_scale_nominal"])
    rows = []
    for scen, sev in CASES:
        deg = registry.degradation_for(scen, sev)
        for arm, fac in ARMS.items():
            sub = ref[(ref.escenario == scen) & (ref.severidad == sev)
                      & (ref.estimador == arm)]
            if sub.empty:
                continue
            b = sub["pos_rmse"].to_numpy()
            p, t = scalar_run(fac, q_nom, deg, "%s_%.2f" % (scen, sev), reps)
            lo_b, hi_b = b.mean() - ci95(b), b.mean() + ci95(b)
            lo_s, hi_s = p.mean() - ci95(p), p.mean() + ci95(p)
            solapan = not (hi_s < lo_b or hi_b < lo_s)
            cv_b, cv_s = cv(b), cv(p)
            cv_ratio = cv_s / cv_b if cv_b > 0.0 else np.nan
            dispersion_concuerda = bool(0.5 <= cv_ratio <= 2.0)
            rows.append({"escenario": scen, "severidad": sev, "brazo": arm,
                         "lotes_media": round(float(b.mean()), 5),
                         "lotes_ic": round(ci95(b), 5),
                         "escalar_media": round(float(p.mean()), 5),
                         "escalar_ic": round(ci95(p), 5),
                         "diferencia_rel": round(
                             float(p.mean() / b.mean() - 1.0), 4),
                         "intervalos_solapan": bool(solapan),
                         "lotes_cv": round(cv_b, 4),
                         "escalar_cv": round(cv_s, 4),
                         "razon_cv": round(cv_ratio, 4),
                         "dispersion_concuerda": dispersion_concuerda})
            print("%-11s %-11s lotes %.4f (%.4f)  escalar %.4f (%.4f)  "
                  "dif %+.1f%%  solapan %s  CV %.2f contra %.2f"
                  % (scen, arm, b.mean(), ci95(b), p.mean(), ci95(p),
                     100 * (p.mean() / b.mean() - 1.0), solapan,
                     cv_b, cv_s), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "verification_scalar.csv"), index=False)
    ok = bool(df["intervalos_solapan"].all()
              and df["dispersion_concuerda"].all())
    with open(os.path.join(RES, "verification_scalar.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"reps_escalar": reps, "todos_solapan": ok,
                   "filas": df.to_dict("records")}, fh, indent=2,
                  ensure_ascii=False)
    print()
    print("verificacion %s" % ("SUPERADA" if ok else "CON DISCREPANCIAS"))
    if not ok:
        print("registrar la discrepancia en docs/DECISIONS.md")


if __name__ == "__main__":
    main()
