"""Control negativo de D9 y prueba directa de H1.

Diseno de dos por dos. Se cruzan planta lineal contra planta no lineal, y
escenario nominal contra escenario degradado. En cada celda se corre la misma
familia de estimadores, obtenida escalando la covarianza de medicion por un
factor gamma declarado. La familia es lineal e invariante en el tiempo, de
modo que en la celda lineal y nominal rige el principio de separacion.

Lo que se mide en cada celda es la correlacion de ordenamientos entre

    error de estimacion de posicion, medido en lazo abierto
    error de seguimiento, medido en lazo cerrado

Prediccion teorica. En la celda lineal y nominal la correlacion debe ser
cercana a mas uno. Si no lo es, hay un error de implementacion y no un
hallazgo. Esa celda es el control negativo.

H1 queda apoyada si la correlacion cae de forma apreciable en la celda no
lineal y degradada.

Salida, results/primary/negative_control.csv
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from estimhyb import batch, batch_runner, linear, plant, registry  # noqa: E402
from estimhyb.filters.adapters import ScaledRAdapter  # noqa: E402
from estimhyb.sensors import (CHANNELS, H_MATRICES, ChannelDegradation,  # noqa: E402
                              nominal_R)

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "primary")

# Familia de estimadores. El factor multiplica la covarianza de medicion de
# los cuatro canales. Con uno el filtro esta correctamente especificado.
GAMMAS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 12.0, 30.0)

V0 = 1.0          # velocidad de la recta de referencia, m/s
WARM = 20.0       # transitorio descartado, s


def run_linear(gamma, degradation, book, duration=120.0, dt=0.01):
    """Planta linealizada con regulador cuadratico y filtro lineal."""
    p = plant.PlantParams()
    K, Ad, Bd = linear.design_lqr(p, V0, dt)
    n_rep = book.n_rep
    n = int(round(duration / dt))
    sensors = batch.BatchSensors(dt, duration, degradation, book, n_rep)

    Q = plant.process_noise_cov(p, dt)
    R_nom = {c: nominal_R(c) for c in CHANNELS}
    P0 = batch_runner.initial_covariance()
    delta = np.zeros((n_rep, plant.STATE_DIM))
    d_hat = book.x0 * np.sqrt(np.diag(P0))
    kf = linear.LinearKF(d_hat, np.broadcast_to(
        P0, (n_rep, plant.STATE_DIM, plant.STATE_DIM)).copy(),
        Ad, Q, R_nom, gamma=gamma)

    scale = np.zeros(plant.STATE_DIM)
    scale[plant.IV] = p.sigma_v * np.sqrt(dt)
    scale[plant.IW] = p.sigma_w * np.sqrt(dt)
    scale[plant.IBG] = p.sigma_bg * np.sqrt(dt)

    track = np.zeros((n, n_rep))
    est_sq = np.zeros((n, n_rep, 2))
    u_log = np.zeros((n, n_rep, 2))
    d_log = np.zeros((n, n_rep, plant.STATE_DIM))

    for k in range(n):
        t = k * dt
        ref = np.zeros(plant.STATE_DIM)
        ref[plant.IX], ref[plant.IV] = V0 * t, V0
        absolute = delta + ref
        meas = sensors.measure(absolute, t, k)
        shifted = {c: (z - H_MATRICES[c] @ ref, av)
                   for c, (z, av) in meas.items()}
        kf.update(shifted, H_MATRICES)

        err = kf.x - delta
        est_sq[k] = err[:, :2] ** 2
        track[k] = np.hypot(delta[:, plant.IX], delta[:, plant.IY])
        d_log[k] = delta

        u = -kf.x @ K.T
        u_abs = np.column_stack([u[:, 0] + V0, u[:, 1]])
        u_abs = batch.saturate_batch(u_abs, p)
        u = np.column_stack([u_abs[:, 0] - V0, u_abs[:, 1]])
        u_log[k] = u

        delta = delta @ Ad.T + u @ Bd.T + book.w[k] * scale
        kf.predict(u, Bd)

    m = np.arange(n) * dt >= WARM
    return {
        "pos_rmse": np.sqrt(est_sq[m].sum(axis=2).mean(axis=0)),
        "track_rmse": np.sqrt((track[m] ** 2).mean(axis=0)),
        "lq_cost": linear.lq_cost(d_log[m], u_log[m], dt),
    }


def run_nonlinear(gamma, degradation, book):
    """Planta no lineal con la misma familia de estimadores."""
    out = batch_runner.run_batch(lambda n: ScaledRAdapter(n, gamma=gamma),
                                 book, degradation=degradation)
    s = batch_runner.summarize(out, warm=WARM)
    return {"pos_rmse": s["pos_rmse"], "track_rmse": s["track_rmse"],
            "lq_cost": s["track_rmse"] ** 2 + s["effort"] * 0.0}


def main():
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    deg_none = None
    deg_bad = {"gnss": ChannelDegradation("noise", 1.0, 20.0)}
    cells = [
        ("lineal", "nominal", run_linear, deg_none),
        ("lineal", "degradado", run_linear, deg_bad),
        ("no lineal", "nominal", run_nonlinear, deg_none),
        ("no lineal", "degradado", run_nonlinear, deg_bad),
    ]

    rows, summary = [], []
    for planta, escenario, fn, deg in cells:
        tag = "NC_%s_%s" % (planta.replace(" ", ""), escenario)
        book = batch_runner.make_book(tag, n_rep,
                                      salt=batch_runner.SALT_CONFIRM)
        est, cl = [], []
        t0 = time.time()
        for g in GAMMAS:
            r = fn(g, deg, book)
            est.append(float(np.mean(r["pos_rmse"])))
            cl.append(float(np.mean(r["track_rmse"])))
            for i in range(n_rep):
                rows.append({"planta": planta, "escenario": escenario,
                             "gamma": g, "repeticion": i,
                             "pos_rmse": float(r["pos_rmse"][i]),
                             "track_rmse": float(r["track_rmse"][i]),
                             "lq_cost": float(r["lq_cost"][i])})
        rho = spearmanr(est, cl).statistic
        tau = kendalltau(est, cl).statistic
        summary.append({"planta": planta, "escenario": escenario,
                        "spearman": round(float(rho), 4),
                        "kendall": round(float(tau), 4),
                        "segundos": round(time.time() - t0, 1)})
        print("%-10s %-10s  spearman %+.3f  kendall %+.3f  (%.0f s)"
              % (planta, escenario, rho, tau, time.time() - t0), flush=True)
        print("   gamma      %s" % "  ".join("%7.2f" % g for g in GAMMAS))
        print("   rmse est   %s" % "  ".join("%7.4f" % v for v in est))
        print("   rmse track %s" % "  ".join("%7.4f" % v for v in cl))

    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "negative_control.csv"),
                              index=False, encoding="utf-8")
    with open(os.path.join(OUT, "negative_control_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"n_rep": n_rep, "gammas": list(GAMMAS),
                   "warm_s": WARM, "v0": V0, "celdas": summary},
                  fh, indent=2, ensure_ascii=False)
    print()
    print("control negativo, celda lineal y nominal, spearman %+.3f"
          % summary[0]["spearman"])
    print("si ese valor no es cercano a mas uno, hay un error de implementacion")


if __name__ == "__main__":
    main()
