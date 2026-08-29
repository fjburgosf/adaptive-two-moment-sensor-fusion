"""Prueba de humo de la planta y del protocolo de degradación.

No produce resultados del estudio. Solo verifica que los módulos corren, que
las tasas de muestreo son las esperadas y que la severidad produce el efecto
declarado sobre la dispersión de las mediciones.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from estimhyb import plant, sensors  # noqa: E402


def run(duration=20.0, dt=0.01, degradation=None, seed=1):
    rng = np.random.default_rng(seed)
    p = plant.PlantParams()
    suite = sensors.SensorSuite(dt, degradation=degradation, rng=rng,
                                duration=duration)
    n = int(duration / dt)
    xi = np.zeros(plant.STATE_DIM)
    counts = {c: 0 for c in sensors.CHANNELS}
    resid = {c: [] for c in sensors.CHANNELS}
    for k in range(n):
        t = k * dt
        xi = plant.step(xi, [1.2, 0.4 * np.sin(0.5 * t)], p, dt, rng)
        z = suite.measure(xi, t)
        for c, zc in z.items():
            counts[c] += 1
            resid[c].append(zc - sensors.H_MATRICES[c] @ xi)
    return counts, {c: np.array(v) for c, v in resid.items()}, n


def main():
    dt, duration = 0.01, 20.0
    counts, resid, n = run(duration, dt)
    print("== nominal, %d pasos de %.3f s" % (n, dt))
    for c in sensors.CHANNELS:
        rate = counts[c] / duration
        expected = 1.0 / (dt * sensors.DECIMATION[c])
        std = resid[c].std(axis=0)
        print("  %-5s n=%5d  tasa=%6.1f Hz (esperada %6.1f)  std=%s"
              % (c, counts[c], rate, expected, np.round(std, 4)))

    print("\n== barrido de severidad sobre gnss, modo noise")
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        deg = {"gnss": sensors.ChannelDegradation("noise", s)}
        _, r, _ = run(duration, dt, deg, seed=2)
        ratio = r["gnss"].std(axis=0).mean() / sensors.NOMINAL_STD["gnss"].mean()
        pred = 1.0 + (sensors.SEV_NOISE_FACTOR - 1.0) * s
        print("  s=%.2f  razon observada=%5.2f  predicha=%5.2f" % (s, ratio, pred))

    # El canal gnss entrega cinco muestras por segundo, de modo que una
    # corrida corta no basta para estimar la disponibilidad. Se usa la
    # duración real de los escenarios y se promedia sobre varias semillas.
    long_dur = 120.0
    print()
    print("== disponibilidad bajo dropout en gnss, %.0f s, 10 semillas"
          % long_dur)
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        deg = {"gnss": sensors.ChannelDegradation("dropout", s)}
        base = long_dur / (dt * sensors.DECIMATION["gnss"])
        av = [run(long_dur, dt, deg, seed=100 + j)[0]["gnss"] / base
              for j in range(10)]
        pred = 1.0 / (1.0 + sensors.SEV_DROPOUT_RATE
                      * sensors.SEV_DROPOUT_MEAN * s * s)
        print("  s=%.2f  disponibilidad=%.3f (sd %.3f)  predicha=%.3f"
              % (s, float(np.mean(av)), float(np.std(av)), pred))

    print("\n== sesgo terminal bajo modo bias en mag")
    for s in (0.0, 0.5, 1.0):
        deg = {"mag": sensors.ChannelDegradation("bias", s)}
        _, r, _ = run(duration, dt, deg, seed=4)
        tail = r["mag"][-20:].mean()
        pred = sensors.SEV_BIAS_SIGMA * s * sensors.NOMINAL_STD["mag"][0]
        print("  s=%.2f  sesgo final=%7.4f rad  predicho=%7.4f" % (s, tail, pred))


if __name__ == "__main__":
    main()
