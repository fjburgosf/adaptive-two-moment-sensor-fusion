"""Verificación del controlador con realimentación de estado verdadero.

Sin estimador. Establece la cota superior de desempeño en lazo cerrado y
confirma que la ley de seguimiento converge sobre la planta con retardo
actuador. No es un resultado del estudio.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from estimhyb import control, plant, reference  # noqa: E402


def main():
    dt, duration = 0.01, 120.0
    ref = reference.Lemniscate()
    gains = control.KanayamaGains()
    p = plant.PlantParams()
    vmin, vmax, ommax = ref.speed_bounds()
    print("referencia, v_r en [%.3f, %.3f] m/s, |omega_r| max %.3f rad/s"
          % (vmin, vmax, ommax))
    print("limites de planta, v_max %.2f, w_max %.2f" % (p.v_max, p.w_max))

    for label, x0 in (("arranque sobre la curva", None),
                      ("arranque desplazado", np.array([2.0, -2.0, 1.0]))):
        rng = np.random.default_rng(7)
        xi = np.zeros(plant.STATE_DIM)
        r0 = ref.pose(0.0)
        xi[plant.IX], xi[plant.IY], xi[plant.ITH] = r0[0], r0[1], r0[2]
        if x0 is not None:
            xi[plant.IX] += x0[0]
            xi[plant.IY] += x0[1]
            xi[plant.ITH] += x0[2]
        errs, effort = [], 0.0
        n = int(duration / dt)
        for k in range(n):
            t = k * dt
            rp = ref.pose(t)
            u, e = control.command(xi, rp, gains)
            u = plant.saturate_input(u, p)
            effort += float(u @ u) * dt
            errs.append(np.hypot(e[0], e[1]))
            xi = plant.step(xi, u, p, dt, rng)
        errs = np.array(errs)
        settled = errs[int(20.0 / dt):]
        print("%-26s  error final medio %.4f m  maximo %.4f m  esfuerzo %.1f"
              % (label, settled.mean(), settled.max(), effort))


if __name__ == "__main__":
    main()
