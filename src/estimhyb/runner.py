"""Ejecutor escalar. Implementacion de verificacion, no de produccion.

Ver el aviso en estimhyb.filters.ekf. La produccion usa
estimhyb.batch_runner.run_batch. Este modulo recorre una repeticion a la vez
y consume los generadores aleatorios en un orden distinto, de modo que sus
resultados coinciden con los del ejecutor por lotes solo en distribucion, no
realizacion a realizacion.

Ejecución de un escenario en lazo cerrado.

Secuencia de un paso. El controlador consume el estado estimado, nunca el
verdadero, salvo en la corrida de referencia que sirve como cota superior.

    1. se calcula la referencia en el instante t
    2. el controlador produce el comando a partir del estado estimado
    3. el comando se satura y se aplica a la planta verdadera
    4. el filtro propaga con ese mismo comando
    5. los sensores producen las mediciones disponibles
    6. el filtro actualiza con lo que haya llegado

Las semillas se derivan de forma determinista del escenario y de la
repetición, de modo que dos estimadores distintos ven exactamente la misma
realización de ruido de proceso y de medición. Esa igualdad es lo que permite
la comparación pareada.
"""
import numpy as np

from . import control, plant, reference
from .geometry import wrap_pi
from .sensors import CHANNELS, SensorSuite


def make_rngs(scenario_id, replicate):
    """Dos generadores independientes, uno para la planta y otro para sensores.

    Separarlos garantiza que un cambio en el protocolo de sensores no altere
    la realización del ruido de proceso, lo que mantiene comparables las
    corridas entre escenarios.
    """
    seq = np.random.SeedSequence([abs(hash(scenario_id)) % (2 ** 32), replicate])
    a, b = seq.spawn(2)
    return np.random.default_rng(a), np.random.default_rng(b)


def initial_covariance():
    """P0. Incertidumbre inicial deliberadamente moderada, no despreciable."""
    return np.diag([1.0 ** 2, 1.0 ** 2, np.deg2rad(15.0) ** 2,
                    0.2 ** 2, 0.1 ** 2, 0.01 ** 2])


def run_scenario(filter_factory, degradation=None, duration=120.0, dt=0.01,
                 scenario_id="nominal", replicate=0, gains=None, params=None,
                 ref=None, oracle=False):
    """Corre un escenario y devuelve las series temporales de diagnóstico."""
    gains = gains or control.KanayamaGains()
    p = params or plant.PlantParams()
    ref = ref or reference.Lemniscate()
    rng_plant, rng_sensor = make_rngs(scenario_id, replicate)

    suite = SensorSuite(dt, degradation=degradation, rng=rng_sensor,
                        duration=duration)

    r0 = ref.pose(0.0)
    xi = np.zeros(plant.STATE_DIM)
    xi[plant.IX], xi[plant.IY], xi[plant.ITH] = r0[0], r0[1], r0[2]
    xi[plant.IV] = r0[3]

    P0 = initial_covariance()
    x0 = xi + np.sqrt(np.diag(P0)) * rng_plant.standard_normal(plant.STATE_DIM)
    x0[plant.ITH] = wrap_pi(x0[plant.ITH])
    kf = filter_factory(x0, P0, p, dt)

    n = int(round(duration / dt))
    out = {
        "t": np.zeros(n),
        "track_err": np.zeros(n),
        "head_err": np.zeros(n),
        "effort": np.zeros(n),
        "est_err": np.zeros((n, plant.STATE_DIM)),
        "nees": np.zeros(n),
        "true": np.zeros((n, plant.STATE_DIM)),
        "est": np.zeros((n, plant.STATE_DIM)),
    }
    for c in CHANNELS:
        out["nis_" + c] = np.full(n, np.nan)
        out["infl_" + c] = np.full(n, np.nan)
        out["acc_" + c] = np.full(n, np.nan)

    for k in range(n):
        t = k * dt
        rp = ref.pose(t)
        source = xi if oracle else kf.x
        u, e = control.command(source, rp, gains)
        u = plant.saturate_input(u, p)

        out["t"][k] = t
        e_true = control.tracking_error(xi, rp)
        out["track_err"][k] = np.hypot(e_true[0], e_true[1])
        out["head_err"][k] = abs(e_true[2])
        out["effort"][k] = float(u @ u)
        out["true"][k] = xi
        out["est"][k] = kf.x
        nees, err = kf.nees(xi)
        out["nees"][k] = nees
        out["est_err"][k] = err

        xi = plant.step(xi, u, p, dt, rng_plant)
        kf.predict(u)
        z = suite.measure(xi, t + dt)
        kf.update(z, t + dt)
        for c, val in kf.diag.nis.items():
            out["nis_" + c][k] = val
            out["infl_" + c][k] = kf.diag.inflation[c]
            out["acc_" + c][k] = 1.0 if kf.diag.accepted[c] else 0.0

    out["dt"] = dt
    out["scenario_id"] = scenario_id
    out["replicate"] = replicate
    return out
