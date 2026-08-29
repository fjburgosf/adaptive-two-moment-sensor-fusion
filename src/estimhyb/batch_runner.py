"""Ejecutor por lotes de un escenario completo.

Orden de un paso, que es el orden estándar de un filtro con realimentación.

    1. los sensores observan el estado verdadero actual
    2. el filtro actualiza y produce el estado a posteriori
    3. se registran las métricas con ese estado a posteriori
    4. el controlador produce el comando a partir del estado estimado
    5. la planta avanza con el comando aplicado
    6. el filtro propaga con el mismo comando

Todas las repeticiones avanzan a la vez. La realización aleatoria proviene de
un NoiseBook pregenerado y compartido, de modo que dos estimadores distintos
reciben exactamente las mismas entradas aleatorias.
"""
import numpy as np

from . import batch, control, plant, reference
from .geometry import wrap_pi
from .noisebook import NoiseBook
from .plant import ITH, IV, IX, IY, STATE_DIM
from .sensors import CHANNELS


def initial_covariance():
    """P0. Incertidumbre inicial moderada, no despreciable."""
    return np.diag([1.0 ** 2, 1.0 ** 2, np.deg2rad(15.0) ** 2,
                    0.2 ** 2, 0.1 ** 2, 0.01 ** 2])


def command_batch(x, ref_pose, gains):
    """Ley de Kanayama evaluada sobre todas las repeticiones."""
    x_r, y_r, th_r, v_r, om_r = ref_pose
    th = x[:, ITH]
    dx = x_r - x[:, IX]
    dy = y_r - x[:, IY]
    c, s = np.cos(th), np.sin(th)
    e_x = c * dx + s * dy
    e_y = -s * dx + c * dy
    e_th = wrap_pi(th_r - th)
    v_cmd = v_r * np.cos(e_th) + gains.k_x * e_x
    om_cmd = om_r + v_r * (gains.k_y * e_y + gains.k_th * np.sin(e_th))
    return np.column_stack([v_cmd, om_cmd]), np.column_stack([e_x, e_y, e_th])


# Sal que separa las semillas del piloto de las del estudio confirmatorio.
# El rediseño del mecanismo fue motivado por datos del piloto, de modo que la
# evaluacion confirmatoria debe correr sobre realizaciones nuevas. Ver D15.
SALT_PILOT = "piloto"
SALT_CONFIRM = "confirmatorio-v1"
# Segunda ronda. El brazo de compensacion fue motivado por R4, que son datos
# de la primera ronda, de modo que su evaluacion exige realizaciones nuevas.
SALT_CONFIRM2 = "confirmatorio-v2"


def seed_material(scenario_id, salt=SALT_PILOT):
    """Semilla determinista y estable derivada del escenario y de la sal."""
    h = 0
    for ch in str(scenario_id) + "|" + str(salt):
        h = (h * 131 + ord(ch)) % (2 ** 31)
    return [h]


def make_book(scenario_id, n_rep, duration=120.0, dt=0.01, salt=SALT_PILOT):
    n = int(round(duration / dt))
    return NoiseBook(n, n_rep, seed_material(scenario_id, salt))


def run_batch(adapter_factory, book, degradation=None, duration=120.0,
              dt=0.01, q_scale=1.0, gains=None, params=None, ref=None,
              oracle=False, keep_traces=0, filter_class=None):
    """Corre un escenario para todas las repeticiones del NoiseBook dado."""
    gains = gains or control.KanayamaGains()
    p = params or plant.PlantParams()
    ref = ref or reference.Lemniscate()
    n_rep = book.n_rep
    n = int(round(duration / dt))
    if n != book.n_steps:
        raise ValueError("el NoiseBook no coincide con la duración pedida")

    sensors = batch.BatchSensors(dt, duration, degradation, book, n_rep)

    r0 = ref.pose(0.0)
    xi = np.zeros((n_rep, STATE_DIM))
    xi[:, IX], xi[:, IY], xi[:, ITH], xi[:, IV] = r0[0], r0[1], r0[2], r0[3]

    P0 = initial_covariance()
    x0 = xi + book.x0 * np.sqrt(np.diag(P0))
    x0[:, ITH] = wrap_pi(x0[:, ITH])
    P0b = np.broadcast_to(P0, (n_rep, STATE_DIM, STATE_DIM)).copy()

    adapter = adapter_factory(n_rep)
    cls = filter_class or batch.BatchEKF
    kf = cls(x0, P0b, p, dt, adapter, q_scale=q_scale)

    out = {
        "t": np.arange(n) * dt,
        "track_err": np.zeros((n, n_rep)),
        "head_err": np.zeros((n, n_rep)),
        "effort": np.zeros((n, n_rep)),
        "nees": np.zeros((n, n_rep)),
        "err_sq": np.zeros((n, n_rep, STATE_DIM)),
    }
    for c in CHANNELS:
        out["nis_" + c] = np.full((n, n_rep), np.nan)
        out["infl_" + c] = np.full((n, n_rep), np.nan)
        out["acc_" + c] = np.full((n, n_rep), np.nan)
    traces = {"true": [], "est": []} if keep_traces else None

    for k in range(n):
        t = k * dt
        meas = sensors.measure(xi, t, k)
        kf.update(meas, t)

        nees, err = kf.nees(xi)
        out["nees"][k] = nees
        out["err_sq"][k] = err ** 2
        for c, val in kf.nis.items():
            out["nis_" + c][k] = val
            out["infl_" + c][k] = kf.infl[c]
            out["acc_" + c][k] = kf.acc[c]

        rp = ref.pose(t)
        source = xi if oracle else kf.x
        u, _ = command_batch(source, rp, gains)
        u = batch.saturate_batch(u, p)
        dxr, dyr = rp[0] - xi[:, IX], rp[1] - xi[:, IY]
        out["track_err"][k] = np.hypot(dxr, dyr)
        out["head_err"][k] = np.abs(wrap_pi(rp[2] - xi[:, ITH]))
        out["effort"][k] = np.sum(u * u, axis=1)
        if traces is not None:
            traces["true"].append(xi[:keep_traces].copy())
            traces["est"].append(kf.x[:keep_traces].copy())

        xi = batch.step_batch(xi, u, p, dt, w=book.w[k])
        kf.predict(u)

    out["dt"] = dt
    out["n_rep"] = n_rep
    out["adapter"] = adapter.config()
    if traces is not None:
        out["trace_true"] = np.array(traces["true"])
        out["trace_est"] = np.array(traces["est"])
    return out


def switching_count(out, channel, warm=20.0):
    """Numero de cambios del estado de aceptacion de un canal por repeticion.

    Mide la conmutacion espuria. Es la cantidad que el tiempo minimo de
    permanencia esta disenado para acotar, y por tanto la que separa el
    esquema propuesto de su ablacion sin permanencia.
    """
    m = out["t"] >= warm
    a = out["acc_" + channel][m]
    n_rep = a.shape[1]
    counts = np.zeros(n_rep)
    for r in range(n_rep):
        v = a[:, r]
        v = v[np.isfinite(v)]
        if v.size > 1:
            counts[r] = np.abs(np.diff(v)).sum()
    return counts


def recovery_time(out, threshold_factor=2.0, warm=20.0, onset=20.0):
    """Tiempo hasta que el error de seguimiento vuelve a su banda nominal.

    Se define la banda como el umbral multiplicado por la mediana del error de
    seguimiento anterior al inicio de la degradacion. Si el error nunca vuelve
    a la banda, se devuelve la duracion restante del escenario, lo que evita
    valores ausentes y penaliza de forma explicita la no recuperacion.
    """
    t = out["t"]
    pre = t < onset
    post = t >= onset
    base = np.median(out["track_err"][pre & (t >= 5.0)], axis=0)
    thr = threshold_factor * base
    err = out["track_err"][post]
    tp = t[post]
    n_rep = err.shape[1]
    rec = np.full(n_rep, tp[-1] - tp[0])
    for r in range(n_rep):
        inside = err[:, r] <= thr[r]
        if inside.any():
            # ultimo instante en que el error sale de la banda
            outside = np.where(~inside)[0]
            if outside.size and outside[-1] + 1 < inside.size:
                rec[r] = tp[outside[-1] + 1] - tp[0]
            elif outside.size == 0:
                rec[r] = 0.0
    return rec


def summarize(out, warm=20.0):
    """Métricas agregadas por repetición, descartando el transitorio inicial."""
    m = out["t"] >= warm
    res = {
        "track_rmse": np.sqrt((out["track_err"][m] ** 2).mean(axis=0)),
        "track_max": out["track_err"][m].max(axis=0),
        "head_rmse": np.sqrt((out["head_err"][m] ** 2).mean(axis=0)),
        "effort": out["effort"][m].mean(axis=0),
        "nees": out["nees"][m].mean(axis=0),
        "pos_rmse": np.sqrt(out["err_sq"][m][:, :, :2].sum(axis=2).mean(axis=0)),
        "theta_rmse": np.sqrt(out["err_sq"][m][:, :, ITH].mean(axis=0)),
    }
    for c in CHANNELS:
        res["nis_" + c] = np.nanmean(out["nis_" + c][m], axis=0)
        res["acc_" + c] = np.nanmean(out["acc_" + c][m], axis=0)
        res["infl_" + c] = np.nanmean(out["infl_" + c][m], axis=0)
        res["sw_" + c] = switching_count(out, c, warm=warm)
    res["recovery"] = recovery_time(out, warm=warm)
    return res
