"""Ejecución vectorizada sobre repeticiones.

Todas las repeticiones de un escenario avanzan en paralelo con álgebra lineal
por lotes. El estado es un arreglo de forma (repeticiones, dimension) y la
covarianza de forma (repeticiones, dimension, dimension).

La implementación escalar de estimhyb.runner se conserva como verificación
independiente. Ver docs/DECISIONS.md D11 y D15.
"""
import numpy as np

from . import plant
from .geometry import wrap_pi
from .plant import IBG, ITH, IV, IW, IX, IY, STATE_DIM
from .sensors import (CHANNELS, DECIMATION, H_MATRICES, IS_ANGULAR,
                      NOMINAL_STD, SEV_BIAS_SIGMA, SEV_DROPOUT_MEAN,
                      SEV_DROPOUT_RATE, SEV_LATENCY_STEPS, SEV_NOISE_FACTOR,
                      SEV_OUTLIER_RATE, SEV_OUTLIER_SIGMA, ChannelDegradation,
                      nominal_R)


# --------------------------------------------------------------------------
# Planta por lotes
# --------------------------------------------------------------------------

def _drift_batch(xi, u, p):
    d = np.zeros_like(xi)
    th = xi[:, ITH]
    v = xi[:, IV]
    d[:, IX] = v * np.cos(th)
    d[:, IY] = v * np.sin(th)
    d[:, ITH] = xi[:, IW]
    d[:, IV] = (u[:, 0] - v) / p.tau_v
    d[:, IW] = (u[:, 1] - xi[:, IW]) / p.tau_w
    return d


def saturate_batch(u, p):
    return np.column_stack([np.clip(u[:, 0], -p.v_max, p.v_max),
                            np.clip(u[:, 1], -p.w_max, p.w_max)])


def step_batch(xi, u, p, dt, w=None):
    """Runge Kutta de cuarto orden por lotes, más ruido aditivo opcional."""
    k1 = _drift_batch(xi, u, p)
    k2 = _drift_batch(xi + 0.5 * dt * k1, u, p)
    k3 = _drift_batch(xi + 0.5 * dt * k2, u, p)
    k4 = _drift_batch(xi + dt * k3, u, p)
    nxt = xi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if w is not None:
        scale = np.zeros(STATE_DIM)
        scale[IV] = p.sigma_v * np.sqrt(dt)
        scale[IW] = p.sigma_w * np.sqrt(dt)
        scale[IBG] = p.sigma_bg * np.sqrt(dt)
        nxt = nxt + w * scale
    return nxt


def jacobian_batch(xi, p, dt, n_rep):
    F = np.broadcast_to(np.eye(STATE_DIM), (n_rep, STATE_DIM, STATE_DIM)).copy()
    th = xi[:, ITH]
    v = xi[:, IV]
    F[:, IX, ITH] = -dt * v * np.sin(th)
    F[:, IX, IV] = dt * np.cos(th)
    F[:, IY, ITH] = dt * v * np.cos(th)
    F[:, IY, IV] = dt * np.sin(th)
    F[:, ITH, IW] = dt
    F[:, IV, IV] = 1.0 - dt / p.tau_v
    F[:, IW, IW] = 1.0 - dt / p.tau_w
    return F


# --------------------------------------------------------------------------
# Sensores por lotes
# --------------------------------------------------------------------------

class BatchSensors:
    """Genera mediciones degradadas para todas las repeticiones a la vez."""

    def __init__(self, dt, duration, degradation, book, n_rep):
        self.dt = float(dt)
        self.duration = float(duration)
        self.book = book
        self.n_rep = int(n_rep)
        self.deg = {c: ChannelDegradation() for c in CHANNELS}
        if degradation:
            self.deg.update(degradation)
        self.drop_left = {c: np.zeros(n_rep, dtype=int) for c in CHANNELS}
        self.hist = []

    def _sev(self, c, t, primary, secondary, factor):
        d = self.deg[c]
        if not d.active(t):
            return 0.0
        if d.mode == primary:
            return d.severity
        if d.mode == secondary:
            return d.severity * factor
        return 0.0

    def measure(self, xi_true, t, k):
        self.hist.append(xi_true.copy())
        if len(self.hist) > SEV_LATENCY_STEPS + 1:
            self.hist.pop(0)
        out = {}
        for c in CHANNELS:
            if k % DECIMATION[c] != 0:
                continue
            j = self.book.sample_index(c, k)
            d = self.deg[c]

            drop = np.zeros(self.n_rep, dtype=bool)
            s_drop = self._sev(c, t, "dropout", "combined", 0.5)
            if s_drop > 0.0:
                ongoing = self.drop_left[c] > 0
                self.drop_left[c][ongoing] -= 1
                fresh = (~ongoing) & (self.book.drop_u[c][j]
                                      < SEV_DROPOUT_RATE * s_drop)
                mean = max(1.0, SEV_DROPOUT_MEAN * s_drop)
                self.drop_left[c][fresh] = (
                    self.book.drop_len[c][j][fresh] * mean).astype(int)
                drop = ongoing | fresh

            lag = 0
            if d.active(t) and d.mode == "latency":
                lag = int(round(SEV_LATENCY_STEPS * d.severity))
            src = self.hist[-1 - min(lag, len(self.hist) - 1)]

            z = src @ H_MATRICES[c].T

            s_noise = self._sev(c, t, "noise", "combined", 0.6)
            scale = 1.0 + (SEV_NOISE_FACTOR - 1.0) * s_noise
            z = z + self.book.v[c][j] * (NOMINAL_STD[c] * scale)

            if d.active(t) and d.mode == "bias":
                span = max(self.duration - d.onset, self.dt)
                frac = min((t - d.onset) / span, 1.0)
                z = z + SEV_BIAS_SIGMA * d.severity * frac * NOMINAL_STD[c]

            s_out = self._sev(c, t, "outlier", "combined", 0.5)
            if s_out > 0.0:
                hit = self.book.out_u[c][j] < SEV_OUTLIER_RATE * s_out
                z = z + (hit[:, None] * self.book.out_sign[c][j]
                         * SEV_OUTLIER_SIGMA * NOMINAL_STD[c])

            out[c] = (z, ~drop)
        return out


# --------------------------------------------------------------------------
# Filtro por lotes
# --------------------------------------------------------------------------

class BatchEKF:
    """Filtro extendido vectorizado sobre repeticiones."""

    def __init__(self, x0, P0, params, dt, adapter, q_scale=1.0, name="ekf"):
        self.x = np.array(x0, dtype=float)
        self.n_rep = self.x.shape[0]
        self.P = np.array(P0, dtype=float)
        self.p = params
        self.dt = float(dt)
        self.adapter = adapter
        self.name = name
        self.Q = plant.process_noise_cov(params, dt) * float(q_scale)
        self.R_nom = {c: nominal_R(c) for c in H_MATRICES}
        self.eye = np.eye(STATE_DIM)
        self.nis = {}
        self.infl = {}
        self.acc = {}

    def predict(self, u):
        F = jacobian_batch(self.x, self.p, self.dt, self.n_rep)
        self.x = step_batch(self.x, u, self.p, self.dt, w=None)
        self.x[:, ITH] = wrap_pi(self.x[:, ITH])
        self.P = F @ self.P @ np.swapaxes(F, 1, 2) + self.Q
        self.P = 0.5 * (self.P + np.swapaxes(self.P, 1, 2))

    def update(self, measurements, t=None):
        self.nis = {}
        self.infl = {}
        self.acc = {}
        for c in sorted(measurements):
            z, available = measurements[c]
            H = H_MATRICES[c]
            nu = z - self.x @ H.T
            if IS_ANGULAR[c]:
                nu = wrap_pi(nu)
            R_nom = self.R_nom[c]
            HP = H @ self.P
            S_nom = HP @ H.T + R_nom
            nis_nom = np.einsum("ri,rij,rj->r", nu, np.linalg.inv(S_nom), nu)

            R_eff, infl = self.adapter.covariance(c, R_nom, nu, S_nom,
                                                  nis_nom, t)
            accept = self.adapter.accept(c, nu, S_nom, nis_nom, t) & available

            S = HP @ H.T + R_eff
            Sinv = np.linalg.inv(S)
            nis = np.einsum("ri,rij,rj->r", nu, Sinv, nu)
            self.nis[c] = np.where(available, nis, np.nan)
            self.infl[c] = np.where(available, infl, np.nan)
            self.acc[c] = accept.astype(float)

            K = np.swapaxes(HP, 1, 2) @ Sinv
            x_new = self.x + np.einsum("rij,rj->ri", K, nu)
            x_new[:, ITH] = wrap_pi(x_new[:, ITH])
            IKH = self.eye - K @ H
            P_new = (IKH @ self.P @ np.swapaxes(IKH, 1, 2)
                     + K @ R_eff @ np.swapaxes(K, 1, 2))
            m = accept[:, None]
            self.x = np.where(m, x_new, self.x)
            self.P = np.where(m[:, :, None], P_new, self.P)
            self.P = 0.5 * (self.P + np.swapaxes(self.P, 1, 2))

    def nees(self, xi_true):
        e = self.x - xi_true
        e[:, ITH] = wrap_pi(e[:, ITH])
        return np.einsum("ri,rij,rj->r", e, np.linalg.inv(self.P), e), e
