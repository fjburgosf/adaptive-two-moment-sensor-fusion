"""Sensores sintéticos y protocolo de degradación parametrizado por severidad.

Cuatro canales, todos con modelo de medición lineal en el estado. La no
linealidad del problema reside en la propagación, no en la observación, lo
que mantiene el paso de actualización exacto y aísla el efecto de la
adaptación de la covarianza.

    gnss   z = [x, y]            5 Hz
    odo    z = v                50 Hz
    gyro   z = omega + b_g     100 Hz
    mag    z = theta            10 Hz

La degradación se aplica en la generación de la medición. Los filtros nunca
son informados de ella y conservan siempre la covarianza nominal, lo que
produce la mala especificación que el estudio necesita.

Severidad. Todo modo de degradación se parametriza por un escalar s en el
intervalo cerrado de cero a uno. En s igual a cero el canal es nominal. La
progresión es continua, de modo que el resultado es una curva de degradación
y no una colección de casos sueltos.
"""
from collections import deque

import numpy as np

from .plant import STATE_DIM, IX, IY, ITH, IV, IW, IBG

CHANNELS = ("gnss", "odo", "gyro", "mag")

# Ganancias máximas del protocolo, alcanzadas en severidad uno.
SEV_NOISE_FACTOR = 10.0     # multiplicador de la desviación estándar
SEV_BIAS_SIGMA = 8.0        # sesgo terminal en múltiplos de la sigma nominal
SEV_OUTLIER_RATE = 0.30     # probabilidad de valor atípico por muestra
SEV_OUTLIER_SIGMA = 25.0    # magnitud del atípico en múltiplos de sigma
SEV_DROPOUT_RATE = 0.05     # probabilidad de entrar en pérdida por muestra
SEV_DROPOUT_MEAN = 40       # duración media de la pérdida, en muestras
SEV_LATENCY_STEPS = 20      # retardo terminal, en pasos de la base temporal

DEGRADATION_MODES = ("none", "noise", "bias", "outlier", "dropout",
                     "latency", "combined")


def _build_H():
    """Matrices de observación de cada canal."""
    H = {}
    h = np.zeros((2, STATE_DIM)); h[0, IX] = 1.0; h[1, IY] = 1.0
    H["gnss"] = h
    h = np.zeros((1, STATE_DIM)); h[0, IV] = 1.0
    H["odo"] = h
    h = np.zeros((1, STATE_DIM)); h[0, IW] = 1.0; h[0, IBG] = 1.0
    H["gyro"] = h
    h = np.zeros((1, STATE_DIM)); h[0, ITH] = 1.0
    H["mag"] = h
    return H


H_MATRICES = _build_H()

# Desviaciones estándar nominales. Son también las que reciben los filtros.
NOMINAL_STD = {
    "gnss": np.array([0.60, 0.60]),   # m
    "odo": np.array([0.05]),          # m/s
    "gyro": np.array([0.012]),        # rad/s
    "mag": np.array([0.035]),         # rad
}

# Periodo de muestreo de cada canal, en pasos de la base temporal de 100 Hz.
DECIMATION = {"gnss": 20, "odo": 2, "gyro": 1, "mag": 10}

IS_ANGULAR = {"gnss": False, "odo": False, "gyro": False, "mag": True}


def nominal_R(channel):
    """Covarianza nominal del canal. Es la que se entrega a los filtros."""
    return np.diag(NOMINAL_STD[channel] ** 2)


class ChannelDegradation:
    """Configuración de degradación de un canal.

    mode      uno de DEGRADATION_MODES
    severity  escalar en el intervalo cerrado de cero a uno
    onset     instante en segundos en que la degradación empieza a actuar
    """

    def __init__(self, mode="none", severity=0.0, onset=0.0):
        if mode not in DEGRADATION_MODES:
            raise ValueError("modo de degradación desconocido, %s" % mode)
        if not 0.0 <= severity <= 1.0:
            raise ValueError("la severidad debe estar entre cero y uno")
        self.mode = mode
        self.severity = float(severity)
        self.onset = float(onset)

    def active(self, t):
        return self.mode != "none" and self.severity > 0.0 and t >= self.onset

    def as_dict(self):
        return dict(mode=self.mode, severity=self.severity, onset=self.onset)


class SensorSuite:
    """Generador de mediciones degradadas a partir del estado verdadero."""

    def __init__(self, dt, degradation=None, rng=None, duration=None):
        self.dt = float(dt)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.deg = {c: ChannelDegradation() for c in CHANNELS}
        if degradation:
            for c, d in degradation.items():
                if c not in CHANNELS:
                    raise ValueError("canal desconocido, %s" % c)
                self.deg[c] = d
        self.duration = duration
        self._k = 0
        self._dropout_left = {c: 0 for c in CHANNELS}
        self._history = deque(maxlen=max(1, SEV_LATENCY_STEPS + 1))

    # -- factores dependientes de la severidad -------------------------------

    def _noise_scale(self, c, t):
        d = self.deg[c]
        if not d.active(t) or d.mode not in ("noise", "combined"):
            return 1.0
        s = d.severity if d.mode == "noise" else 0.6 * d.severity
        return 1.0 + (SEV_NOISE_FACTOR - 1.0) * s

    def _bias(self, c, t):
        """Sesgo con deriva. Rampa lineal desde el instante de inicio."""
        d = self.deg[c]
        if not d.active(t) or d.mode != "bias":
            return np.zeros_like(NOMINAL_STD[c])
        span = max(self.duration - d.onset, self.dt) if self.duration else 1.0
        frac = min((t - d.onset) / span, 1.0)
        return SEV_BIAS_SIGMA * d.severity * frac * NOMINAL_STD[c]

    def _outlier(self, c, t):
        d = self.deg[c]
        if not d.active(t) or d.mode not in ("outlier", "combined"):
            return np.zeros_like(NOMINAL_STD[c])
        s = d.severity if d.mode == "outlier" else 0.5 * d.severity
        if self.rng.random() >= SEV_OUTLIER_RATE * s:
            return np.zeros_like(NOMINAL_STD[c])
        sign = self.rng.choice([-1.0, 1.0], size=NOMINAL_STD[c].shape)
        return sign * SEV_OUTLIER_SIGMA * NOMINAL_STD[c]

    def _dropped(self, c, t):
        d = self.deg[c]
        if not d.active(t) or d.mode not in ("dropout", "combined"):
            return False
        if self._dropout_left[c] > 0:
            self._dropout_left[c] -= 1
            return True
        s = d.severity if d.mode == "dropout" else 0.5 * d.severity
        if self.rng.random() < SEV_DROPOUT_RATE * s:
            mean = max(1.0, SEV_DROPOUT_MEAN * s)
            self._dropout_left[c] = int(self.rng.exponential(mean))
            return True
        return False

    def _latency_steps(self, c, t):
        d = self.deg[c]
        if not d.active(t) or d.mode != "latency":
            return 0
        return int(round(SEV_LATENCY_STEPS * d.severity))

    # -- generación ----------------------------------------------------------

    def measure(self, xi_true, t):
        """Devuelve las mediciones disponibles en el instante t.

        La salida es un diccionario de canal a vector de medición. Un canal
        ausente significa que no correspondía muestrear o que la medición se
        perdió. La ausencia por pérdida es indistinguible de la ausencia por
        muestreo desde el punto de vista del filtro, que es exactamente la
        situación que se quiere estudiar.
        """
        self._history.append(np.array(xi_true, dtype=float))
        out = {}
        for c in CHANNELS:
            if self._k % DECIMATION[c] != 0:
                continue
            if self._dropped(c, t):
                continue
            lag = self._latency_steps(c, t)
            src = self._history[-1 - min(lag, len(self._history) - 1)]
            z = H_MATRICES[c] @ src
            std = NOMINAL_STD[c] * self._noise_scale(c, t)
            z = z + std * self.rng.standard_normal(z.shape)
            z = z + self._bias(c, t) + self._outlier(c, t)
            out[c] = z
        self._k += 1
        return out

    def config(self):
        return {c: self.deg[c].as_dict() for c in CHANNELS}
