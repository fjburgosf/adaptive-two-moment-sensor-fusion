"""Compensacion de sesgo disparada por la prueba del primer momento.

Motivo. El estudio confirmatorio muestra que la exclusion del canal reduce el
error ante sesgo con deriva de 0.73 m a 0.32 m, pero el filtro nominal sin
degradacion esta en 0.064 m. Queda un factor de cinco. La exclusion descarta
el canal entero y con el toda su informacion. Ver docs/RESULTS.md R4.

Idea. En lugar de descartar el canal, repararlo. El estado del filtro lleva
desde el principio un termino de sesgo por canal, pero **inerte**, con ruido
de proceso en cero. Cuando la prueba sobre el primer momento dispara para un
canal, se le libera el ruido de proceso a su termino de sesgo y el filtro
pasa a estimarlo.

Por que disparado y no siempre activo. Un termino de sesgo permanentemente
activo degrada la observabilidad y empeora la estimacion cuando no hay sesgo.
Activarlo solo cuando hay evidencia obtiene lo bueno de las dos situaciones.
El precio es que una falsa alarma deja el termino activo de forma permanente,
porque la activacion es de enclavamiento. Ese precio se mide en el escenario
sin degradacion y se reporta.

Canales con termino de sesgo. Solo aquellos en los que un sesgo es a la vez
fisicamente plausible y observable.

    mag   desfase de rumbo, observable a traves de la direccion del
          movimiento que dan la posicion GNSS y la velocidad de odometria
    odo   desfase de velocidad longitudinal, observable a traves de la
          derivada de la posicion GNSS

El canal GNSS queda **excluido a proposito**. Un desfase constante de
posicion es indistinguible de la posicion verdadera cuando no hay otro sensor
de posicion absoluta, de modo que su termino de sesgo seria inobservable.
El giroscopo ya tiene su sesgo en el estado fisico de la planta.

El termino de odometria nunca llega a activarse en el protocolo actual,
porque el modo de sesgo solo se aplica al magnetometro. Se conserva porque
excluirlo seria disenar el estimador contra el escenario conocido, y porque
su inactividad demuestra que el disparo no ocurre de forma espuria.
"""
import numpy as np

from . import plant
from .batch import jacobian_batch, step_batch
from .geometry import wrap_pi
from .plant import ITH, STATE_DIM
from .sensors import H_MATRICES, IS_ANGULAR, NOMINAL_STD, nominal_R

# Canales con termino de sesgo, en orden. Define el bloque aumentado.
BIAS_CHANNELS = ("mag", "odo")

# Densidad del camino aleatorio del termino de sesgo, una vez activado, en
# multiplos de la desviacion estandar nominal del canal por raiz de segundo.
BIAS_RW_FACTOR = 0.1

# Varianza inicial del termino de sesgo, en multiplos de la varianza nominal.
BIAS_P0_FACTOR = 0.01


def augmented_dim():
    return STATE_DIM + len(BIAS_CHANNELS)


def bias_index(channel):
    """Indice del termino de sesgo dentro del estado aumentado."""
    return STATE_DIM + BIAS_CHANNELS.index(channel)


def augmented_H(channel):
    """Matriz de observacion en el espacio aumentado.

    El termino de sesgo entra de forma aditiva en su propio canal y no toca
    los demas.
    """
    n = augmented_dim()
    H = np.zeros((NOMINAL_STD[channel].size, n))
    H[:, :STATE_DIM] = H_MATRICES[channel]
    if channel in BIAS_CHANNELS:
        H[:, bias_index(channel)] = 1.0
    return H


class AugmentedBatchEKF:
    """Filtro extendido con terminos de sesgo inertes hasta ser disparados.

    Comparte el nucleo con estimhyb.batch.BatchEKF. La unica diferencia es el
    bloque aumentado y el hecho de que su ruido de proceso se libera por
    repeticion y por canal cuando el adaptador lo indica.

    La consistencia se mide siempre sobre los seis estados fisicos, usando la
    covarianza marginal. Eso mantiene el NEES comparable con el de los demas
    estimadores, que tienen exactamente esos seis estados.
    """

    def __init__(self, x0, P0, params, dt, adapter, q_scale=1.0,
                 name="aug", bias_rw=BIAS_RW_FACTOR):
        n_rep = x0.shape[0]
        n = augmented_dim()
        self.n_rep = n_rep
        self.n = n
        self.x = np.zeros((n_rep, n))
        self.x[:, :STATE_DIM] = x0
        self.P = np.zeros((n_rep, n, n))
        self.P[:, :STATE_DIM, :STATE_DIM] = P0
        for c in BIAS_CHANNELS:
            i = bias_index(c)
            self.P[:, i, i] = BIAS_P0_FACTOR * float(NOMINAL_STD[c][0] ** 2)

        self.p = params
        self.dt = float(dt)
        self.adapter = adapter
        self.name = name
        self.bias_rw = float(bias_rw)

        Q6 = plant.process_noise_cov(params, dt) * float(q_scale)
        self.Q = np.zeros((n, n))
        self.Q[:STATE_DIM, :STATE_DIM] = Q6
        # Densidad del sesgo cuando esta activo. Se aplica por repeticion.
        self.q_bias = {c: (self.bias_rw * float(NOMINAL_STD[c][0])) ** 2 * dt
                       for c in BIAS_CHANNELS}

        self.R_nom = {c: nominal_R(c) for c in H_MATRICES}
        self.H = {c: augmented_H(c) for c in H_MATRICES}
        self.eye = np.eye(n)
        self.nis = {}
        self.infl = {}
        self.acc = {}
        self.active = {c: np.zeros(n_rep, dtype=bool) for c in BIAS_CHANNELS}

    def _refresh_active(self):
        """Consulta al adaptador que terminos de sesgo deben estar activos."""
        trig = getattr(self.adapter, "bias_active", None)
        if not trig:
            return
        for c in BIAS_CHANNELS:
            if c in trig:
                self.active[c] = self.active[c] | trig[c]

    def predict(self, u):
        self._refresh_active()
        F6 = jacobian_batch(self.x[:, :STATE_DIM], self.p, self.dt, self.n_rep)
        F = np.broadcast_to(self.eye, (self.n_rep, self.n, self.n)).copy()
        F[:, :STATE_DIM, :STATE_DIM] = F6

        x6 = step_batch(self.x[:, :STATE_DIM], u, self.p, self.dt, w=None)
        x6[:, ITH] = wrap_pi(x6[:, ITH])
        self.x[:, :STATE_DIM] = x6

        Q = np.broadcast_to(self.Q, (self.n_rep, self.n, self.n)).copy()
        for c in BIAS_CHANNELS:
            i = bias_index(c)
            Q[:, i, i] = np.where(self.active[c], self.q_bias[c], 0.0)

        self.P = F @ self.P @ np.swapaxes(F, 1, 2) + Q
        self.P = 0.5 * (self.P + np.swapaxes(self.P, 1, 2))

    def update(self, measurements, t=None):
        self.nis = {}
        self.infl = {}
        self.acc = {}
        for c in sorted(measurements):
            z, available = measurements[c]
            H = self.H[c]
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
            self.nis[c] = np.where(available,
                                   np.einsum("ri,rij,rj->r", nu, Sinv, nu),
                                   np.nan)
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
        """Consistencia sobre los seis estados fisicos, covarianza marginal."""
        e = self.x[:, :STATE_DIM] - xi_true
        e[:, ITH] = wrap_pi(e[:, ITH])
        P6 = self.P[:, :STATE_DIM, :STATE_DIM]
        return np.einsum("ri,rij,rj->r", e, np.linalg.inv(P6), e), e
