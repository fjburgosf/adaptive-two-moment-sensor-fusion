"""Control negativo. Planta linealizada, filtro bien especificado.

Motivo. La hipotesis H1 afirma que el ordenamiento de estimadores por error de
estimacion en lazo abierto no se conserva al ordenarlos por desempeno en lazo
cerrado. Esa afirmacion **solo tiene contenido** bajo planta no lineal y
filtro mal especificado. En el caso lineal gaussiano con modelo correcto rige
el principio de separacion, el costo en lazo cerrado es monotono en la
covarianza del error de estimacion, y el ordenamiento debe conservarse.

Este modulo construye ese caso. Si en el la correlacion de ordenamientos no
resulta cercana a uno, hay un error de implementacion y no un hallazgo.
Ver docs/DECISIONS.md D9.

Linealizacion del uniciclo alrededor de una recta recorrida a velocidad
constante v0, con rumbo nulo. La desviacion respecto de esa referencia sigue

    d(dx)/dt      = dv
    d(dy)/dt      = v0 dtheta
    d(dtheta)/dt  = domega
    d(dv)/dt      = (du_v - dv) / tau_v
    d(domega)/dt  = (du_w - domega) / tau_w
    d(db_g)/dt    = ruido blanco

que es invariante en el tiempo. Las matrices de observacion de los cuatro
canales ya son lineales en el estado, de modo que el sistema completo es
lineal y gaussiano salvo por la degradacion, que es justo lo que se manipula.
"""
import numpy as np
from scipy.linalg import expm, solve_discrete_are

from .plant import IBG, ITH, IV, IW, IX, IY, STATE_DIM

# Subconjunto controlable. El sesgo del giroscopo no recibe accion de control
# y es marginalmente estable, de modo que se excluye del diseno del regulador
# y su ganancia se rellena con cero.
CTRL_IDX = (IX, IY, ITH, IV, IW)

# Pesos del regulador. Se penaliza mas la desviacion lateral y el rumbo, que
# es lo que define la calidad del seguimiento.
Q_LQR = np.diag([1.0, 10.0, 5.0, 1.0, 1.0])
R_LQR = np.diag([1.0, 1.0])


def continuous_matrices(p, v0):
    """Matrices A y B del sistema linealizado de dimension seis."""
    A = np.zeros((STATE_DIM, STATE_DIM))
    A[IX, IV] = 1.0
    A[IY, ITH] = v0
    A[ITH, IW] = 1.0
    A[IV, IV] = -1.0 / p.tau_v
    A[IW, IW] = -1.0 / p.tau_w
    B = np.zeros((STATE_DIM, 2))
    B[IV, 0] = 1.0 / p.tau_v
    B[IW, 1] = 1.0 / p.tau_w
    return A, B


def discretize(A, B, dt):
    """Discretizacion exacta por exponencial de matriz."""
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def design_lqr(p, v0, dt):
    """Ganancia de realimentacion de estado, constante y compartida.

    Se disena sobre el subsistema controlable y se rellena con cero la columna
    del sesgo del giroscopo. La misma ganancia se usa con todos los
    estimadores, que es la condicion para atribuirles cualquier diferencia.
    """
    A, B = continuous_matrices(p, v0)
    Ad, Bd = discretize(A, B, dt)
    idx = np.array(CTRL_IDX)
    Ac = Ad[np.ix_(idx, idx)]
    Bc = Bd[idx, :]
    P = solve_discrete_are(Ac, Bc, Q_LQR, R_LQR)
    Kc = np.linalg.solve(Bc.T @ P @ Bc + R_LQR, Bc.T @ P @ Ac)
    K = np.zeros((2, STATE_DIM))
    K[:, idx] = Kc
    return K, Ad, Bd


def lq_cost(delta, u, dt):
    """Costo cuadratico en lazo cerrado, promediado en el tiempo.

    delta tiene forma (pasos, repeticiones, dimension) y u forma
    (pasos, repeticiones, dos).
    """
    d = delta[:, :, np.array(CTRL_IDX)]
    jx = np.einsum("kri,ij,krj->kr", d, Q_LQR, d)
    ju = np.einsum("kri,ij,krj->kr", u, R_LQR, u)
    return (jx + ju).mean(axis=0)


class LinearKF:
    """Filtro de Kalman del sistema lineal, vectorizado sobre repeticiones.

    La covarianza de medicion de cada canal se multiplica por un factor gamma
    declarado. Con gamma igual a uno el filtro esta correctamente especificado
    y es el estimador optimo. Con gamma distinto de uno sigue siendo lineal e
    invariante en el tiempo, de modo que el principio de separacion se
    mantiene y el ordenamiento debe conservarse. Esa familia es el instrumento
    con el que se mide el ordenamiento.
    """

    def __init__(self, x0, P0, Ad, Q, R_nom, gamma=1.0):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.Ad = Ad
        self.Q = Q
        self.gamma = float(gamma)
        self.R = {c: R_nom[c] * self.gamma for c in R_nom}

    def predict(self, u, Bd):
        self.x = self.x @ self.Ad.T + u @ Bd.T
        self.P = self.Ad @ self.P @ self.Ad.T + self.Q
        self.P = 0.5 * (self.P + np.swapaxes(self.P, 1, 2))

    def update(self, measurements, H_matrices):
        for c in sorted(measurements):
            z, available = measurements[c]
            H = H_matrices[c]
            nu = z - self.x @ H.T
            HP = H @ self.P
            S = HP @ H.T + self.R[c]
            K = np.swapaxes(HP, 1, 2) @ np.linalg.inv(S)
            x_new = self.x + np.einsum("rij,rj->ri", K, nu)
            eye = np.eye(self.x.shape[1])
            IKH = eye - K @ H
            P_new = (IKH @ self.P @ np.swapaxes(IKH, 1, 2)
                     + K @ self.R[c] @ np.swapaxes(K, 1, 2))
            m = available[:, None]
            self.x = np.where(m, x_new, self.x)
            self.P = np.where(m[:, :, None], P_new, self.P)
            self.P = 0.5 * (self.P + np.swapaxes(self.P, 1, 2))
