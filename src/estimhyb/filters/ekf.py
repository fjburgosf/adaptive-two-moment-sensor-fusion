"""Nucleo de filtrado extendido escalar. Implementacion de verificacion.

AVISO SOBRE SU PAPEL. Esta implementacion NO es la de produccion. La de
produccion es estimhyb.batch.BatchEKF, vectorizada sobre repeticiones. Esta
se conserva como camino de codigo independiente para reproducir resultados
segun la exigencia de D11, que aparecio al quedar el proyecto sin auditoria
cruzada entre agentes.

Difiere de la vectorizada en tres aspectos que la hacen util como control.
Recorre una repeticion a la vez. Resuelve los canales de un grado de libertad
con aritmetica escalar en lugar de inversion matricial. Y construye el paso
de actualizacion con un orden de operaciones distinto.

Limitacion que debe declararse. La independencia es de camino de codigo
dentro del mismo autor y del mismo modelo. No equivale a una reimplementacion
desde la especificacion por una segunda parte.

Nucleo comun a todos los estimadores comparados.

Decisión de arquitectura. Los seis estimadores del estudio comparten este
mismo núcleo y difieren únicamente en el objeto adaptador que decide, canal
por canal, qué covarianza de medición usar y si la medición se acepta. Esa
separación garantiza que la comparación aísle el mecanismo de adaptación y no
mezcle diferencias de implementación del filtro.

El paso de actualización es secuencial por canal. Cada canal se procesa con su
propia matriz de observación, lo que permite adaptar y excluir canales de
forma independiente y refleja además que los sensores llegan a tasas
distintas.

La actualización de covarianza usa la forma de Joseph, que preserva la
simetría y la positividad incluso cuando el adaptador entrega una covarianza
de medición muy inflada.
"""
import numpy as np

from ..geometry import wrap_pi
from ..plant import ITH, STATE_DIM, jacobian_f, process_noise_cov, step
from ..sensors import H_MATRICES, IS_ANGULAR, nominal_R


class FilterDiagnostics:
    """Registro por canal de lo ocurrido en el último paso de actualización."""

    def __init__(self):
        self.nis = {}          # innovación normalizada al cuadrado
        self.dof = {}          # grados de libertad del canal
        self.accepted = {}     # si la medición entró en la actualización
        self.inflation = {}    # factor aplicado a la covarianza nominal


class ExtendedKalmanFilter:
    """Filtro extendido con adaptación de covarianza delegada.

    adapter    objeto con la interfaz de estimhyb.filters.adapters.NoiseAdapter
    q_scale    escala aplicada a Q. Permite construir la variante mal
               sintonizada sin tocar el resto del código
    """

    def __init__(self, x0, P0, params, dt, adapter, q_scale=1.0, name="ekf"):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.p = params
        self.dt = float(dt)
        self.adapter = adapter
        self.q_scale = float(q_scale)
        self.name = name
        self.Q = process_noise_cov(params, dt) * self.q_scale
        self.R_nom = {c: nominal_R(c) for c in H_MATRICES}
        self.diag = FilterDiagnostics()
        self.n = STATE_DIM
        self._eye = np.eye(STATE_DIM)

    # -- predicción ----------------------------------------------------------

    def predict(self, u):
        F = jacobian_f(self.x, u, self.p, self.dt)
        self.x = step(self.x, u, self.p, self.dt, rng=None)
        self.x[ITH] = wrap_pi(self.x[ITH])
        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)

    # -- actualización -------------------------------------------------------

    def _innovation(self, channel, z):
        H = H_MATRICES[channel]
        nu = np.asarray(z, dtype=float) - H @ self.x
        if IS_ANGULAR[channel]:
            nu = wrap_pi(nu)
        return H, nu

    def update(self, measurements, t=None):
        """Procesa de forma secuencial los canales disponibles.

        Los canales de un solo grado de libertad se resuelven con aritmética
        escalar. Es algebraicamente idéntico al caso general y evita tres
        factorizaciones por actualización, lo que importa porque el giróscopo
        se muestrea en cada paso de la base temporal.
        """
        self.diag = FilterDiagnostics()
        for channel in sorted(measurements):
            H, nu = self._innovation(channel, measurements[channel])
            R_nom = self.R_nom[channel]
            HP = H @ self.P
            S_nom = HP @ H.T + R_nom
            dof = nu.size
            scalar = dof == 1
            if scalar:
                nis_nom = float(nu[0] * nu[0] / S_nom[0, 0])
            else:
                nis_nom = float(nu @ np.linalg.solve(S_nom, nu))

            # El adaptador tiene interfaz por lotes. Se le presenta esta
            # repeticion unica como un lote de tamano uno.
            nu_b = nu.reshape(1, dof)
            S_b = S_nom.reshape(1, dof, dof)
            nis_b = np.array([nis_nom])
            R_batch, infl_b = self.adapter.covariance(
                channel, R_nom, nu_b, S_b, nis_b, t)
            R_eff = np.asarray(R_batch)[0]
            inflation = float(np.asarray(infl_b)[0])
            accept = bool(np.asarray(
                self.adapter.accept(channel, nu_b, S_b, nis_b, t))[0])

            self.diag.dof[channel] = dof
            self.diag.inflation[channel] = inflation
            self.diag.accepted[channel] = accept

            S = HP @ H.T + R_eff
            if scalar:
                s_val = S[0, 0]
                self.diag.nis[channel] = float(nu[0] * nu[0] / s_val)
                if not accept:
                    continue
                K = (HP[0] / s_val).reshape(self.n, 1)
            else:
                self.diag.nis[channel] = float(nu @ np.linalg.solve(S, nu))
                if not accept:
                    continue
                K = np.linalg.solve(S.T, HP).T

            self.x = self.x + K @ nu
            self.x[ITH] = wrap_pi(self.x[ITH])
            IKH = self._eye - K @ H
            self.P = IKH @ self.P @ IKH.T + K @ R_eff @ K.T
            self.P = 0.5 * (self.P + self.P.T)

    # -- consistencia --------------------------------------------------------

    def nees(self, xi_true):
        """Error de estimación normalizado al cuadrado contra el estado real.

        El componente angular se envuelve antes de normalizar. Sin esa
        precaución un salto de dos pi produce un valor espurio enorme.
        """
        e = self.x - np.asarray(xi_true, dtype=float)
        e[ITH] = wrap_pi(e[ITH])
        return float(e @ np.linalg.solve(self.P, e)), e
