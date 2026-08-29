"""Adaptadores de covarianza de medición, vectorizados sobre repeticiones.

Interfaz común. El núcleo del filtro es idéntico para todos los estimadores
comparados y solo cambia el adaptador, de modo que cualquier diferencia
observada es atribuible al mecanismo de adaptación.

    covariance(canal, R_nom, nu, S_nom, nis_nom, t)
        devuelve R_eff de forma (rep, dof, dof) y el factor de inflación de
        forma (rep,)
    accept(canal, nu, S_nom, nis_nom, t)
        devuelve un arreglo booleano de forma (rep,)

Convención de formas.
    R_nom    (dof, dof)     covarianza nominal del canal, común a todos
    nu       (rep, dof)     innovación
    S_nom    (rep, dof, dof) covarianza de innovación con R nominal
    nis_nom  (rep,)         innovación normalizada al cuadrado con R nominal

Nota sobre la detección. El estadístico que gobierna la adaptación se calcula
con la covarianza efectiva vigente, no con la nominal. El objetivo declarado
del esquema propuesto es que el filtro sea consistente, y la consistencia se
mide sobre la covarianza que el filtro realmente usa. Calcularlo sobre la
nominal produciría un detector que ignora su propia acción.
"""
import numpy as np
from scipy.stats import chi2


def _dof_of(R_nom):
    return R_nom.shape[0]


def _broadcast(R_nom, n_rep):
    return np.broadcast_to(R_nom, (n_rep,) + R_nom.shape).copy()


def _nis_with(nu, S_nom, R_nom, lam):
    """NIS usando la covarianza efectiva con el factor de inflación dado.

    Se aprovecha que H P H^T se obtiene restando R_nom de S_nom, lo que evita
    propagar P hasta el adaptador.
    """
    HPHt = S_nom - R_nom
    S = HPHt + lam[:, None, None] * R_nom
    return np.einsum("ri,rij,rj->r", nu, np.linalg.inv(S), nu), S


class NoiseAdapter:
    """Covarianza fija. Es el comportamiento del filtro extendido clásico."""

    name = "fixed"

    def __init__(self, n_rep=1):
        self.n_rep = int(n_rep)

    def covariance(self, channel, R_nom, nu, S_nom, nis_nom, t):
        n = nu.shape[0]
        return _broadcast(R_nom, n), np.ones(n)

    def accept(self, channel, nu, S_nom, nis_nom, t):
        return np.ones(nu.shape[0], dtype=bool)

    def reset(self):
        pass

    def config(self):
        return dict(name=self.name)


class ScaledRAdapter(NoiseAdapter):
    """Covarianza fija multiplicada por un factor declarado.

    Genera una familia de estimadores de calidad graduada sin cambiar de
    algoritmo. Es el instrumento con el que se mide si el ordenamiento por
    error de estimacion se conserva al ordenar por desempeno en lazo cerrado.
    Con gamma igual a uno coincide con el filtro nominal.
    """

    name = "scaledR"

    def __init__(self, n_rep=1, gamma=1.0):
        super().__init__(n_rep)
        self.gamma = float(gamma)

    def covariance(self, channel, R_nom, nu, S_nom, nis_nom, t):
        n = nu.shape[0]
        lam = np.full(n, self.gamma)
        return _broadcast(R_nom, n) * self.gamma, lam

    def config(self):
        return dict(name=self.name, gamma=self.gamma)


class CovarianceMatchingAdapter(NoiseAdapter):
    """Correspondencia de covarianza con ventana deslizante.

    Estima la covarianza de medición a partir del promedio muestral de la
    innovación externa menos la parte explicada por el estado,

        R_est = media( nu nu^T ) - H P H^T

    Es la formulación clásica basada en innovación. Reacciona a la
    fluctuación estadística normal y no dispone de mecanismo alguno para
    evitar la conmutación espuria, lo que la convierte en el punto de
    comparación natural del esquema propuesto.
    """

    name = "covmatch"

    def __init__(self, n_rep=1, window=30, lam_max=1.0e3):
        super().__init__(n_rep)
        self.window = int(window)
        self.lam_max = float(lam_max)
        self.buf = {}

    def reset(self):
        self.buf = {}

    def covariance(self, channel, R_nom, nu, S_nom, nis_nom, t):
        n, dof = nu.shape
        outer = np.einsum("ri,rj->rij", nu, nu)
        hist = self.buf.setdefault(channel, [])
        hist.append(outer)
        if len(hist) > self.window:
            hist.pop(0)
        mean_outer = np.mean(np.stack(hist, axis=0), axis=0)
        HPHt = S_nom - R_nom
        R_est = mean_outer - HPHt
        # La resta puede producir una matriz no definida positiva. Se proyecta
        # sobre el cono de matrices válidas manteniendo la nominal como piso.
        tr_est = np.trace(R_est, axis1=1, axis2=2)
        tr_nom = float(np.trace(R_nom))
        lam = np.clip(tr_est / tr_nom, 1.0, self.lam_max)
        return _broadcast(R_nom, n) * lam[:, None, None], lam

    def config(self):
        return dict(name=self.name, window=self.window, lam_max=self.lam_max)


class SageHusaAdapter(NoiseAdapter):
    """Estimador adaptativo de Sage Husa con factor de olvido.

        R_k = (1 - d_k) R_{k-1} + d_k ( nu nu^T - H P H^T )
        d_k = (1 - b) / (1 - b^(k+1))

    Es una de las referencias más usadas en la literatura de filtrado
    adaptativo. Su debilidad conocida es la pérdida de definición positiva
    cuando la innovación es pequeña, que aquí se contiene con un piso sobre
    la traza.
    """

    name = "sagehusa"

    def __init__(self, n_rep=1, b=0.98, lam_min=0.2, lam_max=1.0e3):
        super().__init__(n_rep)
        self.b = float(b)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.lam = {}
        self.k = {}

    def reset(self):
        self.lam = {}
        self.k = {}

    def covariance(self, channel, R_nom, nu, S_nom, nis_nom, t):
        n = nu.shape[0]
        lam = self.lam.setdefault(channel, np.ones(n))
        k = self.k.get(channel, 0)
        d = (1.0 - self.b) / (1.0 - self.b ** (k + 1))
        HPHt = S_nom - R_nom
        outer = np.einsum("ri,rj->rij", nu, nu)
        tr_new = np.trace(outer - HPHt, axis1=1, axis2=2) / float(np.trace(R_nom))
        lam = (1.0 - d) * lam + d * tr_new
        lam = np.clip(lam, self.lam_min, self.lam_max)
        self.lam[channel] = lam
        self.k[channel] = k + 1
        return _broadcast(R_nom, n) * lam[:, None, None], lam

    def config(self):
        return dict(name=self.name, b=self.b, lam_min=self.lam_min,
                    lam_max=self.lam_max)


class HuberAdapter(NoiseAdapter):
    """Filtro robusto de Huber en su forma de peso equivalente.

    El residuo estandarizado se compara contra un umbral. Dentro del umbral el
    estimador se comporta como mínimos cuadrados. Fuera de él, la covarianza
    se infla en proporción al exceso, lo que equivale a la norma uno.
    """

    name = "huber"

    def __init__(self, n_rep=1, kappa=1.345, lam_max=1.0e3):
        super().__init__(n_rep)
        self.kappa = float(kappa)
        self.lam_max = float(lam_max)

    def covariance(self, channel, R_nom, nu, S_nom, nis_nom, t):
        n = nu.shape[0]
        dof = _dof_of(R_nom)
        r = np.sqrt(np.maximum(nis_nom, 0.0) / dof)
        lam = np.where(r <= self.kappa, 1.0,
                       np.minimum((r / self.kappa) ** 2, self.lam_max))
        return _broadcast(R_nom, n) * lam[:, None, None], lam

    def config(self):
        return dict(name=self.name, kappa=self.kappa, lam_max=self.lam_max)


class DwellTimeAdapter(NoiseAdapter):
    """Esquema propuesto. Adaptación por canal con conmutación restringida.

    Tres elementos, ninguno nuevo por separado, que no aparecen combinados en
    la literatura revisada. Ver docs/LITERATURE.md sección G.

    1. Inflación acotada guiada por consistencia. El factor se actualiza con
       la razón entre el estadístico observado y su valor esperado, elevada a
       un exponente de suavizado, y se satura en un intervalo cerrado. La cota
       superior garantiza que la ganancia permanezca acotada y que el paso de
       actualización siga bien planteado.

    2. Zona muerta. El factor no se mueve mientras la razón permanezca dentro
       de una banda de tolerancia alrededor de uno. Sin ella el esquema
       reacciona a la fluctuación estadística normal.

    3. Exclusión con tiempo de permanencia y readmisión. Si el estadístico
       supera el umbral chi cuadrado durante un número mínimo de muestras
       consecutivas, el canal deja de actualizar. Vuelve a entrar solo tras un
       periodo de espera. El tiempo mínimo de permanencia es lo que impide la
       conmutación de alta frecuencia entre inflar y excluir.

    Configurando dead_zone en cero y desactivando la compuerta se obtiene la
    ablación que aísla la contribución de los elementos dos y tres.
    """

    name = "dwell"

    def __init__(self, n_rep=1, rho=0.5, dead_zone=0.5, lam_max=1.0e3,
                 alpha=0.01, dwell_up=3, dwell_hold=25, gating=True,
                 beta=0.02, alpha_bias=1.0e-3, mean_test=True,
                 compensate=False):
        super().__init__(n_rep)
        self.rho = float(rho)
        self.dead_zone = float(dead_zone)
        self.lam_max = float(lam_max)
        self.alpha = float(alpha)
        self.dwell_up = int(dwell_up)
        self.dwell_hold = int(dwell_hold)
        self.gating = bool(gating)
        self.beta = float(beta)
        self.alpha_bias = float(alpha_bias)
        self.mean_test = bool(mean_test)
        self.compensate = bool(compensate)
        self.bias_active = {}
        self.bias_up = {}
        self.lam = {}
        self.up = {}
        self.hold = {}
        self.excluded = {}
        self.ewma = {}
        self.tbias = {}
        self._thr = {}
        self._thr_bias = {}

    def reset(self):
        self.lam = {}
        self.up = {}
        self.hold = {}
        self.excluded = {}
        self.ewma = {}
        self.tbias = {}
        self.bias_active = {}
        self.bias_up = {}

    def _state(self, channel, n, dof):
        if channel not in self.lam:
            self.lam[channel] = np.ones(n)
            self.up[channel] = np.zeros(n, dtype=int)
            self.hold[channel] = np.zeros(n, dtype=int)
            self.excluded[channel] = np.zeros(n, dtype=bool)
            self.ewma[channel] = np.zeros((n, dof))
            self.tbias[channel] = np.zeros(n)
            self.bias_active[channel] = np.zeros(n, dtype=bool)
            self.bias_up[channel] = np.zeros(n, dtype=int)
            self._thr[channel] = float(chi2.ppf(1.0 - self.alpha, dof))
            self._thr_bias[channel] = float(
                chi2.ppf(1.0 - self.alpha_bias, dof))
        return (self.lam[channel], self.up[channel], self.hold[channel],
                self.excluded[channel], self._thr[channel])

    def _mean_statistic(self, channel, nu, S_eff):
        """Prueba sobre el primer momento de la innovación blanqueada.

        La innovación blanqueada r es normal estándar bajo filtro consistente.
        Un sesgo del sensor desplaza su media sin alterar de forma apreciable
        su magnitud cuadrática instantánea, que es el punto ciego de todos los
        esquemas basados en el segundo momento. Ver docs/ANALYSIS.md H-P4.

        Se acumula una media exponencialmente ponderada de tasa beta. Bajo la
        hipótesis nula su covarianza asintótica es beta sobre dos menos beta
        por la identidad, de modo que el estadístico normalizado sigue una chi
        cuadrado con los grados de libertad del canal.
        """
        L = np.linalg.cholesky(S_eff)
        r = np.linalg.solve(L, nu[:, :, None])[:, :, 0]
        m = (1.0 - self.beta) * self.ewma[channel] + self.beta * r
        self.ewma[channel] = m
        scale = (2.0 - self.beta) / self.beta
        T = scale * np.sum(m * m, axis=1)
        self.tbias[channel] = T
        return T

    def covariance(self, channel, R_nom, nu, S_nom, nis_nom, t):
        n, dof = nu.shape
        lam, up, hold, excl, thr = self._state(channel, n, dof)

        # Segundo momento. Estadístico con la covarianza efectiva vigente.
        nis_eff, S_eff = _nis_with(nu, S_nom, R_nom, lam)
        ratio = nis_eff / float(dof)

        # Zona muerta. Fuera de la banda el factor se mueve, dentro no.
        above = ratio > (1.0 + self.dead_zone)
        below = ratio < (1.0 - self.dead_zone)
        step = np.power(np.maximum(ratio, 1.0e-6), self.rho)
        lam_new = np.where(above, np.minimum(lam * step, self.lam_max),
                           np.where(below, np.maximum(lam * step, 1.0), lam))

        # Primer momento. Detecta el sesgo que el segundo momento no ve.
        if self.mean_test:
            T = self._mean_statistic(channel, nu, S_eff)
            breach_bias = T > self._thr_bias[channel]
        else:
            breach_bias = np.zeros(n, dtype=bool)

        # Modo compensacion. La firma de sesgo ya no manda al canal a la
        # exclusion. Libera el termino de sesgo de ese canal en el estado del
        # filtro, que pasa de inerte a estimado. La activacion es de
        # enclavamiento, de modo que una falsa alarma tiene costo permanente.
        if self.compensate:
            bu = np.where(breach_bias, self.bias_up[channel] + 1, 0)
            self.bias_up[channel] = bu
            self.bias_active[channel] = (self.bias_active[channel]
                                         | (bu >= self.dwell_up))
            breach_bias = np.zeros(n, dtype=bool)

        # Compuerta con tiempo mínimo de permanencia.
        if self.gating:
            breach = (nis_eff > thr) | breach_bias
            up = np.where(breach, up + 1, 0)
            enter = (~excl) & (up >= self.dwell_up)
            hold = np.where(enter, self.dwell_hold, np.maximum(hold - 1, 0))
            # Readmisión solo si el canal ya no viola la prueba. Un fallo
            # persistente vuelve a excluirse, y el tiempo de permanencia acota
            # la frecuencia de esa conmutación en lugar de eliminarla.
            expire = excl & (hold <= 0) & (~breach)
            excl = np.where(enter, True, excl & ~expire)
            up = np.where(excl, 0, up)
            self.up[channel] = up
            self.hold[channel] = hold
            self.excluded[channel] = excl

        self.lam[channel] = lam_new
        return _broadcast(R_nom, n) * lam_new[:, None, None], lam_new

    def accept(self, channel, nu, S_nom, nis_nom, t):
        n = nu.shape[0]
        if not self.gating or channel not in self.excluded:
            return np.ones(n, dtype=bool)
        return ~self.excluded[channel]

    def config(self):
        return dict(name=self.name, rho=self.rho, dead_zone=self.dead_zone,
                    lam_max=self.lam_max, alpha=self.alpha,
                    dwell_up=self.dwell_up, dwell_hold=self.dwell_hold,
                    gating=self.gating, beta=self.beta,
                    alpha_bias=self.alpha_bias, mean_test=self.mean_test,
                    compensate=self.compensate)


def ablation_adapter(n_rep=1, **kw):
    """Ablación completa. Solo inflación acotada, sin los elementos dos y tres.

    Es el punto de comparación que aísla lo que aportan la zona muerta, la
    prueba del primer momento y la exclusión con tiempo de permanencia.
    """
    a = DwellTimeAdapter(n_rep=n_rep, dead_zone=0.0, gating=False,
                         mean_test=False, **kw)
    a.name = "dwell_ablation"
    return a
