"""Planta. Vehículo terrestre de tracción diferencial en el plano.

Estado verdadero de dimensión seis.

    xi = [x, y, theta, v, omega, b_g]

con posición en metros, rumbo en radianes, velocidad longitudinal en metros
por segundo, velocidad angular en radianes por segundo y sesgo del giróscopo
en radianes por segundo.

Dinámica en tiempo continuo, modelo de uniciclo con retardo actuador de
primer orden en las dos entradas y con sesgo del giróscopo modelado como
camino aleatorio.

    x_punto      = v cos(theta)
    y_punto      = v sin(theta)
    theta_punto  = omega
    v_punto      = (v_cmd - v) / tau_v
    omega_punto  = (omega_cmd - omega) / tau_w
    b_g_punto    = ruido blanco

La no linealidad de este modelo es esencial para el estudio. Bajo planta
lineal y filtro correctamente especificado el problema se separa y la
hipótesis H1 carece de contenido. Ver docs/LITERATURE.md sección F.
"""
import numpy as np

STATE_DIM = 6
IX, IY, ITH, IV, IW, IBG = range(STATE_DIM)
STATE_NAMES = ("x", "y", "theta", "v", "omega", "b_g")
STATE_UNITS = ("m", "m", "rad", "m/s", "rad/s", "rad/s")

INPUT_DIM = 2
IU_V, IU_W = range(INPUT_DIM)


class PlantParams:
    """Parámetros físicos y de ruido de proceso de la planta."""

    def __init__(self, tau_v=0.35, tau_w=0.20, sigma_v=0.08, sigma_w=0.06,
                 sigma_bg=2.0e-4, v_max=2.0, w_max=2.5):
        self.tau_v = float(tau_v)      # constante de tiempo actuador lineal, s
        self.tau_w = float(tau_w)      # constante de tiempo actuador angular, s
        self.sigma_v = float(sigma_v)  # densidad de ruido en v_punto, m/s^1.5
        self.sigma_w = float(sigma_w)  # densidad de ruido en omega_punto
        self.sigma_bg = float(sigma_bg)  # densidad del camino aleatorio del sesgo
        self.v_max = float(v_max)      # saturación de comando lineal, m/s
        self.w_max = float(w_max)      # saturación de comando angular, rad/s

    def as_dict(self):
        return dict(tau_v=self.tau_v, tau_w=self.tau_w, sigma_v=self.sigma_v,
                    sigma_w=self.sigma_w, sigma_bg=self.sigma_bg,
                    v_max=self.v_max, w_max=self.w_max)


def saturate_input(u, p):
    """Aplica los límites físicos del actuador al comando."""
    u = np.asarray(u, dtype=float)
    return np.array([np.clip(u[IU_V], -p.v_max, p.v_max),
                     np.clip(u[IU_W], -p.w_max, p.w_max)])


def drift(xi, u, p):
    """Campo vectorial determinista f(xi, u). El sesgo no tiene deriva."""
    dxi = np.zeros(STATE_DIM)
    dxi[IX] = xi[IV] * np.cos(xi[ITH])
    dxi[IY] = xi[IV] * np.sin(xi[ITH])
    dxi[ITH] = xi[IW]
    dxi[IV] = (u[IU_V] - xi[IV]) / p.tau_v
    dxi[IW] = (u[IU_W] - xi[IW]) / p.tau_w
    dxi[IBG] = 0.0
    return dxi


def _diffusion_std(p, dt):
    """Desviación estándar del incremento de ruido de proceso en un paso."""
    s = np.zeros(STATE_DIM)
    s[IV] = p.sigma_v * np.sqrt(dt)
    s[IW] = p.sigma_w * np.sqrt(dt)
    s[IBG] = p.sigma_bg * np.sqrt(dt)
    return s


def step(xi, u, p, dt, rng=None):
    """Un paso de integración Runge Kutta de cuarto orden más ruido aditivo.

    El ruido se añade después de la integración determinista, esquema de
    Euler Maruyama sobre las componentes difusivas. Si rng es None el paso
    es determinista, lo que se usa para linealizar y para pruebas.
    """
    u = saturate_input(u, p)
    k1 = drift(xi, u, p)
    k2 = drift(xi + 0.5 * dt * k1, u, p)
    k3 = drift(xi + 0.5 * dt * k2, u, p)
    k4 = drift(xi + dt * k3, u, p)
    nxt = xi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if rng is not None:
        nxt = nxt + _diffusion_std(p, dt) * rng.standard_normal(STATE_DIM)
    return nxt


def process_noise_cov(p, dt):
    """Covarianza discreta equivalente del ruido de proceso, Q.

    Aproximación diagonal de primer orden, consistente con el esquema de
    integración usado en step. Es la Q que reciben los filtros y coincide
    con la planta solo en el caso nominal.
    """
    q = np.zeros(STATE_DIM)
    q[IV] = (p.sigma_v ** 2) * dt
    q[IW] = (p.sigma_w ** 2) * dt
    q[IBG] = (p.sigma_bg ** 2) * dt
    q[IX] = 1e-9
    q[IY] = 1e-9
    q[ITH] = 1e-9
    return np.diag(q)


_EYE = np.eye(STATE_DIM)


def jacobian_f(xi, u, p, dt):
    """Jacobiano discreto aproximado, F = I + dt * df/dxi.

    Suficiente para el filtro extendido con el paso de muestreo usado.
    """
    F = _EYE.copy()
    th, v = xi[ITH], xi[IV]
    F[IX, ITH] = -dt * v * np.sin(th)
    F[IX, IV] = dt * np.cos(th)
    F[IY, ITH] = dt * v * np.cos(th)
    F[IY, IV] = dt * np.sin(th)
    F[ITH, IW] = dt
    F[IV, IV] = 1.0 - dt / p.tau_v
    F[IW, IW] = 1.0 - dt / p.tau_w
    return F
