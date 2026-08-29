"""Trayectoria de referencia. Lemniscata de Gerono, figura de ocho.

    x_r(t) = A sin(w t)
    y_r(t) = (A/2) sin(2 w t)

La curva se elige porque tiene curvatura de signo alternante y velocidad
lineal de referencia acotada lejos de cero, condición que el controlador de
seguimiento necesita para que el error lateral converja. Las derivadas se
obtienen de forma analítica, de modo que la referencia no introduce error
numérico propio en la comparación entre estimadores.
"""
import numpy as np

from .geometry import wrap_pi


class Lemniscate:
    """Referencia de figura de ocho con amplitud y periodo configurables."""

    def __init__(self, amplitude=6.0, period=40.0):
        self.A = float(amplitude)
        self.T = float(period)
        self.w = 2.0 * np.pi / self.T

    def pose(self, t):
        """Devuelve x_r, y_r, theta_r, v_r, omega_r en el instante t."""
        A, w = self.A, self.w
        s1, c1 = np.sin(w * t), np.cos(w * t)
        s2, c2 = np.sin(2.0 * w * t), np.cos(2.0 * w * t)
        x = A * s1
        y = 0.5 * A * s2
        dx = A * w * c1
        dy = A * w * c2
        ddx = -A * w * w * s1
        ddy = -2.0 * A * w * w * s2
        speed_sq = dx * dx + dy * dy
        v = np.sqrt(speed_sq)
        th = np.arctan2(dy, dx)
        om = (dx * ddy - dy * ddx) / speed_sq
        return x, y, wrap_pi(th), v, om

    def speed_bounds(self, n=2001):
        """Cotas de v_r y de omega_r sobre un periodo completo."""
        t = np.linspace(0.0, self.T, n)
        vals = np.array([self.pose(ti)[3:] for ti in t])
        return (vals[:, 0].min(), vals[:, 0].max(),
                np.abs(vals[:, 1]).max())
