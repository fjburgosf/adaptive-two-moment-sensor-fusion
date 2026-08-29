"""Controlador de seguimiento de trayectoria para uniciclo.

Ley de Kanayama. El error de seguimiento se expresa en el marco del vehículo

    e_x =  cos(th) (x_r - x) + sin(th) (y_r - y)
    e_y = -sin(th) (x_r - x) + cos(th) (y_r - y)
    e_th = wrap(th_r - th)

y el comando es

    v_cmd     = v_r cos(e_th) + k_x e_x
    omega_cmd = omega_r + v_r ( k_y e_y + k_th sin(e_th) )

Las ganancias son idénticas para todos los estimadores comparados. Esa
igualdad es la condición que permite atribuir cualquier diferencia en el lazo
cerrado al estimador y no al controlador.

El controlador consume el estado estimado. Bajo estado verdadero sirve como
cota superior de desempeño y se usa para verificar la implementación.
"""
import numpy as np

from .geometry import wrap_pi
from .plant import IX, IY, ITH


class KanayamaGains:
    def __init__(self, k_x=1.6, k_y=3.2, k_th=2.4):
        self.k_x = float(k_x)
        self.k_y = float(k_y)
        self.k_th = float(k_th)

    def as_dict(self):
        return dict(k_x=self.k_x, k_y=self.k_y, k_th=self.k_th)


def tracking_error(xi, ref_pose):
    """Error de seguimiento expresado en el marco del vehículo."""
    x_r, y_r, th_r = ref_pose[0], ref_pose[1], ref_pose[2]
    th = xi[ITH]
    dx, dy = x_r - xi[IX], y_r - xi[IY]
    c, s = np.cos(th), np.sin(th)
    return np.array([c * dx + s * dy,
                     -s * dx + c * dy,
                     wrap_pi(th_r - th)])


def command(xi, ref_pose, gains):
    """Comando de velocidad lineal y angular a partir del estado dado."""
    v_r, om_r = ref_pose[3], ref_pose[4]
    e = tracking_error(xi, ref_pose)
    v_cmd = v_r * np.cos(e[2]) + gains.k_x * e[0]
    om_cmd = om_r + v_r * (gains.k_y * e[1] + gains.k_th * np.sin(e[2]))
    return np.array([v_cmd, om_cmd]), e
