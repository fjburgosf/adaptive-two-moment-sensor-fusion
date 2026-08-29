"""Realizaciones de ruido pregeneradas y compartidas entre estimadores.

Motivo. La comparación pareada exige que dos estimadores distintos vean la
misma realización aleatoria. El primer ejecutor escalar no lo garantizaba,
porque los sorteos de valor atípico y de pérdida de datos se consumían de
forma condicional y por tanto en un orden que dependía del estimador. Aquí
todos los sorteos se generan por adelantado, en arreglos indexados por paso y
por repetición, de modo que el consumo es independiente de lo que haga el
filtro.

La trayectoria verdadera sí difiere entre estimadores, porque el controlador
consume el estado estimado. Eso es justamente lo que el estudio mide. Lo que
se mantiene idéntico es la entrada aleatoria, no la salida.
"""
import numpy as np

from .plant import STATE_DIM
from .sensors import CHANNELS, DECIMATION, NOMINAL_STD


class NoiseBook:
    """Sorteos pregenerados para un escenario y un conjunto de repeticiones."""

    def __init__(self, n_steps, n_rep, seed_material):
        seq = np.random.SeedSequence(seed_material)
        r_proc, r_init, r_meas, r_out, r_drop = [
            np.random.default_rng(s) for s in seq.spawn(5)]

        self.n_steps = int(n_steps)
        self.n_rep = int(n_rep)

        # Ruido de proceso en unidades normalizadas. La escala física la
        # aplica el integrador, que conoce dt y las densidades de la planta.
        self.w = r_proc.standard_normal((n_steps, n_rep, STATE_DIM))

        # Perturbación del estado inicial del filtro, en unidades de P0.
        self.x0 = r_init.standard_normal((n_rep, STATE_DIM))

        # Ruido de medición normalizado, sorteo de atípicos y de pérdida.
        self.v = {}
        self.out_u = {}
        self.out_sign = {}
        self.drop_u = {}
        self.drop_len = {}
        for c in CHANNELS:
            dof = NOMINAL_STD[c].size
            n_c = n_steps // DECIMATION[c] + 1
            self.v[c] = r_meas.standard_normal((n_c, n_rep, dof))
            self.out_u[c] = r_out.random((n_c, n_rep))
            self.out_sign[c] = r_out.choice([-1.0, 1.0], size=(n_c, n_rep, dof))
            self.drop_u[c] = r_drop.random((n_c, n_rep))
            self.drop_len[c] = r_drop.exponential(1.0, size=(n_c, n_rep))

    def sample_index(self, channel, k):
        """Índice de muestra del canal correspondiente al paso k de la base."""
        return k // DECIMATION[channel]
