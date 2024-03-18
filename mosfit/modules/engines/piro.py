"""Definitions for the `Shock Cooling` Piro Model class."""
from math import isnan
from astrocats.catalog.source import SOURCE

import numpy as np

from mosfit.constants import C_CGS, DAY_CGS,AU_CGS, FOUR_PI, KM_CGS, M_SUN_CGS
from mosfit.modules.engines.engine import Engine


# Important: Only define one ``Module`` class per file.


class Shock_Cooling(Engine):
    """

    Piro 2021 Shock Cooling Emission from Extended Material Revisited

    """

    DIFF_CONST = M_SUN_CGS / (FOUR_PI * C_CGS * KM_CGS)
    C_KMS = C_CGS / KM_CGS

    _REFERENCES = [
        {SOURCE.BIBCODE: '2021ApJ...909..209P'}
    ]

    def process(self, **kwargs):

        """Process module."""
        self._times = kwargs[self.key('dense_times')]
        self._rest_t_explosion = kwargs[self.key('texplosion')]
        self._kappa = kwargs[self.key('kappa')]
        self._v_ejecta = kwargs[self.key('vejecta')] * KM_CGS
        self._m_ejecta = kwargs[self.key('mejecta')] * M_SUN_CGS
        self._m_csm = kwargs[self.key('mcsm')] * M_SUN_CGS
        self._r_init = kwargs[self.key('rinit')]
        self._n = kwargs[self.key('n')]
        self._delta = kwargs[self.key('delta')]

        E_SN =  0.3 * self._m_ejecta * self._v_ejecta**2 # This should really be the photospheric velocity 
        E51 = E_SN / 1e51
        Ee = 2e49 * E51 * ( (self._m_ejecta / (3 * M_SUN_CGS) ) ** (-0.7) ) * ( ((self._m_csm / (0.01 * M_SUN_CGS))) ** (0.7) )
        K = (self._n-3)*(3 - self._delta) / ( FOUR_PI * (self._n-self._delta)) 
        vt = ( (((self._n -5 )*(5 - self._delta))/((self._n -3 )*(3 - self._delta))) ** 0.5 ) * ( (2 * Ee / (self._m_csm)) ** 0.5 )
        td = ((3 * self._kappa * K * self._m_csm )/((self._n - 1 ) * vt *C_CGS)) ** 0.5

        ts = [
            np.inf
            if self._rest_t_explosion > x else (x - self._rest_t_explosion)
            for x in self._times
        ]
        ts = np.array(ts)

        def shock_cooling_luminosity(L_0, td, n, t):
            if t <= td:
                return L_0 * ( (td / t) ** (4. / (n - 2)))
            if t > td:
                return L_0 * np.exp(-0.5 * ((t ** 2) / (td ** 2.)  - 1.))
            
        L_0 = ((np.pi * (self._n - 1.)) / (3. * (self._n - 5.))) * ((C_CGS * self._r_init * vt**2. )/(self._kappa))
            
        luminosities = [shock_cooling_luminosity(L_0, td, self._n, t) for t in ts]

        luminosities = [0.0 if isnan(x) else x for x in luminosities]

        return {self.dense_key('luminosities'): luminosities}
