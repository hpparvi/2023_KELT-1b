from pathlib import Path

from numpy import radians, median, squeeze, log, ones, array
from pytransit import NormalPrior as NP
from pytransit.lpf.loglikelihood import CeleriteLogLikelihood
from pytransit.lpf.baselines.linearbaseline import LinearModelBaseline
from pytransit.lpf.phasecurvelpf import PhaseCurveLPF

from src.io import read_tess_spoc
from src.kelt1 import (zero_epoch, period, star_rho, )


class KELT1LPF(PhaseCurveLPF):
    noise_models = ('gp', 'white')

    def __init__(self, savedir: Path = Path('results'), noise_model='gp'):
        """KELT-1b TESS S17 & S57 LPF.
        """
        if noise_model in self.noise_models:
            self.noise_model = noise_model
        else:
            raise ValueError('The noise model should be either "white" or "gp".')

        run_name = '01_gp_fit' if noise_model == 'gp' else '02_white_noise_fit'

        t1, f1, _, _, _ = read_tess_spoc(432549364, 'data', sectors=[17], use_pdc=True, use_quality=True)
        t2, f2, _, _, _ = read_tess_spoc(432549364, 'data', sectors=[57], use_pdc=True, use_quality=True)

        covs = [array([[]]), array([[]])]
        self.ins = ['S17', 'S57']

        super().__init__(run_name, ['T17', 'T57'], [t1, t2], [f1, f2], pbids=[0, 1], wnids=[0, 1],
                         covariates=covs, result_dir=savedir)

    def _post_initialisation(self):
        super()._post_initialisation()
        self.set_prior('tc', 'NP', zero_epoch.n, 0.01)  # - Zero epoch: normal prior with an inflated uncertainty
        self.set_prior('p', 'NP', period.n, 3 * period.s)  # - Orbital period: normal prior with an inflated uncertainty
        self.set_prior('rho', 'NP', star_rho.n, 3 * star_rho.s)  # - Stellar density: wide normal prior
        self.set_prior('secw', 'NP', 0.0, 1e-6)  # - Circular orbit: sqrt(e) cos(w) and sqrt(e) sin(w) forced
        self.set_prior('sesw', 'NP', 0.0, 1e-6)  # to zero with normal priors.
        self.set_prior('k2', 'NP', 0.078 ** 2, 0.001)  # - Area ratio: wide normal prior based on Siverd et al. (2012)

        # Set the default phase curve priors
        # ----------------------------------
        self.set_prior('oev', 'NP', 0.0, 0.09)
        for pb in self.passbands:
            self.set_prior(f'aev_{pb}', 'UP', 0.0, 1000e-6)  # - Ellipsoidal variation amplitude
            self.set_prior(f'log10_ted_{pb}', 'UP', -3.0, 0.0)  # - Emission dayside flux ratio
            self.set_prior(f'log10_ten_{pb}', 'UP', -4.0, 0.0)  # - Emission nightside flux ratio
            self.set_prior(f'teo_{pb}', 'NP', 0.0, radians(10))  # - Emission peak offset

        # Set the GP hyperparameter priors
        # --------------------------------
        if self.noise_model == 'gp':
            self.set_prior('gp_T17_ln_out', 'NP', round(log(self.fluxes[0].std()), 1), 1.0)
            self.set_prior('gp_T57_ln_out', 'NP', round(log(self.fluxes[1].std()), 1), 1.0)
            self.set_prior('gp_T17_ln_in', 'NP', 1.0, 1.0)
            self.set_prior('gp_T57_ln_in', 'NP', 1.0, 1.0)

        # Set a prior on the geometric albedo
        # -----------------------------------
        for pb in self.passbands:
            self.set_prior(f'ag_{pb}', 'NP', 1e-4, 1e-6)

        pr_aev_difference = NP(0.0, 0.01)
        def aev_difference(pvp):
            return pr_aev_difference.logpdf((pvp[:, 8] - pvp[:, 14]) / pvp[:, 14])

        self.add_prior(aev_difference)

    # Define the log likelihoods
    # --------------------------
    # Use a Celerite GP log likelihood for the two TESS light curves.
    def _init_lnlikelihood(self):
        if self.noise_model == 'gp':
            self._add_lnlikelihood_model(CeleriteLogLikelihood(self, noise_ids=[0], name='gp_T17'))
            self._add_lnlikelihood_model(CeleriteLogLikelihood(self, noise_ids=[1], name='gp_T57'))
        else:
            super()._init_lnlikelihood()

    def _init_baseline(self):
        self._add_baseline_model(LinearModelBaseline(self))

    def lnposterior(self, pv):
        return squeeze(super().lnposterior(pv))
