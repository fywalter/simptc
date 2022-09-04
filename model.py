from sklearn.mixture import BayesianGaussianMixture
import warnings
from abc import ABCMeta, abstractmethod
from time import time

import numpy as np
from scipy.special import logsumexp

from sklearn.utils import check_random_state
from sklearn.exceptions import ConvergenceWarning

class VariationalSimPTC(BayesianGaussianMixture):
    def __init__(self, *args, **kwargs):
       super(VariationalSimPTC, self).__init__(*args, **kwargs)

    def initialize_parameters_with_data(self, X, labels):
        self._check_initial_parameters(X)
        n_samples, _ = X.shape
        n_components = np.max(labels)+1
        resp = np.zeros((n_samples, n_components))
        resp[np.arange(n_samples), labels] = 1
        self._initialize(X, resp)
        self.lower_bound_ = -np.inf

    def fit_predict(self, X, y=None):
        """Estimate model parameters using X and predict the labels for X.
        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.
        .. versionadded:: 0.20
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        y : Ignored
            Not used, present for API consistency by convention.
        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_initial_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        # do_init = not (self.warm_start and hasattr(self, "converged_")) # modificatioin!!!!!
        # do_init = not self.warm_start
        n_init = 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = check_random_state(self.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            # if do_init:
            #     self.initialize_parameters_with_data(X, y) # modificatioin!!!!!

            lower_bound = self.lower_bound_ # modificatioin!!!!!

            for n_iter in range(1, self.max_iter + 1):
                prev_lower_bound = lower_bound

                log_prob_norm, log_resp = self._e_step(X)
                self._m_step(X, log_resp)
                lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)

                change = lower_bound - prev_lower_bound
                self._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.tol:
                    self.converged_ = True
                    break

            self._print_verbose_msg_init_end(lower_bound)

            if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = n_iter

        # if not self.converged_:
        #     warnings.warn(
        #         "Initialization %d did not converge. "
        #         "Try different init parameters, "
        #         "or increase max_iter, tol "
        #         "or check for degenerate data." % (init + 1),
        #         ConvergenceWarning,
        #     )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X)

        return log_resp.argmax(axis=1)