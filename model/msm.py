"""
The MSM models centered returns as:
    y_t = sigma_t * eps_t, eps_t follows N(0,1)

    where sigma_t^2 = sigma^2 * prod{i=1,...,k} M_t(i)

    Each component M_t^{(i)} follows an independent two-state Markov chain with transition probability:
    gamma_i = 1 - (1 - gamma_1)^(b^{i-1})

    When a component switches, its new value is drawn from a two-point
    distribution with support {v0, 2 - v0}, each with probability 1/2.
"""

import logging
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from itertools import product as iterproduct

logger = logging.getLogger(__name__)


class MSMModel:
    """Markov Switching Multifractal Volatility Model"""

    def __init__(self, k: int = 5):
        self.k = k
        self.n = 2 ** k  # number of states

        self.params_: dict | None = None
        self.states_: np.ndarray | None = None
        self.h_: np.ndarray | None = None
        self.gamma_: np.ndarray | None = None
        self.A_: np.ndarray | None = None
        self.state_probs_: np.ndarray | None = None
        self.log_likelihood_: float | None = None

    def fit(self, returns: np.ndarray) -> "MSMModel":
        """
        Estimate MSM parameters by Maximum Likelihood.
        Input : returns : centered log-returns (x100)
        """
        self._returns = np.asarray(returns, dtype=float)

        logger.info(f"MSMModel(k={self.k}) — starting MLE on T={len(self._returns)} observations")
        logger.info(f"n_states = 2^{self.k} = {self.n}")
        logger.info(f"returns : mean={self._returns.mean():.4f}  std={self._returns.std():.4f}")

        theta0 = np.array([1.5, np.std(self._returns), 5.0, 0.5])
        bounds = [
            (1.001, 1.999),
            (1e-4,  10.0),
            (1.001, 50.0),
            (1e-6,  0.999),
        ]
        logger.info("  Running L-BFGS-B optimisation...")

        self._iter_count = 0
        result = minimize(
            fun=self._neg_log_likelihood,
            x0=theta0,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-12},
        )

        m0, sigma, b, gamma_k = result.x

        self.params_ = {
            "m0":      m0,
            "sigma":   sigma,
            "b":       b,
            "gamma_k": gamma_k,
        }
        self.log_likelihood_ = -result.fun

        logger.info(f"Optimisation done in {self._iter_count} likelihood evaluations")
        logger.info(f"Converged : {result.success}  —  {result.message}")
        logger.info(f"Estimated parameters :")
        logger.info(f" m0      = {m0:.6f}")
        logger.info(f" sigma   = {sigma:.6f}")
        logger.info(f" b       = {b:.6f}")
        logger.info(f" gamma_k = {gamma_k:.6f}")
        logger.info(f" log L = {self.log_likelihood_:.4f}")

        logger.info("  Building model objects at estimated parameters...")
        self.states_ = self._build_states(m0)
        self.h_ = self._compute_h(self.states_)
        self.gamma_ = self._compute_gammas(b, gamma_k)

        logger.info(f" Transition probabilities gamma_i : {np.round(self.gamma_, 6)}")

        self.A_ = self._build_transition_matrix(self.gamma_, self.states_)

        logger.info("  Running final Hamilton filter to store state probabilities...")
        _, self.state_probs_ = self._hamilton_filter(
            self._returns, sigma, self.h_, self.A_
        )

        logger.info(f"MSMModel(k={self.k}) — fit complete")
        return self

    def predict_cdf(self, y: float, t: int) -> float:
        """
        Conditional CDF F(y | I_{t-1}) — Eq. (9).
        F(y_t | I_{t-1}) = sum_j P(M_{t-1} = m_j | I_{t-1}) * Phi(y / (sigma * h(m_j)))
        """
        self._check_fitted()
        sigma = self.params_["sigma"]
        probs = self.state_probs_[t]
        cdf = np.sum(probs * norm.cdf(y / (sigma * self.h_)))
        return float(cdf)

    def predict_cdf_series(self, returns: np.ndarray) -> np.ndarray:
        """Compute the full series of conditional CDF values — Eq. (9)."""
        self._check_fitted()
        returns = np.asarray(returns, dtype=float)
        sigma   = self.params_["sigma"]

        u = np.array([
            np.sum(self.state_probs_[t] * norm.cdf(returns[t] / (sigma * self.h_)))
            for t in range(len(returns))
        ])
        return u

    # -------------------------------------------------------------------------
    # Private — model building
    # -------------------------------------------------------------------------

    def _build_states(self, m0: float) -> np.ndarray:
        """
        Enumerate all n = 2^k volatility state vectors.
        Each component takes value m0 or (2 - m0).
        """
        values = [m0, 2.0 - m0]
        states = np.array(list(iterproduct(values, repeat=self.k)))
        return states

    def _compute_h(self, states: np.ndarray) -> np.ndarray:
        """h(m_j) = sqrt(prod_{i=1}^k m_j^(i)) — denominator in Eq. (7)."""
        h = np.sqrt(np.prod(states, axis=1))
        return h

    def _compute_gammas(self, b: float, gamma_k: float) -> np.ndarray:
        """
        Transition probabilities for each component — Eq. (4):
            gamma_i = 1 - (1 - gamma_1)^(b^{i-1})
        """
        gamma_1 = 1.0 - (1.0 - gamma_k) ** (1.0 / b ** (self.k - 1))
        gamma_1 = np.clip(gamma_1, 1e-12, 1.0 - 1e-12)
        gammas  = np.array([
            1.0 - (1.0 - gamma_1) ** (b ** i)
            for i in range(self.k)
        ])

        return np.clip(gammas, 1e-12, 1.0 - 1e-12)

    def _build_transition_matrix(
        self,
        gamma: np.ndarray,
        states: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build the (2^k x 2^k) Markov transition matrix A over all states."""
        n = self.n
        if states is None:
            states = self.states_
        A = np.ones((n, n))

        for i in range(self.k):
            gi = gamma[i]
            for j in range(n):
                for l in range(n):
                    same = (states[j, i] == states[l, i])
                    A[j, l] *= (1.0 - gi / 2.0) if same else (gi / 2.0)


        return A

    def _hamilton_filter(
        self,
        returns: np.ndarray,
        sigma: float,
        h: np.ndarray,
        A: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Hamilton (1994) filter — computes log-likelihood and filtered
        state probabilities P(M_t = m_j | I_{t-1}).
        """
        T = len(returns)
        n = self.n
        log_lik = 0.0
        state_probs = np.zeros((T, n))
        pi = np.ones(n) / n    # uniform stationary prior

        for t in range(T):
            # Prediction step : P(M_t = m_j | I_{t-1})
            pi_pred = A.T @ pi
            pi_pred = np.maximum(pi_pred, 1e-300)
            state_probs[t] = pi_pred

            # Conditional densities f(y_t | M_t = m_j) — Eq. (7)
            f_j = norm.pdf(returns[t], loc=0.0, scale=sigma * h)

            # Marginal likelihood f(y_t | I_{t-1}) — Eq. (6)
            f_t = np.dot(pi_pred, f_j)
            if f_t <= 0:
                f_t = 1e-300
            log_lik += np.log(f_t)

            # Update step : P(M_t = m_j | I_t)
            pi = (pi_pred * f_j) / f_t
            pi = np.maximum(pi, 1e-300)
            pi /= pi.sum()

        return log_lik, state_probs

    def _neg_log_likelihood(self, theta: np.ndarray) -> float:
        """Negative log-likelihood for optimisation."""
        m0, sigma, b, gamma_k = theta
        self._iter_count += 1

        states  = self._build_states(m0)
        h = self._compute_h(states)
        gamma = self._compute_gammas(b, gamma_k)
        A = self._build_transition_matrix(gamma, states)
        log_lik, _ = self._hamilton_filter(self._returns, sigma, h, A)

        return -log_lik

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.params_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

    def __repr__(self) -> str:
        if self.params_ is None:
            return f"MSMModel(k={self.k}, fitted=False)"
        p = self.params_
        return (
            f"MSMModel(k={self.k}, "
            f"m0={p['m0']:.3f}, sigma={p['sigma']:.3f}, "
            f"b={p['b']:.3f}, gamma_k={p['gamma_k']:.4f}, "
            f"log_L={self.log_likelihood_:.2f})"
        )
    
    def save(self, path: str) -> None:
        """
        Save the fitted MSMModel to a pickle file.

        Parameters
        ----------
        path : str
            Full file path, e.g. "results/msm/NASDAQ_k3.pkl"
        """
        self._check_fitted()
        import pickle, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"MSMModel(k={self.k}) — saved to '{path}'")

    @classmethod
    def load(cls, path: str) -> "MSMModel":
        """
        Reload a fitted MSMModel from a pickle file.

        Parameters
        ----------
        path : str
            Path to the .pkl file saved by save().
        """
        import pickle
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"MSMModel(k={model.k}) — loaded from '{path}'")
        return model