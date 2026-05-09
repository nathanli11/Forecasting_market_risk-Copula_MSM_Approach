import os
import logging

from data_loader import DataLoader
from model.msm import MSMModel
from utils import save_msm_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

MSM_DIR = "results/msm"


def run_or_load_msm(returns, asset_name, k_range=range(1, 8)):
    """
    Load MSM models from disk if available, otherwise run MLE and save.
    Avoids rerunning the full estimation on subsequent runs.
    """
    results = {}
    for k in k_range:
        path = os.path.join(MSM_DIR, f"{asset_name}_k{k}.pkl")
        if os.path.exists(path):
            results[k] = MSMModel.load(path)
        else:
            results[k] = MSMModel(k=k).fit(returns)
            results[k].save(path)
    return results


def select_best_k(results: dict) -> int:
    """Return the k with the highest log-likelihood."""
    return max(results, key=lambda k: results[k].log_likelihood_)


def print_table(results: dict, asset_name: str) -> None:
    """Print the k-selection summary table to the console."""
    print(f"\n--- Table 2 ({asset_name}) ---")
    print(f"{'k':>3} {'m0':>8} {'sigma':>8} {'b':>8} {'gamma_k':>10} {'log L':>12}")
    for k, msm in results.items():
        p = msm.params_
        print(
            f"{k:>3} {p['m0']:>8.3f} {p['sigma']:>8.3f} "
            f"{p['b']:>8.3f} {p['gamma_k']:>10.4f} "
            f"{msm.log_likelihood_:>12.1f}"
        )


# ── 1. Données ─────────────────────────────────────────────────────────
loader   = DataLoader().load()
r_nasdaq = loader.returns["r_NASDAQ"].values
r_sp500  = loader.returns["r_SP500"].values

# ── 2. MSM — cache automatique ─────────────────────────────────────────
print("\n=== Sélection de k (NASDAQ) ===")
results_nasdaq = run_or_load_msm(r_nasdaq, "NASDAQ")

print("\n=== Sélection de k (SP500) ===")
results_sp500 = run_or_load_msm(r_sp500, "SP500")

# ── 3. Tables récapitulatives ──────────────────────────────────────────
print_table(results_nasdaq, "NASDAQ")
print_table(results_sp500,  "SP500")

save_msm_table(results_nasdaq, "NASDAQ", directory=MSM_DIR)
save_msm_table(results_sp500,  "SP500",  directory=MSM_DIR)

# ── 4. Sélection du k optimal ──────────────────────────────────────────
best_k_nasdaq = select_best_k(results_nasdaq)
best_k_sp500  = select_best_k(results_sp500)
print(f"\nk optimal NASDAQ : {best_k_nasdaq}")
print(f"k optimal SP500  : {best_k_sp500}")

msm_nasdaq = results_nasdaq[best_k_nasdaq]
msm_sp500  = results_sp500[best_k_sp500]

# ── 5. PITs pour la copule ─────────────────────────────────────────────
u1 = msm_nasdaq.predict_cdf_series(r_nasdaq)
u2 = msm_sp500.predict_cdf_series(r_sp500)
