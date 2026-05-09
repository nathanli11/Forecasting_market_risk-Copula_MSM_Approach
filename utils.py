import os
import pandas as pd

def save_msm_table(
    results: dict,
    asset_name: str,
    directory: str = "results/msm",
) -> None:
    
    os.makedirs(directory, exist_ok=True)

    rows = []
    for k, msm in results.items():
        p = msm.params_
        rows.append({
            "k": k,
            "m0": round(p["m0"], 6),
            "sigma": round(p["sigma"], 6),
            "b": round(p["b"], 6),
            "gamma_k": round(p["gamma_k"], 6),
            "log_L": round(msm.log_likelihood_, 4),
        })

    df = pd.DataFrame(rows).set_index("k")
    path = os.path.join(directory, f"{asset_name}_table.csv")
    df.to_csv(path)
