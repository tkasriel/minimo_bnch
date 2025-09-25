from get_logprobs_by_model import compute_logprobs_usefulness
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

path = "outputs/line33"

model_its = [i for i in range(0, 11)]
theorems_it = [i for i in range(1, 11)]

# Pre-allocate structure
outcomes_by_model = {m: {} for m in model_its}

def _worker(path: str, model_it: int, theorem_it: int):
    # One task: compute and return identifiers with result
    out = compute_logprobs_usefulness(path, model_it, theorem_it)
    return model_it, theorem_it, out

tasks = [(path, m, t) for m in model_its for t in theorems_it]
total = len(tasks)

with ProcessPoolExecutor(max_workers=8) as ex:
    futures = [ex.submit(_worker, *args) for args in tasks]
    for fut in tqdm(as_completed(futures), total=total, desc="Computing"):
        m, t, out = fut.result()
        outcomes_by_model[m][t] = out

def plot_agg(column):
    # Aggregate across all theorem_its per model_it
    rows = []
    for m in model_its:
        vals = []
        for t in theorems_it:
            for d in outcomes_by_model[m].get(t, []):  # list of dicts
                vals.append(d[column])
        mean_val = np.mean(vals) if vals else np.nan
        rows.append({"model_it": m, f"mean_{column}": mean_val})

    df = pd.DataFrame(rows).sort_values("model_it")

    # Plot: x = model_it, y = mean original_logprob
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["model_it"], df[f"mean_{column}"], marker="o")
    ax.set_xlabel("model_it")
    ax.set_ylabel(f"mean {column} (aggregated over theorems it 1-9)")
    ax.set_title(f"Mean {column} vs. model_it")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/figure_{column}_vs_model-it.png") 

plot_agg("original_logprob")
plot_agg("usefulness_logprob")
plot_agg("improvement")