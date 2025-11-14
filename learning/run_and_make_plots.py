from get_logprobs_by_model import compute_logprobs_usefulness
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os

try_read = True
model_its = [i for i in range(0, 10)]
theorems_it = [i for i in range(1, 10)]

path_prefix = "nat-mul"
paths = ["outputs/line33", "/srv/share/minimo/line33_2", "/srv/share/minimo/line33_3"]

# path_prefix = "propositional-logic"
# paths = ["/srv/share/minimo/line33_group", "/srv/share/minimo/line33_group_2", "/srv/share/minimo/line33_group_3"]


outcomes_by_model_all = []
for path in paths:
    cache_path = os.path.join("plots", os.path.basename(path)+f"_{path_prefix}_logprob_outcomes.json")

    if (try_read):
        try:
            with open(cache_path, "r") as f:
                outcomes_by_model = json.load(f)

            outcomes_by_model_all.append(outcomes_by_model)
            print(f"successfully loaded {path}")
            continue
        except Exception:
            print(f"failed to load {path}")



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
    
    with open(cache_path, "w") as f:
        json.dump(outcomes_by_model, f)

    outcomes_by_model_all.append(outcomes_by_model)

from matplotlib.lines import Line2D

def plot_agg(column):
    # Aggregate across all theorem_its per model_it
    fontsize=12
    rows = []
    for m in model_its:
        vals = []
        for t in theorems_it:
            for outcomes_by_model in outcomes_by_model_all:
                for d in outcomes_by_model[str(m)].get(str(t), []):  # list of dicts
                    if not (d[column] == 1):
                        vals.append(d[column])
        mean_val = np.mean(vals) if vals else np.nan
        rows.append({"model_it": m, f"mean_{column}": mean_val})

    df = pd.DataFrame(rows).sort_values("model_it")

    # Plot: x = model_it, y = man original_logprob
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["model_it"], df[f"mean_{column}"], c = "g")
    ax.set_xlabel("Iteration", fontsize=fontsize)
    ax.set_ylabel(f"Log-probability", fontsize=fontsize)
    ax.set_title(f"Arithmetic", y=1.0, pad=-14, fontsize =14)
    ax.set_xticks([i for i in range(0, 10)])
    ax.tick_params("both", labelsize=12)
    fig.tight_layout()

    handles = [
        Line2D([0], [0], color="g", linestyle='-', label="Our method")
    ]
    fig.legend(handles, ["Our method"], loc="upper center", frameon=True, edgecolor='lightgray', facecolor="whitesmoke", fontsize=12)
    fig.subplots_adjust(top = 0.87)

    plt.ylim(-18, -2)
    plt.savefig(f"plots/figure_{column}_vs_model-it_{path_prefix}.png") 

plot_agg("original_logprob")
plot_agg("usefulness_logprob")
plot_agg("improvement")