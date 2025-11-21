from get_logprobs_by_model import compute_logprobs_usefulness_revision
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

data_paths = [
    "outputs/line33",
    "/srv/share/minimo/line33_2",
    "/srv/share/minimo/line33_3",
]

# the model for model_paths[i] is evaluated on the data for data_paths[i]
model_paths = [
    "outputs/line33",
    "/srv/share/minimo/line33_2",
    "/srv/share/minimo/line33_3",
]


# --- baseline model paths ---
baseline_model_paths = [
    "/srv/share/minimo/line33.10/",
    "/srv/share/minimo/line33.10/",
    "/srv/share/minimo/line33.10/"
]

# baseline_model_paths = [
#     "/srv/share/minimo/line33_group/",
#     "/srv/share/minimo/line33_group/",
#     "/srv/share/minimo/line33_group/"
# ]

# main model
outcomes_by_model_all = []

for model_path, data_path in zip(model_paths, data_paths):
    cache_path = os.path.join("plots", os.path.basename(model_path)+f"_{path_prefix}_filtered_logprob_outcomes.json")

    if try_read:
        try:
            with open(cache_path, "r") as f:
                outcomes_by_model = json.load(f)
            outcomes_by_model_all.append(outcomes_by_model)
            print(f"successfully loaded main model from {cache_path}")
            continue
        except Exception:
            print(f"failed to load main model cache for {model_path}")

    # Pre-allocate structure
    outcomes_by_model = {str(m): {} for m in model_its}

    def _worker(model_path: str, data_path: str, model_it: int, theorem_it: int):
        # One task: compute and return identifiers with result
        out = compute_logprobs_usefulness_revision(model_path, data_path, model_it, theorem_it)
        return model_it, theorem_it, out

    tasks = [(model_path, data_path, m, t) for m in model_its for t in theorems_it]
    total = len(tasks)

    with ProcessPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_worker, *args) for args in tasks]
        for fut in tqdm(as_completed(futures), total=total, desc=f"Computing main @ {model_path}"):
            m, t, out = fut.result()
            m, t = str(m), str(t)
            outcomes_by_model[m][t] = out
    
    with open(cache_path, "w") as f:
        json.dump(outcomes_by_model, f)

    outcomes_by_model_all.append(outcomes_by_model)

# baseline
baseline_outcomes_by_model_all = []

# Only run if you actually have baselines defined
if baseline_model_paths:
    for baseline_model_path, data_path in zip(baseline_model_paths, data_paths):
        data_name = data_path.split("/")[-1]
        cache_path = os.path.join(
            "plots",
            "fix-" + baseline_model_path.split("/")[-2] + f"_{path_prefix}_{data_name}_filtered_logprob_outcomes.json"
        )


        if try_read:
            try:
                with open(cache_path, "r") as f:
                    outcomes_by_model = json.load(f)
                baseline_outcomes_by_model_all.append(outcomes_by_model)
                print(f"successfully loaded baseline model from {cache_path}")
                continue
            except Exception:
                print(f"failed to load baseline cache for {baseline_model_path}")

        outcomes_by_model = {str(m): {} for m in model_its}

        def _worker_baseline(model_path: str, data_path: str, model_it: int, theorem_it: int):
            out = compute_logprobs_usefulness_revision(model_path, data_path, model_it, theorem_it)
            return model_it, theorem_it, out

        tasks = [(baseline_model_path, data_path, m, t) for m in model_its for t in theorems_it]
        total = len(tasks)

        with ProcessPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(_worker_baseline, *args) for args in tasks]
            for fut in tqdm(as_completed(futures), total=total, desc=f"Computing baseline @ {baseline_model_path}"):
                m, t, out = fut.result()
                m, t = str(m), str(t)
                outcomes_by_model[m][t] = out

        with open(cache_path, "w") as f:
            json.dump(outcomes_by_model, f)

        baseline_outcomes_by_model_all.append(outcomes_by_model)

from matplotlib.lines import Line2D

def _aggregate_across_paths(outcomes_list, column):
    """Helper to aggregate mean(column) over all paths in outcomes_list."""
    rows = []
    for m in model_its:
        vals = []
        for t in theorems_it:
            for outcomes_by_model in outcomes_list:
                m_key = str(m)
                t_key = str(t)
                if m_key not in outcomes_by_model:
                    continue
                for d in outcomes_by_model[m_key].get(t_key, []):  # list of dicts
                    if d[column] != 1:  # keep your original filter
                        vals.append(d[column])
        mean_val = np.mean(vals) if vals else np.nan
        rows.append({"model_it": m, f"mean_{column}": mean_val})
    return pd.DataFrame(rows).sort_values("model_it")

def plot_agg(column):
    fontsize = 12

    # main models
    df_main = _aggregate_across_paths(outcomes_by_model_all, column)

    # baseline models (may be empty)
    df_baseline = None
    if baseline_outcomes_by_model_all:
        df_baseline = _aggregate_across_paths(baseline_outcomes_by_model_all, column)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Our method
    main_line, = ax.plot(
        df_main["model_it"],
        df_main[f"mean_{column}"],
        label="Our method",
        c = "green"
    )

    # Baseline curve (if any)
    if df_baseline is not None:
        baseline_line, = ax.plot(
            df_baseline["model_it"],
            df_baseline[f"mean_{column}"],
            label="No usefulness training",
            c = "red"
        )
        handles = [main_line, baseline_line]
        labels = ["Our method", "No usefulness training"]
    else:
        handles = [main_line]
        labels = ["Our method"]

    ax.set_xlabel("Iteration", fontsize=fontsize)
    ax.set_ylabel("Log-probability", fontsize=fontsize)
    ax.set_title("Arithmetic", y=1.0, pad=-14, fontsize=14)
    ax.set_xticks([i for i in range(0, 10)])
    ax.tick_params("both", labelsize=12)
    fig.tight_layout()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        frameon=True,
        edgecolor="lightgray",
        facecolor="whitesmoke",
        fontsize=12,
        ncol = 2
    )
    fig.subplots_adjust(top=0.87)

    # plt.ylim(-18, -2)
    plt.savefig(f"plots/revised_figure_{column}_vs_model-it_{path_prefix}.png")

plot_agg("original_logprob")
plot_agg("usefulness_logprob")
plot_agg("improvement")
