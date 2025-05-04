import json
import os
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from omegaconf import OmegaConf
import torch

from problems import load_natural_number_game_problemset
from proofsearch import evaluate_agent

visual_path = "vis"

def get_color(name: str) -> str:
    if "Minimo" in name and "50" in name:
        return "orange"
    if "Minimo" in name and "200" in name:
        return "red"
    if "Our" in name and "50" in name:
        return "navy"
    if "Our" in name and "200" in name:
        return "indigo"
    print(name)
    return "black"

def setup_plot (num_its: int, y_lims: tuple[int,int], y_name: str, exp_name: str) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    ax.set(xlim=(0,num_its), ylim=y_lims)
    ax.set_ylabel(y_name)
    ax.set_xlabel("Iteration count")
    ax.set_title(exp_name)
    return fig, ax

def get_max_it(fp: str) -> int:
    ind = 0
    files = os.listdir(fp)
    while f"{ind}.pt" in files:
        ind += 1
    assert ind > 0
    return ind


def graph_success_rate (fp: str, ax: Axes, color: str) -> None:
    max_iter = get_max_it(fp)
    with open(os.path.join(fp, f"outcomes_{max_iter-1}.json")) as f:
        data = json.load(f)
    it_solve_counts = np.zeros(max_iter)
    it_tot_counts = np.zeros(max_iter)
    for conj in data:
        if not conj["hindsight"]:
            it_tot_counts[conj["iteration"]] += 1
            it_solve_counts[conj["iteration"]] += 1 if conj["proof"] else 0
    assert np.all(it_tot_counts > 0)
    graph_counts = it_solve_counts / it_tot_counts
    exp_name = os.path.basename(fp).replace("_", " ")
    ax.plot(graph_counts, color=color, linewidth=1, label=exp_name)

def graph_log_prob (fp: str, ax: Axes, color: str) -> None:
    max_iter = get_max_it(fp)
    with open(os.path.join(fp, f"outcomes_{max_iter-1}.json")) as f:
        data = json.load(f)
    it_tot_prob = np.zeros(max_iter)
    it_tot_counts = np.zeros(max_iter)
    for conj in data:
        if conj["proof"]:
            it_tot_counts[conj["iteration"]] += 1
            it_tot_prob[conj["iteration"]] += np.exp(conj["logprob"])
    assert np.all(it_tot_counts > 0)
    graph_counts = np.log(it_tot_prob / it_tot_counts)
    exp_name = os.path.basename(fp).replace("_", " ")
    ax.plot(graph_counts, color=color, linewidth=1, label=exp_name)

def graph_log_prob_no_hs (fp: str, ax: Axes, color: str) -> None:
    max_iter = get_max_it(fp)
    with open(os.path.join(fp, f"outcomes_{max_iter-1}.json")) as f:
        data = json.load(f)
    it_tot_prob = np.zeros(max_iter)
    it_tot_counts = np.zeros(max_iter)
    for conj in data:
        if conj["proof"] and not conj["hindsight"]:
            it_tot_counts[conj["iteration"]] += 1
            it_tot_prob[conj["iteration"]] += np.exp(conj["logprob"])
    assert np.all(it_tot_counts > 0)
    graph_counts = np.log(it_tot_prob / it_tot_counts)
    exp_name = os.path.basename(fp).replace("_", " ")
    ax.plot(graph_counts, color=color, linewidth=1, label=exp_name)

def graph_thm_use (fp: str, ax: Axes, color: str) -> None:
    max_iter = get_max_it(fp)
    if not "generated_theorems_0.json" in os.listdir(fp):
        return # Minimo doesn't generate these
    counts: dict[str, int] = {}
    tot_counts = np.zeros(max_iter)
    for it in range(max_iter):
        with open(os.path.join(fp, f"generated_theorems_{it}.json")) as f:
            data = json.load(f)
        for thm in data:
            thm_name = thm["theorem"]
            freq = int(thm["freq_used"])
            if thm_name in counts and freq > counts[thm_name]:
                tot_counts[it] += freq-counts[thm_name]
                counts[thm_name] = freq
            if thm_name not in counts and freq > 0:
                tot_counts[it] += freq
                counts[thm_name] = freq
    exp_name = os.path.basename(fp).replace("_", " ")
    ax.plot(tot_counts, color=color, linewidth=1, label=exp_name)

def useful_thm_rate(fp: str, ax: Axes, color: str) -> None:
    max_iter = get_max_it(fp)
    if not "generated_theorems_0.json" in os.listdir(fp):
        return # Minimo doesn't generate these
    with open(os.path.join(fp, f"generated_theorems_{max_iter-1}.json")) as f:
        data = json.load(f)
    thm_count = np.zeros(max_iter)
    for thm in data:
        if int(thm["freq_used"] ) > 0:
            thm_count[int(thm["iter_generated"])] += 1
    exp_name = os.path.basename(fp).replace("_", " ")
    ax.plot(thm_count, color=color, linewidth=1, label=exp_name)


def graph_capabilities(fp: str, ax: Axes, color: str) -> None:
    """The Big Boi"""
    max_iter = get_max_it(fp)
    solve_counts = np.zeros(max_iter)
    conf = OmegaConf.create({"problemset": "nng"})
    exp_name = os.path.basename(fp).replace("_", " ")
    print(f"Current experiment: {exp_name}")
    for it in range(max_iter):
        print(f"Current iteration: {it+1}/{max_iter}")
        if torch.cuda.is_available():
            agent = torch.load(os.path.join(fp, f"{it}.pt"), weights_only=False)
        else:
            agent = torch.load(os.path.join(fp, f"{it}.pt"), weights_only=False, map_location=torch.device('cpu'))
        problems = evaluate_agent(conf, agent)
        solve_counts[it] = len(problems._solved)
    
    ax.plot(solve_counts, color=color, linewidth=1, label=exp_name)


def run_experiment_on_all (fp_all: str, exp_function: Callable, y_lims: tuple[int, int], y_name: str, exp_name: str = "") -> None:
    fig, ax = setup_plot(10, y_lims, y_name, exp_name)
    str(exp_function)
    exps = os.listdir(fp_all)
    for exp in sorted(exps):
        color = get_color(exp)
        exp_function(os.path.join(fp_all, exp), ax, color)
    fig.legend()
    fig.savefig(os.path.join(visual_path, f"{exp_function.__name__}.png"))


if __name__ == "__main__":
    os.makedirs(visual_path, exist_ok=True)
    outs = "learning/outputs/presentation_outputs"
    run_experiment_on_all(outs, graph_success_rate, (0,1), "")
    run_experiment_on_all(outs, graph_log_prob, (-5,0), "Log Probability")
    run_experiment_on_all(outs, graph_thm_use, (0,70), "Use count")
    run_experiment_on_all(outs, useful_thm_rate, (0, 10), "Generated Theorems")
    pset = load_natural_number_game_problemset()
    run_experiment_on_all(outs, graph_capabilities, (0,len(pset)+1), "Problems solved")

    