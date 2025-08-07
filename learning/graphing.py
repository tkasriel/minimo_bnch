import json
import os
import matplotlib.pyplot as plt

OUTPUT_FOLDER="/home/timothekasriel/minimo/learning/graphs"

def _make_graph (name: str, y_axis: str, labels: list[str], values: list[list[float]], filename: str) -> None:
    fig,ax = plt.subplots()
    ax.set_ylabel(y_axis)
    ax.set_xlabel("Iteration")
    ax.set_title(name)
    for label, vals in zip(labels, values):
        ax.plot(range(len(vals)), vals, label=label)
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_FOLDER, filename))

def make_variable_use_count_graph (outcome_filepaths: list[str]) -> None:
    outs = []
    exp_names = []
    for outcome_filepath in outcome_filepaths:
        exp_name = os.path.basename(os.path.dirname(outcome_filepath))
        var_counts = [0 for i in range(10)]
        instances = [0 for i in range(10)]
        with open(outcome_filepath) as f:
            thms = json.load(f)
        for thm in thms:
            if thm["hindsight"]:
                continue
            var_count = thm["problem_translated"].count("v") - thm["problem_translated"].count("Nat")
            var_counts[int(thm["iteration"])] += var_count
            instances[int(thm["iteration"])] += 1
        outs.append([v / i for v,i in zip(var_counts, instances)])
        exp_names.append(exp_name)
    _make_graph("Variable Use Count / Iteration", "Use Count", exp_names, outs, f"var_use_count.jpg")

def make_success_rate_graph (outcome_filepaths: list[str]) -> None:
    outs = []
    exp_names = []
    for outcome_filepath in outcome_filepaths:
        exp_name = os.path.basename(os.path.dirname(outcome_filepath))
        success_counts = [0 for i in range(10)]
        instances = [0 for i in range(10)]
        with open(outcome_filepath) as f:
            thms = json.load(f)
        for thm in thms:
            if thm["hindsight"]:
                continue
            it = int(thm["iteration"])
            instances[it] += 1
            if thm["proof"]:
                success_counts[it] += 1
        outs.append([s / i for s,i in zip(success_counts, instances)])
        exp_names.append(exp_name)
    _make_graph("Proof Success Rate / Iteration", "Success Rate", exp_names, outs, f"success_rate.jpg")

def make_logprob_graph (outcome_filepaths: list[str]) -> None:
    outs = []
    exp_names = []
    for outcome_filepath in outcome_filepaths:
        exp_name = os.path.basename(os.path.dirname(outcome_filepath))
        tot_logprob = [0.0 for i in range(10)]
        instances = [0 for i in range(10)]
        with open(outcome_filepath) as f:
            thms = json.load(f)
        for thm in thms:
            if thm["hindsight"]:
                continue
            it = int(thm["iteration"])
            if thm["proof"]:
                instances[it] += 1
                tot_logprob[it] += float(thm["logprob"])
        outs.append([s / i for s,i in zip(tot_logprob, instances)])
        exp_names.append(exp_name)
    _make_graph("Average Logprob / Iteration", "Logprob", exp_names, outs, f"logprobs.jpg")
            
            

if __name__ == "__main__":
    exps = [
        "/home/timothekasriel/minimo/learning/outputs/line19/outcomes_9.json",
        # "/home/timothekasriel/minimo/learning/outputs/line18/outcomes_9.json",
        # "/home/timothekasriel/minimo/learning/outputs/line15/outcomes_9.json",
        # "/home/timothekasriel/minimo/learning/outputs/line10/outcomes_9.json"
        ]
    make_success_rate_graph(exps)
    make_variable_use_count_graph(exps)
    make_logprob_graph(exps)



