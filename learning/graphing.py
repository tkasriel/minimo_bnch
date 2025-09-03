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


def make_graph (outcomes_filepath: str, output_folder: str) -> None:
    import dsprover
    random.seed(8)
    thm_logprob = random.sample(_extract_theorems_from_outcomes(outcomes_filepath, True), 100)
    adj = {}
    to_prove = []
    for thm_1 in thm_logprob:
        for thm_2 in thm_logprob:
            if thm_1 == thm_2:
                continue
            # if thm_2[0] != '((v0 : Nat) -> (v1 : (v0 = 0)) -> (v2 : (v0 = v0)) -> (v3 : ((Nat.succ 0) = v0)) -> (v4 : (0 = v0)) -> (v5 : (0 = v0)) -> ((Nat.succ (Nat.succ 0)) = 0))':
            #     continue
            print(len(to_prove))
            new_theorem = f"({thm_1[0]}) -> {thm_2[0]}"
            to_prove.append(new_theorem)
    print(len(to_prove))
    res = [r[0] for r in dsprover.prove(to_prove, debug=True) if r[0]]
    # with open(os.path.join(output_folder, "res_chosen.txt"), "w") as f:

    #     f.write("\n\n\n".join(res))
    # return
    index = 0
    for thm_1 in thm_logprob:
        for thm_2 in thm_logprob:
            if thm_1 == thm_2:
                continue
            logprob_new = res[index][1]
            if res[index][0] and logprob_new > thm_2[1]:
                if thm_1 not in adj:
                    adj[thm_1[0]] = []
                adj[thm_1[0]].append((thm_2[0], logprob_new - thm_2[1]))
            index += 1
    with open(os.path.join(output_folder, "graph.csv"), "w") as f:
        f.write("A,B,weight")
        for k,v in adj.items():
            for edge in v:
                f.write(f"{k},{edge[0]},{edge[1]}\n")

def draw_graph (input_filepath: str, output_folder: str) -> None:
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    df = pd.read_csv(input_filepath)
    G = nx.DiGraph()
    edges = []
    weights = []
    for _, row in df.iterrows():
        G.add_edge(row["A"],row["B"],weight=row["weight"])
        edges.append((row["A"], row["B"]))
        weights.append(row["weight"] * 10)
    
    plt.figure(figsize=(12,8))

    # edges, weights = nx.get_edge_attributes(G, "weight")
    # nodelist = G.nodes()
    pos = nx.shell_layout(G)
    nx.draw(G, pos, node_color='b', edgelist=edges, edge_color=weights, width=5.0, alpha=0.3, edge_cmap=plt.cm.Blues)

    plt.savefig(os.path.join(output_folder, "graph.png"))

def look_at_graph (input_filepath: str) -> None:
    import pandas as pd
    df = pd.read_csv(input_filepath)
    cnts = df["B"].value_counts()
    print(cnts)
    print("-------")
    cnts = df["A"].value_counts()
    print(cnts)
    # for v in cnts[]
            

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



