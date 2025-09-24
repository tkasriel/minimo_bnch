import ast
import json
import os
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from classes import UsefulConjectureList, LLMUsefulnessEvalResult, LLMUsefulnessEvalTheorem, UsefulnessOutcomeList

OUTPUT_FOLDER="/home/timothekasriel/minimo/learning/graphs"

def _make_graph (name: str, y_axis: str, labels: list[str] | None, values: list[list[float]], filename: str, styles: list[str] = [], error_bars = [], ax = None, legend=True, ax_label=True, fontsize=10) -> None:
    if not ax:
        fig,ax = plt.subplots()
    ax.set_ylabel(y_axis, fontsize=fontsize)
    ax.tick_params("both", labelsize=12)
    if ax_label:
        ax.set_xlabel("Iteration", fontsize=fontsize)
    ax.set_xticks(range(max([len(vals) for vals in values])))

    if int(np.max(values)) > 0:
        # print(int(np.max(values)))
        # print(int(np.max(values)) > 0)
        # print(np.max(values) * 1.1)
        # print((np.max(values)*1.1) // 10)
        if error_bars:
            yticks = np.arange(0,np.max(np.add(values, error_bars))*1.1, max((np.max(np.add(values, error_bars))*1.1) // 5, 1))
            ax.set_ylim(top=np.max(values + error_bars) * 1.1)
        else:
            yticks = np.arange(0,np.max(values)*1.1, max((np.max(values)*1.1) // 5, 1))
            ax.set_ylim(top=np.max(values) * 1.1)
        ax.set_yticks(yticks)
    ax.set_title(name, y=0.9, fontsize=int(1.2 * fontsize))
    ax.set_ylim(bottom=-1)
    if styles:
        if labels:
            for label, vals, style in zip(labels, values, styles):
                ax.plot(range(len(vals)), vals, style, label=label)
        else:
            if error_bars:
                for vals, bar, style in zip(values, error_bars, styles):
                    ax.plot(range(len(vals)), vals, style)
                    ax.fill_between(range(len(vals)), 
                                    [max(v-b, 0) for v,b in zip(vals,bar)], 
                                    [v+b for v,b in zip(vals,bar)],
                                    color=style[0],
                                    alpha=0.2)
            else:
                for vals, style in zip(values, styles):
                    ax.plot(range(len(vals)), vals, style)

    else:
        if labels:
            for label, vals in zip(labels, values):
                ax.plot(range(len(vals)), vals, label=label)
    if legend:
        ax.legend()
    ax.set_xticks(range(len(values[0])))
    plt.savefig(os.path.join(OUTPUT_FOLDER, filename))

def _align_zero(axes, ref=0, draw_zero_line=False):
    """
    Align y=0 across axes by shifting y-limits so that data y=0 maps to the same
    display (pixel) y-coordinate in each axes.

    Parameters
    ----------
    axes : iterable of Axes
        The axes to align.
    ref : int or Axes, optional
        Index of the reference axis (or the Axes object itself). Default: 0.
    draw_zero_line : bool, optional
        If True, draw axhline(0) on each axis after aligning.
    """
    axes = list(axes)
    # allow passing an Axes object as ref
    if hasattr(ref, "get_position"):
        ref_ax = ref
    else:
        ref_ax = axes[int(ref)]

    fig = ref_ax.figure

    # Ensure transforms/layout are finalized
    fig.canvas.draw()

    # display y coordinate (pixels) of data y==0 on reference axis
    ref_disp_y = ref_ax.transData.transform((0, 0))[1]

    for ax in axes:
        # find which data-y currently maps to that display y on this axis
        y_at_ref_disp = ax.transData.inverted().transform((0, ref_disp_y))[1]
        y0, y1 = ax.get_ylim()
        # shift limits so data y==0 maps to ref_disp_y
        ax.set_ylim(y0 - y_at_ref_disp, y1 - y_at_ref_disp)
        if draw_zero_line:
            ax.axhline(0, color="black", lw=1, zorder=0)


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

def make_usage_count_graph (exp_folders: list[str]) -> None:
    usage_counts = []
    thm_counts = []
    thm_count_by_it_gen = []
    exp_names = []
    for exp_folder in exp_folders:
        exp_name = os.path.basename(os.path.dirname(exp_folder))
        pre = lambda x : os.path.join(exp_folder, x)
        it = 0
        usage_counts_local = []
        thm_counts_local = []
        while True:
            if not os.path.exists(pre(f"generated_theorems_{it}.json")):
                break
            with open (pre(f"generated_theorems_{it}.json")) as f:
                theorems = UsefulConjectureList.validate_python(json.load(f))
            usage_counts_local.append(sum([thm.freq_used for thm in theorems]))
            thm_counts_local.append (len([thm for thm in theorems if thm.freq_used > 0]))
            it += 1
        thm_count_by_it_gen_local = [0 for i in range(it)]
        with open (pre(f"generated_theorems_{it-1}.json")) as f:
            theorems = UsefulConjectureList.validate_python(json.load(f))
            for thm in theorems:
                if thm.freq_used > 0:
                    thm_count_by_it_gen_local[thm.iter_generated] += 1
        
        thm_count_by_it_gen.append(thm_count_by_it_gen_local)
        usage_counts.append(usage_counts_local[:-1])
        thm_counts.append(thm_counts_local[:-1])
        exp_names.append(exp_name)
    _make_graph("Cummulative Theorem usage count / it", "Count", exp_names, usage_counts, "usage_counts.png")
    _make_graph("Cummulative # of used theorems / it", "", exp_names, thm_counts, "thm_counts.png")
    _make_graph("# of generated theorems / it", "", exp_names, thm_count_by_it_gen, "thm_gen.png")
        
def make_domains_graph (exp_folders: list[list[str]], names: list[str]) -> None:
    external_usefulness = []
    internal_usefulness = []
    external_error_bar = []
    internal_error_bar = []
    domain = []
    for exp in exp_folders:
        run_internal_vals = [[0] for run in exp]
        run_external_vals = [[0] for run in exp]

        with open(os.path.join(exp[0], "flags.json")) as f:
            data = json.load(f)
            domain.append(data["theory"]["name"])
        
        for run_it, run in enumerate(exp):
            pre = lambda x: os.path.join(run, x)
            try:
                with open(pre("useful_theorem_dedup.json")) as f:
                    useful_theorems = [LLMUsefulnessEvalTheorem.model_validate(thm) for thm in json.load(f)]
            except:
                with open(pre("useful_theorem_dedup.json")) as f:
                    data_str = f.read()
                    list_of_json_strings = ast.literal_eval(data_str)
                    data = [json.loads(item) for item in list_of_json_strings]
                    useful_theorems = [LLMUsefulnessEvalTheorem.model_validate(thm) for thm in data]

            for it in range(10):
                totals = []
                for usefulness_iteration in range(len(useful_theorems[0].explanations)):
                    locally_useful_theorems = [ut for ut in useful_theorems if ut.dedup_useful_at_k[usefulness_iteration] and ut.iteration <= it]
                    totals.append(len(locally_useful_theorems))
                # print(os.path.basename(run), totals, np.average(totals))
                run_external_vals[run_it].append(np.average(totals))

                # Internal usefulness
                if not os.path.exists(pre(f"usefulness_outcomes_{it}.json")):
                    run_internal_vals[run_it].append(0)
                    continue
                with open(pre(f"usefulness_outcomes_{it}.json")) as f:
                    outcomes = UsefulnessOutcomeList.validate_python(json.load(f))
                total_use = 0
                for outcome in outcomes:
                    if outcome.proof:
                        for line in outcome.proof:
                            if "by c" in line or "apply c" in line:
                                total_use += 1
                run_internal_vals[run_it].append(total_use)
        external_usefulness.append(np.average(run_external_vals, axis=0))
        internal_usefulness.append(np.average(run_internal_vals, axis=0))
        external_error_bar.append(np.sqrt(np.var(run_external_vals, axis=0)))
        internal_error_bar.append(np.sqrt(np.var(run_internal_vals, axis=0)))

    with open(os.path.join(OUTPUT_FOLDER, "domains_graph.csv"), "w") as f:
        f.write("domain,experiment_name,iteration,internal_usefulness,external_usefulness,internal_error_bar,external_error_bar\n")
        for i, exp in enumerate(names):
            assert len(internal_usefulness[i]) == len(external_usefulness[i])
            for it in range(len(internal_usefulness[i])):
                f.write(f"{domain[i]},{exp},{it},{internal_usefulness[i][it]},{external_usefulness[i][it]},{internal_error_bar[i][it]},{external_error_bar[i][it]}\n")
    
    colors = ["r", "g", "b", "y", "k", "c"]
    color_map = {name:color for name,color in zip(sorted(list(set(names))), colors)}
    handles = [
        Line2D([0], [0], color=color_map[name], linestyle='-', label=name) for name in set(names)
    ]
    handles2 = [
        Line2D([0], [0], color=color_map[name], linestyle='--', label=name) for name in set(names)
    ]
    name_dict = {
        "nat-mul": "Arithmetic",
        "propositional-logic": "Propositional Logic",
        "groups": "Group Theory"
    }
    labels = [name for name in sorted(list(set(names)))]


    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4), sharey=False)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=True, edgecolor='lightgray', facecolor="whitesmoke", fontsize=12)

    for i,curr_domain in enumerate(("nat-mul", "propositional-logic", "groups")):
        zipped = list(zip(*[_ for _ in zip(internal_usefulness, internal_error_bar, names, domain) if _[-1] == curr_domain]))
        title = "Intrinsically Useful Conjectures" if curr_domain == "nat-mul" else ""
        styles = [color_map[z] for z in zipped[2]]
        _make_graph (name_dict[curr_domain], title, None, list(zipped[0]), "internal_domains_graph.png", error_bars=list(zipped[1]), styles=styles, ax=axes[i],legend=False, fontsize=12)
    
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.15)
    for ax in axes:
        ax.set_ylim(bottom=-5)
    _align_zero(axes)
    plt.savefig(os.path.join(OUTPUT_FOLDER,"internal_domains_graph.png"))



    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4), sharey=False)
    fig.legend(handles2, labels, loc="upper center", ncol=len(labels), frameon=True, edgecolor='lightgray', facecolor="whitesmoke", fontsize=12)

    for i,curr_domain in enumerate(("nat-mul", "propositional-logic", "groups")):
        zipped = list(zip(*[_ for _ in zip(external_usefulness, external_error_bar, names, domain) if _[-1] == curr_domain]))
        title = "Extrinsically Useful Conjectures" if curr_domain == "nat-mul" else ""
        styles = [color_map[z] + "--" for z in zipped[2]]
        _make_graph (name_dict[curr_domain], title, None, list(zipped[0]), "external_domains_graph.png", error_bars=list(zipped[1]), styles=styles, ax=axes[i],legend=False, fontsize=12)

    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.15)
    _align_zero(axes)
    plt.savefig(os.path.join(OUTPUT_FOLDER,"external_domains_graph.png"))

def make_model_comparison_graph (exp_folders: list[str], names: list[str]) -> None:
    external_usefulness = []
    internal_usefulness = []
    for exp in exp_folders:
        external_usefulness.append([0])
        internal_usefulness.append([0])
        pre = lambda x: os.path.join(exp, x)
        with open(pre("useful_theorem_dedup.json")) as f:
            useful_theorems = [LLMUsefulnessEvalTheorem.model_validate(thm) for thm in json.load(f)]

        for it in range(10):
            totals = []
            for usefulness_iteration in range(len(useful_theorems[0].explanations)):
                locally_useful_theorems = [ut for ut in useful_theorems if ut.dedup_useful_at_k[usefulness_iteration] and ut.iteration <= it]
                totals.append(len(locally_useful_theorems))
            external_usefulness[-1].append(np.average(totals))

            # Internal usefulness
            if not os.path.exists(pre(f"usefulness_outcomes_{it}.json")):
                internal_usefulness[-1].append(0)
                continue
            with open(pre(f"usefulness_outcomes_{it}.json")) as f:
                outcomes = UsefulnessOutcomeList.validate_python(json.load(f))
            total_use = 0
            for outcome in outcomes:
                if outcome.proof:
                    for line in outcome.proof:
                        if "by c" in line or "apply c" in line:
                            total_use += 1
            internal_usefulness[-1].append(total_use)
    with open(os.path.join(OUTPUT_FOLDER, "model_comparison_graph.csv"), "w") as f:
        f.write("experiment_name,iteration,internal_usefulness,external_usefulness\n")
        for i, exp in enumerate(names):
            assert len(internal_usefulness[i]) == len(external_usefulness[i])
            for it in range(len(internal_usefulness[i])):
                f.write(f"{exp},{it},{internal_usefulness[i][it]},{external_usefulness[i][it]}\n")
    
    external_values = external_usefulness
    internal_values = internal_usefulness
    colors = ["g", "r", "b", "k", "c"]
    styles = [colors[i] for i in range(len(exp_folders))]
    styles2 = [colors[i] + "--" for i in range(len(exp_folders))]
    fig = plt.figure(figsize=(14, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.3], wspace=0.3)  # last column narrower for legend
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
    ]

    _make_graph("Internal Usefulness", "Intrinsically Useful Conjectures", names, internal_values, "model_comparison.png", styles, ax=axes[0], legend=False, fontsize=12)
    _make_graph("External Usefulness", "Extrinsically Useful Conjectures", names, external_values, "model_comparison.png", styles2, ax=axes[1], legend=False, fontsize=12)
    handles = [
        Line2D([0], [0], color=color, linestyle='-', label=name) for color,name in zip(colors, names)
    ]
    fig.subplots_adjust(left=0.05, bottom=0.15)
    _align_zero(axes[:1])
    axes[2].axis("off")
    axes[2].legend(handles, names, loc="center", frameon=True, edgecolor='lightgray', facecolor="whitesmoke", fontsize=12)
    plt.savefig(os.path.join(OUTPUT_FOLDER, "model_comparison.png"))



def draw_usefulness_usage_graph (input_filepath: str, output_folder: str) -> None:
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
    # exps = [
        # "/home/timothekasriel/minimo/learning/outputs/line19/outcomes_9.json",
        # "/home/timothekasriel/minimo/learning/outputs/line18/outcomes_9.json",
        # "/home/timothekasriel/minimo/learning/outputs/line15/outcomes_9.json",
        # "/home/timothekasriel/minimo/learning/outputs/line10/outcomes_9.json"
        # "/home/timothekasriel/minimo/learning/outputs/line25/",
        # "/home/timothekasriel/minimo/learning/outputs/line30",
        # "/home/timothekasriel/minimo/learning/outputs/line33",
        # "/home/timothekasriel/minimo/learning/outputs/line33_prop",
        # "/home/timothekasriel/minimo/learning/outputs/line33_group",
        # "/home/timothekasriel/minimo/learning/outputs/line29/outcomes_9.json",
        # ]
    # make_usage_count_graph(exps)
    exps = [
        # Nat-mul
        [
            "/home/timothekasriel/minimo/learning/outputs/line33",
            "/home/timothekasriel/minimo/learning/outputs/line33",
            "/home/timothekasriel/minimo/learning/outputs/line33_3",
        ],
        [
            "/home/timothekasriel/minimo_org/learning/outputs/base",
            "/home/timothekasriel/minimo_org/learning/outputs/base_2",
            "/home/timothekasriel/minimo_org/learning/outputs/base_3",
        ],
        # Prop-logic
        [
            "/home/timothekasriel/minimo/learning/outputs/line33_prop",
            "/home/timothekasriel/minimo/learning/outputs/line33_prop_2",
            "/home/timothekasriel/minimo/learning/outputs/line33_prop_3",
        ],
        [
            "/home/timothekasriel/minimo_org/learning/outputs/base_prop",
            "/home/timothekasriel/minimo_org/learning/outputs/base_prop",
            "/home/timothekasriel/minimo_org/learning/outputs/base_prop",
        ],
        # Group theory
        [
            "/home/timothekasriel/minimo/learning/outputs/line33_group",
            "/home/timothekasriel/minimo/learning/outputs/line33_group_2",
            "/home/timothekasriel/minimo/learning/outputs/line33_group_2",
        ],
        [
            "/home/timothekasriel/minimo_org/learning/outputs/base_group",
            # "/home/timothekasriel/minimo_org/learning/outputs/base_group",
            # "/home/timothekasriel/minimo_org/learning/outputs/base_group",
        ],
    ]
    make_domains_graph(exps, ["Our method", "Base minimo"] * 3)

    exps = [
        "/home/timothekasriel/minimo/learning/outputs/line33",
        "/home/timothekasriel/minimo/learning/outputs/line33.8",
        "/home/timothekasriel/minimo/learning/outputs/line33.9",
        "/home/timothekasriel/minimo/learning/outputs/line33.10",
    ]
    make_model_comparison_graph(exps, ["Our model", "Only training conjecturer", "Only training prover", "No extra training"])
    # make_success_rate_graph(exps)
    # make_variable_use_count_graph(exps)
    # make_logprob_graph(exps)



