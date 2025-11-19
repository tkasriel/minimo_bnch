import os
import sys
import worker
import peano
import torch
import json
from omegaconf import OmegaConf
from proofsearch import TreeSearchNode, LeftmostFirstSearchNode, HolophrasmNode, ProofSearchAgent
import numpy as np
import celery
import re
from tqdm import tqdm

from extract_actions_prop_logic import convert_proof_to_actions_prop

def get_logprob(cfg, agent, theory, statement, solution_actions):
    state = peano.PyProofState(theory.theory,
                               theory.premises,
                               statement)
    
    node_type = ({'vanilla': LeftmostFirstSearchNode,
                            'holophrasm': HolophrasmNode})[cfg.get('node_type', 'holophrasm')]
    root = TreeSearchNode(node_type([state]))

    try:
        solution_logprob = root.solution_logprob_under_policy(cfg, agent._policy, solution_actions)
    except Exception:
        print(f"Failed at {statement}")

        return 1
    return solution_logprob
# NOTE FROM TL: I only tested the following helpers for nat-mul
def convert_proof_to_actions(proof_lines: list[str]):
    actions = []
    show_re = re.compile(r"""^\s*show\s+(?P<goal>.+?)\s+by\s+(?P<tactic>[^.]+)\.\s*$""")

    for raw in proof_lines:
        line = raw.strip().rstrip(',')

        # Skip headers/footers
        if not line or line.startswith("theorem ") or line == "}":
            continue

        # intro.
        if line.startswith("intro ") and line.endswith("."):
            actions.append("intro.")
            continue

        # apply
        if line.startswith("apply ") and line.endswith("."):
            tactic = line[len("apply "):-1].strip()
            if not tactic:
                raise ValueError(f"Empty apply tactic: {line}")
            actions.append(f"a {tactic}")
            actions.append("=> .")
            continue

        # show … by …
        m = show_re.match(line)
        if m:
            goal = m.group("goal").strip()
            tactic = m.group("tactic").strip()
            actions.append(f"c {tactic}")
            goal_with_period = goal if goal.endswith(".") else f"{goal}."
            actions.append(f"=> {goal_with_period}")
            continue

        # If a line doesn't match any known pattern, raise error
        raise ValueError(f"Unrecognized proof line: {line}")

    return actions

def extract_peano_statement_from_proof(proof_lines: list[str]):
    for i, raw in enumerate(proof_lines):
        s = (raw or "").strip().rstrip(',')
        if s.startswith("theorem "):
            tail = s.split(":", 1)[1].strip()
            parts, bal, j = [tail], tail.count('[') - tail.count(']'), i + 1
            while bal > 0 and j < len(proof_lines):
                t = (proof_lines[j] or "").strip().rstrip(',')
                parts.append(t)
                bal += t.count('[') - t.count(']')
                j += 1
            txt = " ".join(parts)
            if ']' in txt:
                txt = txt[:txt.rfind(']') + 1]
            return txt.split('{', 1)[0].strip()
    raise ValueError("No theorem header found.")

def compute_improvement(cfg, useful_theorem_outcome, baseline_outcomes, agent, base_theory, base_premises):
    problem_statement = extract_peano_statement_from_proof(useful_theorem_outcome["proof"])

    baseline_problem = None
    for regular_outcome in baseline_outcomes:
        if regular_outcome["problem"] == problem_statement:
            baseline_problem = regular_outcome
            break
    
    assert baseline_problem, f"Could not find problem {problem_statement} in outcomes.json file"

    # test proof without theorems
    base_worker_theory = worker.BackgroundTheory(base_theory, base_premises)
    base_actions = baseline_problem["actions"]
    base_logprob = get_logprob(cfg, agent, base_worker_theory, problem_statement, base_actions)

    # build augmented theory
    used_theorems = useful_theorem_outcome["used_theorems"]
    new_theory = base_theory + "\n\n" + "\n\n".join(used_theorems)
    new_premises = base_premises + [thm.split(" : ")[0] for thm in used_theorems]

    # ## Test proof with theorems
    usefulness_worker_theory = worker.BackgroundTheory(new_theory, new_premises)
    usefulness_actions = convert_proof_to_actions(useful_theorem_outcome["proof"])
    usefulness_logprob = get_logprob(cfg, agent, usefulness_worker_theory, problem_statement, usefulness_actions)

    improvement = usefulness_logprob - base_logprob

    return {"original_logprob": base_logprob, "usefulness_logprob": usefulness_logprob, "improvement": improvement}    

def compute_logprobs_usefulness(path, model_it, theorems_it, single_theorem_it_only = True, used_theorem_only = True):
    with open(os.path.join(path, "flags.json"), "r") as f:
        cfg_dict = json.load(f)
        cfg = OmegaConf.create(cfg_dict)

    agent: ProofSearchAgent = torch.load(os.path.join(path, f'{model_it}.pt'), weights_only=False)
    agent._policy._lm.eval()

    with open(os.path.join(os.path.dirname(__file__), "theories", cfg.theory.name + '.p')) as f:
        theory = f.read()

    with open(os.path.join(path, f"outcomes_{theorems_it}.json"), "r", encoding = "utf-8") as f:
        outcomes = json.load(f)

    with open(os.path.join(path, f"usefulness_outcomes_{theorems_it}.json"), "r", encoding = "utf-8") as f:
        usefulness_outcomes = json.load(f)
        
    if(single_theorem_it_only):
        outcomes = [i for i in outcomes if (not i["hindsight"] and i["logprob"] and i["iteration"] <= theorems_it)]
        usefulness_outcomes = [i for i in usefulness_outcomes if i["iteration"] == theorems_it]
    else:
        outcomes = [i for i in outcomes if (not i["hindsight"] and i["logprob"])]
        usefulness_outcomes = [i for i in usefulness_outcomes if True]

    premises = cfg.theory.premises

    results = []
    for usefulness_outcome in usefulness_outcomes:
        if(not "by c" in str(usefulness_outcome["proof"])):
            continue
        out = compute_improvement(cfg, usefulness_outcome, outcomes, agent, theory, premises)

        results.append(out)

    return results

def compute_logprobs_usefulness_revision(
    model_path,
    data_path,
    model_it,
    theorems_it,
    single_theorem_it_only: bool = True,
    used_theorem_only: bool = True,
):
    # Config + theory come from the data_path
    with open(os.path.join(data_path, "flags.json"), "r") as f:
        cfg_dict = json.load(f)
        cfg = OmegaConf.create(cfg_dict)

    # Model comes from the model_path
    agent: ProofSearchAgent = torch.load(
        os.path.join(model_path, f"{model_it}.pt"),
        weights_only=False
    )
    agent._policy._lm.eval()

    # Theory file is still resolved from this source file's directory
    with open(
        os.path.join(os.path.dirname(__file__), "theories", cfg.theory.name + ".p")
    ) as f:
        theory = f.read()

    # Outcomes & usefulness outcomes come from the data_path
    with open(
        os.path.join(data_path, f"outcomes_{theorems_it}.json"),
        "r",
        encoding="utf-8",
    ) as f:
        outcomes = json.load(f)

    with open(
        os.path.join(data_path, f"usefulness_outcomes_{theorems_it}.json"),
        "r",
        encoding="utf-8",
    ) as f:
        usefulness_outcomes = json.load(f)

    if single_theorem_it_only:
        outcomes = [
            i
            for i in outcomes
            if (not i["hindsight"] and i["logprob"] and i["iteration"] <= theorems_it)
        ]
        usefulness_outcomes = [
            i for i in usefulness_outcomes if i["iteration"] == theorems_it
        ]
    else:
        outcomes = [
            i for i in outcomes if (not i["hindsight"] and i["logprob"])
        ]
        usefulness_outcomes = [i for i in usefulness_outcomes]

    premises = cfg.theory.premises

    results = []
    for usefulness_outcome in usefulness_outcomes:
        if "by c" not in str(usefulness_outcome["proof"]):
            continue

        out = compute_improvement(
            cfg,
            usefulness_outcome,
            outcomes,
            agent,
            theory,
            premises,
        )
        results.append(out)

    return results


def test_logprob_on_ref():
    path = "outputs\\new_test"
    iteration = 14

    with open(os.path.join(path, "flags.json"), "r") as f:
        cfg_dict = json.load(f)
        cfg = OmegaConf.create(cfg_dict)

    # cfg.support_theorem_use = False

    print(f"Using theory from config: {cfg.theory.name}")
    print(f"Support theorem use: {cfg.support_theorem_use}")

    agent: ProofSearchAgent = torch.load(os.path.join(path, f'{iteration}.pt'), weights_only=False)
    agent._policy._lm.eval()

    with open(os.path.join(os.path.dirname(__file__), "theories", cfg.theory.name + '.p')) as f:
        theory = f.read()

    with open(os.path.join(path, f"outcomes_{iteration}.json"), "r", encoding = "utf-8") as f:
        outcomes = json.load(f)

    with open(os.path.join(path, f"usefulness_outcomes_{iteration}.json"), "r", encoding = "utf-8") as f:
        usefulness_outcomes = json.load(f)

    outcomes = [i for i in outcomes if (not i["hindsight"] and i["logprob"] and i["iteration"] == iteration)]
    premises = cfg.theory.premises

    usefulness_outcomes = [i for i in usefulness_outcomes if i["iteration"] == iteration]

    eps = 0.01
    for usefulness_outcome in usefulness_outcomes:
        out = compute_improvement(cfg, usefulness_outcome, outcomes, agent, theory, premises)

        assert np.abs(out["usefulness_logprob"] - usefulness_outcome["logprob"]) < eps

    print(f"Test succesfully passed for {len(usefulness_outcomes)} theorems")

if __name__ == "__main__":
    test_logprob_on_ref()
