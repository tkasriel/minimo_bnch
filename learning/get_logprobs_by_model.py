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

def get_logprob(cfg, agent, theory, statement, solution_actions):
    state = peano.PyProofState(theory.theory,
                               theory.premises,
                               statement)
    
    node_type = ({'vanilla': LeftmostFirstSearchNode,
                            'holophrasm': HolophrasmNode})[cfg.get('node_type', 'holophrasm')]
    root = TreeSearchNode(node_type([state]))

    return root.solution_logprob_under_policy(cfg, agent._policy, solution_actions)

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

## This test is without any additional conjectures in the theory and should pass
# eps = 0.01
# print("Testing to see if calculated logprobs match with reference logprobs in outcomes.json...")
# for useful_theorem_outcome in usefulness_outcomes:
#     problem_statement = extract_peano_statement_from_proof(useful_theorem_outcome["proof"])

#     baseline_problem = None
#     for regular_outcome in outcomes:
#         if regular_outcome["problem"] == problem_statement:
#             baseline_problem = regular_outcome
#             break
    
#     assert baseline_problem, f"Could not find problem {problem_statement} in outcomes.json file"

#     used_theorems = useful_theorem_outcome["used_theorems"]
#     new_theory = theory + "\n\n" + "\n\n".join(used_theorems)
#     new_premises = premises + [thm.split(" : ")[0] for thm in used_theorems]

#     ## Test proof without using theorems
#     worker_theory = worker.BackgroundTheory(theory, premises)

#     statement = baseline_problem["problem"]
#     solution_actions = baseline_problem["actions"]

#     calculated_logprob = get_logprob(cfg, agent, worker_theory, statement, solution_actions)
#     actual_logprob = baseline_problem['logprob']

#     assert np.abs(calculated_logprob - actual_logprob) < eps, f"Non-usefulness: calculated logprob of {calculated_logprob} diferred from {actual_logprob} for problem {statement}"

# print("Tests passed")

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

# Calculate improvement for usefulness (results do not match)
# deltas = []
# for useful_theorem_outcome in usefulness_outcomes:
#     problem_statement = extract_peano_statement_from_proof(useful_theorem_outcome["proof"])

#     baseline_problem = None
#     for regular_outcome in outcomes:
#         if regular_outcome["problem"] == problem_statement:
#             baseline_problem = regular_outcome
#             break
    
#     assert baseline_problem, f"Could not find problem {problem_statement} in outcomes.json file"

#     used_theorems = useful_theorem_outcome["used_theorems"]
#     new_theory = theory + "\n\n" + "\n\n".join(used_theorems)
#     new_premises = premises + [thm.split(" : ")[0] for thm in used_theorems]

#     # ## Test proof with theorems
#     worker_theory = worker.BackgroundTheory(new_theory, new_premises)

#     ref_logprob = useful_theorem_outcome['logprob']
#     solution_actions = convert_proof_to_actions(useful_theorem_outcome["proof"])

#     calculated_logprob = get_logprob(cfg, agent, worker_theory, problem_statement, solution_actions)

#     # print(statement)
#     # print(f"Calculated logprob:{calculated_logprob}, reference improvement: {ref_logprob}")

#     print("---------------------------------------------------------------------------------------")
#     print(f"Original logprob: {baseline_problem['logprob']}")
#     print(f"Usefulness logprob (ref): {useful_theorem_outcome['logprob']}")
#     print(f"Usefulness logprob (calc): {calculated_logprob}")
#     print(f"Usefulness improvements: {useful_theorem_outcome['improvement']}")
#     print(f"Disparity: {np.abs(calculated_logprob - baseline_problem['logprob'] - useful_theorem_outcome['improvement'])}")

#     deltas.append(np.abs(calculated_logprob - baseline_problem['logprob'] + useful_theorem_outcome['improvement']))

# print(f"Average absolute difference between calculated and ref improvement for used theorems: {np.mean(deltas)}")



# for useful_theorem_outcome in usefulness_outcomes:
#     problem_statement = extract_peano_statement_from_proof(useful_theorem_outcome["proof"])

#     baseline_problem = None
#     for regular_outcome in outcomes:
#         if regular_outcome["problem"] == problem_statement:
#             baseline_problem = regular_outcome
#             break
    
#     assert baseline_problem, f"Could not find problem {problem_statement} in outcomes.json file"


#     used_theorems = useful_theorem_outcome["used_theorems"]
#     new_theory = theory + "\n\n" + "\n\n".join(used_theorems)
#     new_premises = premises + [thm.split(" : ")[0] for thm in used_theorems]

#     # ## Test proof with theorems
#     worker_theory = worker.BackgroundTheory(new_theory, new_premises)

#     baseline_logprob = useful_theorem_outcome['logprob']
#     statement = problem_statement

#     solution_actions = convert_proof_to_actions(useful_theorem_outcome["proof"])

#     calculated_logprob = get_logprob(cfg, agent, worker_theory, statement, solution_actions)

#     print(statement)
#     print(f"Calculated logprob:{calculated_logprob}, reference improvement: {useful_theorem_outcome['logprob']}")

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
