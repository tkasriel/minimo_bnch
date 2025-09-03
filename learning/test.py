import json
import os

from matplotlib import pyplot as plt
from conjecture import Context, App
from peano import PyDerivation
import re

from llm_eval import get_reproof_values

def test_app_parse():
    def has_trivial_outcome(conjecture):
        # Find the statement after the last "->"
        parts = conjecture.split("->")
        if len(parts) <= 1:
            return True  # Not enough arrows, probably incomplete
            
        # check if it only involves constant values and operators
        last_statement = parts[-1].strip()
        if ']' in last_statement:
            last_statement = last_statement[:last_statement.index(']')].strip()
        print(f"Last statement: {last_statement}")
        
        tokens = set([ch for ch in last_statement])
        constant_tokens = set(['z', ' ', '+', '*', 'o', '(', ')', '=', 's', ']', '.'])
        non_trivial_tokens = tokens - constant_tokens
        print(non_trivial_tokens)
        if not non_trivial_tokens:
            return True  # only involves trivial tokens
        
        # matches trivial arithmetic identities (modulo symmetry)
        trvial_identities = [
            r"\(= +\(\* +'[a-zA-Z0-9_]+ +z\) +z\)",   # (* n z) = z
            r"\(= +z +\(\* +'[a-zA-Z0-9_]+ +z\)\)",   # z = (* n z)
            r"\(= +\(\+ +'[a-zA-Z0-9_]+ +z\) +'[a-zA-Z0-9_]+\)",  # (+ n z) = n
            r"\(= +'[a-zA-Z0-9_]+ +\(\+ +'[a-zA-Z0-9_]+ +z\)\)",  # n = (+ n z)
            r"\(= +\(\* +o +'[a-zA-Z0-9_]+\) +'[a-zA-Z0-9_]+\)",  # (* o n) = n
            r"\(= +'[a-zA-Z0-9_]+ +\(\* +o +'[a-zA-Z0-9_]+\)\)",  # n = (* o n)
            r"\(= +'[a-zA-Z0-9_]+ +'[a-zA-Z0-9_]+\)",  # (= 'a0 'a0)
        ]
        for pattern in known_identities:
            if re.fullmatch(pattern, last_statement):
                return True
            
        # tautology — conclusion appeared earlier
        prior_string = "->".join(parts[:-1])
        if last_statement in prior_string:
            return True
            
        return False
    
    trivial_patterns = [
            "[('a0 : nat) -> ('a1 : nat) -> ('a2 : (= o 'a0)) -> ('a3 : (= 'a0 o)) -> ('a4 : nat) -> (= o 'a0)]",  # one equals one
            "[('a0 : (= z o)) -> ('a1 : nat) -> ('a2 : (= o o)) -> (= z o)]",
            "[('a0 : nat) -> (= (* (+ (+ (+ z z) o) z) z) z)]",
            "[('a0 : nat) -> ('a1 : (= 'a0 'a0)) -> ('a2 : nat) -> (= 'a0 'a0)]",
            "[('a0 : nat) -> (= z (* 'a0 z))]"
        ]
    
    for conjecture in trivial_patterns:
        if has_trivial_outcome(conjecture):
            print(f"Conjecture '{conjecture}' has a trivial outcome.")
        else:
            print(f"Conjecture '{conjecture}' does not have a trivial outcome.")

def test_reproof_improvement (old_outcomes: str, new_outcomes: str) -> list[int] :
    key = lambda x: x["problem"]
    with open(old_outcomes) as f:
        old_o = sorted([o for o in json.load(f) if not o["hindsight"]], key=key)
    with open(new_outcomes) as f:
        new_o = sorted([o for o in json.load(f) if not o["hindsight"]], key=key)
    old_proof_count = len([o for o in old_o if o["proof"]])
    new_proof_count = len([o for o in new_o if o["proof"]])
    
    improved = 0
    degraded = 0
    improvement_by_it = [0 for i in range(10)]

    for _ in zip(old_o, new_o):
        old, new = _
        assert old["problem"] == new["problem"]
        if old["proof"] and not new["proof"]:
            degraded += 1
            improvement_by_it[int(old["iteration"])] -= 1
        if new["proof"] and not old["proof"]:
            improved += 1
            improvement_by_it[int(old["iteration"])] += 1
    print (f"""RESULTS:
Original proof count: {old_proof_count}
New proof count: {new_proof_count}
Number of newly proven theorems: {improved}
Number of degraded theorems (unproven by 'stronger' prover) {degraded}""")
    return improvement_by_it

def graph_improvements (exp_names: list[str], improvements_by_it: list[list[int]]):
    from graphing import _make_graph
    _make_graph("Reproof improvement per iteration", "improvement", exp_names, improvements_by_it, "reproof_succ.png")
    



if __name__ == "__main__":
    exp_folder = "/home/timothekasriel/minimo/learning/outputs/"
    exps = os.listdir(exp_folder)
    exp_outs = []
    exp_names = []
    for exp in exps:
        if (os.path.exists(os.path.join(exp_folder, exp, "final_outcomes.json")) and
           os.path.exists(os.path.join(exp_folder, exp, "useful_theorem_dedup_proven.txt"))):
            print(f"Experiment: {exp}")
            get_reproof_values(os.path.join(exp_folder, exp))
    #         try:
    #             exp_outs.append(get_reproof_values(exp_folder))
    #             exp_outs.append(test_reproof_improvement(
    #                 os.path.join(exp_folder, exp, "outcomes_9.json"),
    #                 os.path.join(exp_folder, exp, "final_outcomes.json")
    #             ))
    #             exp_names.append(exp)
    #         except AssertionError:
    #             print (exp)
    # graph_improvements(exp_names, exp_outs)
    # test_app_parse()


# (a0: nat) -> (a1: nat) -> (a2: (= o a0)) -> ... -> (a3: (= a0 o)) -> (a4: nat) -> (= z z)
# log-prob low : model think it's good and it's very very hard