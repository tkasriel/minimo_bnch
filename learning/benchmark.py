import itertools
import re
import time
import json
import peano
from tqdm import tqdm
from util import format_blocks_with_indent
from proofsearch import HolophrasmNode, MonteCarloTreeSearch, TreeSearchNode, UniformPolicy


input_file = "learning/outputs/outcomes_4.json"
theory_file = "learning/theories/nat-mul.p"



ths_to_prove = [
    "[(x : nat) -> (= z (* z x))]",
    "[('a : nat) -> (= (+ 'a (s z)) (s 'a))]",
    "[('m : nat) -> (= (* z 'm) z)]",
    "[('x : nat) -> ('y : nat) -> ('z : nat) -> (= (+ (* x 'y) z) (+ (* x 'y) z))]",
    # "[('x : nat) -> ('y : nat) -> (= 'y (+ 'x n7)) -> (= (* n2 'y) (* n2 (+ 'x n7)))]",
    "[('a : nat) -> ('b : nat) -> (= (s 'a) 'b) -> (= (s (s 'a)) (s 'b))]",
    "[('n : nat) -> (= (+ z 'n) 'n)]",
    "[('a : nat) -> ('b : nat) -> ('c : nat) -> (= (+ (+ 'a 'b) 'c) (+ 'a (+ 'b 'c)))]",
    "[('a : nat) -> ('b : nat) -> (= (+ (s 'a) 'b) (s (+ 'a 'b)))]",
    "[('n : nat) -> (= (s 'n) (+ 'n (s z)))]",
    "[('m : nat) -> (= (* 'm (s z)) 'm)]"
]
th_to_prove = "[(m' : nat) -> (n' : nat) -> (= (+ (s n') m') (s (+ n' m')))]"
mcts = MonteCarloTreeSearch(UniformPolicy({}), 30)
premises = ['t', 'eq_symm', 'eq_refl']#, 'rewrite'] #eq_refl', 'eq_symm', 'rewrite', '+_z', '+_s', '*_s', '*_z', 'nat_ind', 't']

with open(theory_file, "r") as file:
    theory = file.readlines()
with open(input_file, "r") as file:
    conjs = json.load(file)

def full_proof_method (conjs, theory, premises, ths_to_prove):

    for i, element in enumerate(tqdm(list(itertools.product(conjs, ths_to_prove)))):
        proof = element[0]["proof"]
        if not proof:
            continue
        proof = [
            """theorem t: [(x : nat) -> (= (* z x) z)] {
                intro x0 : nat.
                show (= z z) by *_z.
            }"""
        ]

        # print("Current proof: " + "\n".join(proof))
        # print("Theorem to prove: " + element[1])
        for j in ths_to_prove:
            initial_state = peano.PyProofState("\n".join(theory) + "\n".join(proof),
                                    premises,
                                    j) # noqa
            root = TreeSearchNode(HolophrasmNode([initial_state])) # noqa
            solved, pi, _, it = mcts.evaluate(root, verbose=False) # noqa
            if solved:
                print("Current proof: " + "\n".join(proof))
                print(j, "\nPROVEN!")
                print(format_blocks_with_indent(root.reconstruct_proof()))
            else:
                print (j, "\n unproven")
        return


def instant_proof_method(conjs, theory, premises, ths_to_prove):
    for i, element in enumerate(tqdm(conjs)):
        proof = element["proof"]
        if not proof:
            continue
        proof = [
            """theorem t: [(x : nat) -> (= z (* z x))] {
                intro x0 : nat.
                show (= z z) by *_z.
            }"""
        ]
        # vars = re.findall(r"\[\((.*) : nat", proof[0])
        # print(vars)
        for theorem in ths_to_prove:
            var_proof = "\n".join(["intro x" + str(i) + " : nat." for i in range(len(vars))])
            proof_from_conj = f"""theorem q: {theorem} {{
                apply t.
            }}
            """
            full_theory = "\n".join(theory) + "\n".join(proof) + "\n" + proof_from_conj + "\nverify q {}"
            # print(full_theory)
            initial_state = peano.PyProofState(full_theory,
                                        premises,
                                        "(= z z)") # noqa
            if initial_state:
                print(f"{theorem} proven!")
        return
full_proof_method(conjs,theory, premises, ths_to_prove)
        