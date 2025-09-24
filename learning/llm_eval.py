import ast
import asyncio
import json
import os
import re
import sys
import time
from typing import Awaitable
import numpy as np
from openai import AsyncOpenAI
import dotenv
from pydantic import BaseModel
import tqdm
import tqdm.asyncio
from z3.z3 import ArithRef, Bool, Int, Implies, Or, Solver, solve, And, Not
import json

from classes import ProofOutcomeList, UsefulConjecture, UsefulConjectureList, UsefulnessOutcomeList, LLMUsefulnessEvalResult, LLMUsefulnessEvalTheorem
from convert_to_lean import _find_atoms, convert_peano_to_lean

if not dotenv.load_dotenv():
    raise ValueError("Need to set api key in .env")
print (f"OPENAI API KEY: {os.getenv('OPENAI_API_KEY')}")
client = AsyncOpenAI()
MODEL = "gpt-4.1"

        

def _extract_theorems_from_outcomes(outcomes_filepath: str, include_hindsight: bool = False, include_unproven: bool = False, theory_name: str = "") -> list[LLMUsefulnessEvalTheorem]:
    with open(outcomes_filepath) as f:
        outcomes = json.load(f)
    results = []
    for i, o in enumerate(outcomes):
        if o.get("hindsight", False) and not include_hindsight:
            continue
        if not o.get("proof", False) and not include_unproven:
            continue
        # if "problem_translated" not in o.keys():
            # Old non-translated problems, pre-dsprover
        o["problem_translated"] = convert_peano_to_lean(o["problem"], i, simplify=True, theory_string = theory_name)
        # if i == 2815:
        #     print(o)
        #     sys.exit(0)
        if o["logprob"] is None:
            o["logprob"] = 1.0
        thm_str = f"theorem problem{i} : {o['problem_translated']}"
        results.append(LLMUsefulnessEvalTheorem(thm_string=thm_str,
                                proven=bool(o["proof"]),
                                thm_org=o["problem"],
                                thm_string_simple=o["problem_translated"],
                                iteration=int(o.get("iteration", -1)),
                                logprob=float(o.get("logprob", 0.0))))
    return results

def _make_prompts (theorems: list[str], theory_name: str = "") -> list[list[dict[str, str]]]:
    if("nat" in theory_name):
        known_theorems = """Nat : type.
0 : Nat.
Nat.succ : [Nat -> Nat].
1 : Nat.
+ : [Nat -> Nat -> Nat].
* : [Nat -> Nat -> Nat].
o_s : Nat.succ 0 = 1.
+_z : (n : Nat) -> n + 0 = n.
+_s : (n : Nat) -> (m : Nat) -> n + (Nat.succ m) = Nat.succ (n + m).
*_z : (n : Nat) -> n * 0 = 0.
*_s : (n : Nat) -> (m : Nat) -> n * (Nat.succ m) = n + n * m.
nat_ind : (p : (Nat -> Prop)) -> (p z) -> ((n : Nat) -> (p n) -> p (Nat.succ n)) -> (n : Nat) -> (p n)."""
    elif("group" in theory_name):
        known_theorems = """
Group : type.
• : [Group -> Group -> Group].
1 : Group.

op_assoc : (v0 : Group) -> (v1 : Group) -> (v2 : Group) -> (((v0 • v1) • v2) = (v0 • (v1 • v2))).
op_comm : (v0 : Group) -> (v1 : Group) -> ((v0 • v1) = (v1 • v0)).
id_1 : (1 • v0) = v0).
inv_1 : (v0 : Group) -> (((v0⁻¹) • v0) = 1).
"""
    elif ("prop" in theory_name):
        known_theorems = """Prop : type.

false : Prop.
¬ : [Prop -> Prop].
∧ : [Prop -> Prop -> Prop].
∨ : [Prop -> Prop -> Prop].
↔ : [Prop -> Prop -> Prop].

and_i : ((v0 : Prop) → (v1 : Prop) → v0 → v1 → (v0 ∧ v1))
and_el : ((v0 : Prop) → (v1 : Prop) → (v0 ∧ v1) → v0)
and_er : ((v0 : Prop) → (v1 : Prop) → (v0 ∧ v1) → v1)
or_il : ((v0 : Prop) → (v1 : Prop) → v0 → (v0 ∨ v1))
or_ir : ((v0 : Prop) → (v1 : Prop) → v1 → (v0 ∨ v1))
not_i : ((v0 : Prop) → (v0 → false) → (¬ v0))
not_e : ((v0 : Prop) → (¬ v0) → v0 → false)
exfalso : (false → (v0 : Prop) → v0)
iff_i : ((v0 : Prop) → (v1 : Prop) → (v0 → v1) → (v1 → v0) → (v0 ↔ v1))
iff_el : ((v0 : Prop) → (v1 : Prop) → (v0 ↔ v1) → (v0 → v1))
iff_er : ((v0 : Prop) → (v1 : Prop) → (v0 ↔ v1) → (v1 → v0))
em : ((v0 : Prop) → (v0 ∨ (¬ v0)))
"""
    else:
        raise Exception("Unknown theory")
    prompt = f"""You are tasked to judge whether a given lean theorem could be considered useful for an automatic theorem prover to have among its known theorems.
This theorem prover has only access to the following axioms and known theorems:
```
{known_theorems}
```
As well as access to the `rfl` and `rewrite` commands
Here is the theorem you are to evaluate
```lean4
{{}}
```
Think through the problem step by step. Translate the problem into natural language, then think of what the possible uses of the theorem could be, whether it's obviously true and whether it means something.
On the last line, say either USEFUL or NOT USEFUL and nothing else.
"""
    if theory_name == 'groups':
        prompt += """Note: you can ignore irrelevant hypotheses. For example, the following theorems would not be considered useful.
theorem problem1 : ((v0 : Group) -> (v0 = (v0 • 1)))
theorem problem2 : ((v0 : Group) -> ((v0 • 1) = v0))
theorem problem3 : ((v0 : (1 = 1)) -> (1 = (1 • 1)))

As the first two are rewrites of axiom id_1
Problem 3 has an irrelevant hypothesis so we can ignore it, but more importantly has a straighforward application of id_1: By running id_1 1, we get this exact theorem, so it's not useful.
"""
    chats = [[
        {"role": "developer", "content": "You are an expert at evaluating lean theorems"},
        {"role": "user", "content": prompt.format(theorem)}
    ] for theorem in theorems]
    return chats


async def _remove_dedup (useful_theorems: list[LLMUsefulnessEvalTheorem], theory_name: str = "nat-mul") -> tuple[list[LLMUsefulnessEvalTheorem], tuple[str, str]]:
    if theory_name == "groups":
        prompt = """I have a set of lean theorems, some of which are very similar to each other. I want to use them as lemmas for proof generation. 
Please remove the duplicates, so that I can have a list of only unique theorems.
For example, the following four theorems would be duplicates of each other:
```lean4
theorem problem1 : ((v0 : Group) -> (v1 : (v0 = (v0 • (1⁻¹)))) -> ((1⁻¹) = 1))
theorem problem2 : ((v0 : Group) -> (v1 : Group) -> ((1⁻¹) = 1))
theorem problem3 : ((v0 : Group) -> ((1⁻¹) = 1))
theorem problem4 : ((v0 : Group) -> (1 = (1⁻¹))
```
Problem 1 introduces an irrelevant hypothesis as compared to problem 3, as it makes no mention of v0 in its final claim. Therefore, these two problems are duplicates of each other.
Problem 2 is a similar case to problem 1: It introduces an extra variable, but does nothing with it. This is irrelevant, and makes for the same problem.
Problem 4 is the same as problem 3, but is flipped. As we are running this using rw, we can simply call this problem in the inverse direction, so these two lemmas are the same.

In this case, our final result would likely be:
```lean4
theorem problem3 : ((v0 : Group) -> ((1⁻¹) = 1))
```

Here is my list of theorems for you to remove duplicates for. 
{}
I also have attached an explanation for why each could be useful for a theorem prover.
{}
Think it through step by step, and then return the list of unique theorems from this list in a list format inside of a ```lean4``` code block. Make sure your answer is inside the very last lean codeblock. Please make sure to repeat the theorems exactly as I wrote them.
"""
    else:
        prompt = """I have a set of lean theorems, some of which are very similar to each other. I want to use them as tactics for proof generation. 
Please remove the duplicates, so that I can have a list of only unique theorems.
For example, the following four theorems would be duplicates of each other:
```lean4
theorem problem1 : (v0 : Nat) -> v0 * 1 = v0
theorem problem2 : (v0 : Nat) -> (v1 : Nat) -> v1 * 1 = v1
theorem problem3 : (v0 : Nat) -> (v1 : Nat) -> (v2 : v0 = v1) -> v1 * 1 = v1
theorem problem4 : (v0 : Nat) -> v0 * (Nat.succ 0) = v0
```
The inclusion of an extra variable in problem 2 doesn't change the fact that the result is exactly the same, and the different names for the variable doesn't affect the result.
Problem 3 introduces an irrelevant hypothesis, which doesn't get used in the theorem, and the conclusion is still the same.
The last one is a trivial result of the others, as 1 is defined as Nat.succ 0 in this case.
Here is my list of theorems for you to remove duplicates for. 
{}
I also have attached an explanation for why each could be useful for a theorem prover.
{}
Think it through step by step, and then return the list of unique theorems from this list in a list format inside of a ```lean4``` code block. Make sure your answer is inside the very last lean codeblock. Please make sure to repeat the theorems exactly as I wrote them.
"""
    chat = [
        {"role": "developer", "content": "You are an expert at evaluating lean theorems"},
        {"role": "user", "content": prompt.format('\n'.join([u.thm_string for u in useful_theorems]), "\n\n".join(['\n'.join(u.thm_string + "\n" + u.explanations[0]) for u in useful_theorems]))}
    ]
    # print(chat[1]["content"][:8000])
    completion = await client.chat.completions.create(
        model=MODEL,
        messages=chat, # type: ignore
    )
    res = completion.choices[0].message.content
    assert res
    res_theorems: str = re.findall(r'(?s)```lean.*```', res)[-1]
    outs: set[LLMUsefulnessEvalTheorem] = set()
    for res_thm in res_theorems.split('\n'):
        if '```' in res_thm:
            continue
        for ut in useful_theorems:
            if res_thm == ut.thm_string:
                outs.add(ut)
    return list(outs), (res_theorems, res)

def _what_if_usage_only_metrics (exp_folder: str, it: int) -> tuple[int, int]:
    theorems: list[UsefulConjecture] = []
    pre = lambda x : os.path.join(exp_folder, x)
    with open (pre(f"usefulness_outcomes_{it}.json")) as f:
        outcomes = UsefulnessOutcomeList.validate_python(json.load(f))
    
    theorem_names = set()
    total_usage = 0
    for outcome in outcomes:
        if outcome.proof:
            for line in outcome.proof:
                if "by c" in line or "apply c" in line:
                    name = line.split()[-1]
                    theorem_names.add(name)
                    total_usage += 1
    return total_usage, len(theorem_names)

def _fix_eval_metrics (exp_folder: str, it: int) -> tuple[int, int]:
    
    pre = lambda x : os.path.join(exp_folder, x)
    # with open (pre(f"outcomes_{it}.json")) as f:
    #     outcomes = ProofOutcomeList.validate_python(json.load(f))
    with open (pre(f"generated_theorems_{it}.json")) as f:
        theorems = UsefulConjectureList.validate_python(json.load(f))
    for thm in theorems:
        thm.freq_used = 0
        thm.tot_improvement = 0
        thm_def = " : ".join(thm.theorem.split(" : ")[1:])
    
    with open(pre(f"usefulness_outcomes_{it}.json")) as f:
        u_outcomes = UsefulnessOutcomeList.validate_python(json.load(f))
    for outcome in u_outcomes:
        if outcome.improvement > 0:
            for line in outcome.proof:
                for thm in theorems:
                    thm_name = thm.theorem.split(" : ")[0]
                    if thm_name in line:
                        thm.freq_used += 1
                        thm.tot_improvement += outcome.improvement
                        break
    return sum([thm.freq_used for thm in theorems]), len([thm for thm in theorems if thm.freq_used > 0])

def _smt_prove (theorem: LLMUsefulnessEvalTheorem) -> bool:
    variables = dict()
    def _convert_to_smt(thm_str: str):
        """ Convert lean to SMT. Is good for naturals, maybe not for prop logic/group theory
        """
        nonlocal variables
        if thm_str[0] == "(": 
            thm_str = thm_str[1:-1]
            
            atoms = _find_atoms (thm_str)
            if atoms[0] == "Nat.succ":
                return 1 + _convert_to_smt(atoms[1])
            if atoms[0] == "¬":
                return Not(_convert_to_smt(atoms[1]))
            if atoms[1] == "->" or atoms[1] == "→":
                rest = ' '.join(atoms[2:])
                if "->" in rest or "→" in rest:
                    rest = "(" + rest + ")"
                
                if ": Nat" in atoms[0]:
                    varname = atoms[0].split()[0][1:]
                    variables[varname] = Int(varname)
                    return _convert_to_smt(rest)
                elif ": Prop" in atoms[0]:
                    varname = atoms[0].split()[0][1:]
                    variables[varname] = Bool(varname)
                    return _convert_to_smt(rest)
                elif ": G" in atoms[0]:
                    varname = atoms[0].split()[0][1:]
                    variables[varnae] = Group()
                else:
                    # First atom will be of the form (vX : <hypothesis>), when we want it to be (<hypothesis>)
                    hyp = " : ".join(atoms[0][1:-1].split(" : ")[1:])
                    return Implies(_convert_to_smt(hyp), _convert_to_smt(rest))
            if atoms[1] == "*":
                return _convert_to_smt(atoms[0]) * _convert_to_smt(atoms[2])
            if atoms[1] == "+":
                return _convert_to_smt(atoms[0]) + _convert_to_smt(atoms[2])
            if atoms[1] == "=" or atoms[1] == "↔":
                return _convert_to_smt(atoms[0]) == _convert_to_smt(atoms[2])
            if atoms[1] == "∧":
                return And(_convert_to_smt(atoms[0]), _convert_to_smt(atoms[2]))
            if atoms[1] == "∨":
                return Or(_convert_to_smt(atoms[0]), _convert_to_smt(atoms[2]))
        

        else:
            # this is either a number or a variable.
            if thm_str in variables:
                return variables[thm_str]
            # else:
                # print(f"Not a variable: {thm_str}") # Check manually that I'm not accidentally losing variables
            if thm_str == "false":
                return False
            if thm_str == "true":
                return True
            return int(thm_str)


    try:
        to_solve = Not(_convert_to_smt(theorem.thm_string_simple))
    except:
        print("help")
        to_solve = Not(_convert_to_smt(theorem.thm_string_simple))
    # print(to_solve)
    for varname in variables:
        if isinstance(variables[varname], ArithRef):
            to_solve = And(to_solve, variables[varname] >= 0)
        else:
            to_solve = And(to_solve)

    s = Solver()
    s.add(to_solve)
    res = s.check()
    return res.r == -1


async def run_evaluation_at_k (outcomes_filepath: str, output_folder_path: str, k: int = 1) -> None:
    os.makedirs(output_folder_path, exist_ok = True)
    if not os.path.exists(outcomes_filepath):
        raise FileNotFoundError(f"Cannot find outcomes file")
    
    cfg_path = os.path.join(os.path.dirname(outcomes_filepath), "flags.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Cannot find flags file (needed to detect theory)")
    
    with open(cfg_path) as f:
        flags = json.load(f)
    theory_name = flags["theory"]["name"]

    print(f"Detected theory: {theory_name}")

    theorem_list: list[LLMUsefulnessEvalTheorem] = _extract_theorems_from_outcomes(outcomes_filepath, include_unproven=True, theory_name = theory_name)
    prompt_list = _make_prompts([t.thm_string for t in theorem_list], theory_name = theory_name)
    out = LLMUsefulnessEvalResult()

    print(f"Sending {k * len(prompt_list)} requests in 5s")
    time.sleep(5)
    print("Sending")

    iteration_results = np.zeros((k, 3))

    for i in range(k):
        local_results = LLMUsefulnessEvalResult()
        promises: list[Awaitable] = []

        for prompt in prompt_list:
            completion = client.chat.completions.create(
                model= MODEL,
                messages=prompt, # type: ignore
            )
            promises.append(completion)
        print("All requests sent")

        results = []
        batch_size = 100
        # send by batches of batch_size because program crashes with 2k requests at once
        progress = tqdm.tqdm(total=len(promises))
        for batch_i in range(0, len(promises), batch_size):
            batch = promises[batch_i:min(batch_i+batch_size, len(promises))]
            results.extend(await asyncio.gather(*batch))
            progress.update(len(batch))
        progress.close()
    
        for theorem, completion in zip(theorem_list, results):
            response: str = completion.choices[0].message.content
            theorem.explanations.append(response)
            if "USEFUL" not in response.split("\n")[-1]:
                continue # malformed response, rare and even rarer for this to be a false negative
            if "NOT" not in response.split("\n")[-1]:
                local_results.useful_theorems.add(theorem)
        local_results.deduplicated_theorems = set((await _remove_dedup(list(local_results.useful_theorems)))[0])
        for dedup_thm in local_results.deduplicated_theorems:
            if not dedup_thm.dedup_useful_at_k:
                dedup_thm.dedup_useful_at_k = [False] * k
            dedup_thm.dedup_useful_at_k[i] = True
        local_results.proven_deduplicated = set(filter(lambda x : x.proven, local_results.deduplicated_theorems))
        os.makedirs(os.path.join(output_folder_path, str(i)), exist_ok=True)
        local_results.dump_to_folder(os.path.join(output_folder_path, str(i)))

        iteration_results[i] = [len(local_results.useful_theorems), len(local_results.deduplicated_theorems), len(local_results.proven_deduplicated)]
        out.extend(local_results)
    print(iteration_results)
    with open(os.path.join(output_folder_path, "iteration_results.txt"), "w") as f:
        f.write("\n".join(list(map(str, iteration_results))))
    exp_name = os.path.basename(output_folder_path)
    averages = np.average(iteration_results, 0)
    var = np.sqrt(np.var(iteration_results, 0))
    print (f"Results for {exp_name}:")
    print (f"Useful theorems: avg {averages[0]}, std {var[0]}")
    print (f"Deduplicated theorems: avg {averages[1]}, std {var[1]}")
    print (f"Proven theorems: avg {averages[2]}, std {var[2]}")
    out.dump_to_folder(output_folder_path)
    

async def remove_dedup_test (output_folder_path: str) -> None:
    pre = lambda x: os.path.join(output_folder_path, x)
    with open(pre("useful_theorems.json")) as f:
        useful_theorems = [LLMUsefulnessEvalTheorem.model_validate(d) for d in json.load(f)]
    res = []
    processes = []
    for it in range(5):
        local_theorems = []
        for ut in useful_theorems:
            response = ut.explanations[it]
            if "USEFUL" not in response.split("\n")[-1]:
                continue # malformed response, rare and even rarer for this to be a false negative
            if "NOT" not in response.split("\n")[-1]:
                local_theorems.append(ut)
        processes.append(_remove_dedup(local_theorems, "groups"))
    res = await tqdm.asyncio.tqdm_asyncio.gather(*processes)
    for it in range(len(res)):
        with open(pre(f"dedup_test_{it}.txt"), "w") as f:
            f.write(res[it][1][1])
    res_thms = [len(r[0]) for r in res]
    print (f"Average: {np.average(res_thms)}")
    print (f"Std: {np.sqrt(np.var(res_thms))}")


def get_reproof_values (output_folder_path: str) -> None:

    cfg_path = os.path.join(os.path.dirname(output_folder_path), "flags.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Cannot find flags file (needed to detect theory)")
    
    with open(cfg_path) as f:
        flags = json.load(f)
    theory_name = flags["theory"]["name"]

    print(f"Detected theory: {theory_name}")

    # It's more difficult than it would seem as the function is non-injective. So we first need to match which theorems were originally evaluated.
    assert os.path.exists(os.path.join(output_folder_path, "final_outcomes.json"))
    useful_theorems = _extract_theorems_from_explanation_file(os.path.join(output_folder_path, "useful_theorem_dedup.txt"))
    original_outcomes = _extract_theorems_from_outcomes(os.path.join(output_folder_path, "outcomes_9.json"), include_unproven=True, theory_name=theory_name)
    old_proven_theorems = _extract_theorems_from_explanation_file(os.path.join(output_folder_path, "useful_theorem_dedup_proven.txt"))
    useful_theorem_names = list(map(lambda x: x.thm_string, useful_theorems))
    original_names = []
    for i,thm in enumerate(original_outcomes):
        if thm.thm_string in useful_theorem_names:
            original_names.append(thm.thm_org)
    
    # So here original_names will have the peano theorems that we want to find

    with open(os.path.join(output_folder_path, "final_outcomes.json")) as f:
        proven_theorems = [o for o in json.load(f) if not o["hindsight"]]
    intersect = []
    count = 0 
    for thm in proven_theorems:
        if thm["problem"] in original_names:
            count += 1
            if thm["proof"]:
                intersect.append(thm["problem_translated"])
    print (f"Found theorems {count}/{len(useful_theorems)}") # Note: count can be larger than useful_theorems. This is because the model is allowed to conjecture the same theorem multiple times if it failed the proof.
    print (f"Proven theorems: {len(intersect)}/{(len(useful_theorems))} ({len(intersect) - len(old_proven_theorems)} improvement)")

def get_fix_evaluation_metrics (exp_folder: str, it: int = 9) -> None:
    add_pre = lambda file : os.path.join(exp_folder, file)
    with open(add_pre(f"outcomes_{it}.json")) as f:
        outcomes = ProofOutcomeList.validate_python(json.load(f))
    correct_proof_count = len([o for o in outcomes if o.proof and not o.hindsight])
    with open (add_pre(f"generated_theorems_{it}.json")) as f:
        generated_theorems = json.load(f)
    
    inc_total_use_count = sum([int(gt["freq_used"]) for gt in generated_theorems])
    inc_number_of_theorems = len([1 for gt in generated_theorems if int(gt["freq_used"])])
    exp_name = os.path.basename(exp_folder)
    total_use_count, number_of_theorems = _fix_eval_metrics (exp_folder, it)
    usage_use_count, usage_theorem_count = _what_if_usage_only_metrics (exp_folder, it)

    print (f"Results for {exp_name} (iteration {it}):")
    print (f"Number of proven conjectures: {correct_proof_count}")
    print (f"(I) Total theorem usage count: {inc_total_use_count}")
    print (f"(I) Number of useful theorems: {inc_number_of_theorems}")
    print (f"(C) Total theorem usage count: {total_use_count}")
    print (f"(C) Number of useful theorems: {number_of_theorems}")
    print (f"If we were to only consider usage...")
    print (f"(U) Total theorem usage count: {usage_use_count}")
    print (f"(U) Number of useful theorems: {usage_theorem_count}")
    print (f"{number_of_theorems} | {total_use_count} | {usage_theorem_count} | {usage_use_count}")
    print()
def count_its (filename: str) -> None:
    with open(filename) as f:
        lines = f.readlines()
    counts = [0 for i in range(10)]
    for l in lines:
        x = l.split()
        for i, c in enumerate(x):
            if "iteration" in c:
                it = int(x[i+1][0])
                counts[it] += 1
    print(counts)

def reproof_using_smt (exp_folder: str, k=5) -> None:
    pre = lambda x: os.path.join(exp_folder, x)
    with open(pre("useful_theorem_dedup.json")) as f:
        useful_theorems_tmp = [LLMUsefulnessEvalTheorem.model_validate(thm) for thm in json.load(f)]
    results = []
    len_useful = []
    for it in range(k):
        useful_theorems: list[LLMUsefulnessEvalTheorem] = [thm for thm in useful_theorems_tmp if thm.dedup_useful_at_k[it]]
        len_useful.append(len(useful_theorems))
        proven = []
        for useful_theorem in useful_theorems:
            if _smt_prove(useful_theorem):
                proven.append(useful_theorem)
        results.append(len(proven))
    print(len_useful)
    print(f"Deduplicated conjectures: {np.average(len_useful)}, std {np.sqrt(np.var(len_useful))}")
    print(f"Proven conjectures: {np.average(results)}, std {np.sqrt(np.var(results))}")

async def fix_dedup (output_folder: str) -> None:
    pre = lambda x : os.path.join(output_folder, x)
    with open(pre("useful_theorem_dedup.json")) as f:
        try:
            useful_theorems_dedup = [LLMUsefulnessEvalTheorem.model_validate(d) for d in json.load(f)]
        except Exception as e:
            print(output_folder, e)
            return
    if useful_theorems_dedup[0].dedup_useful_at_k: # Doesn't need fixing
        return
    # dedup_map: dict[str, LLMUsefulnessEvalTheorem] = {}
    # for thm in useful_theorems_dedup:
    #     dedup_map[thm.thm_string] = thm
    
    with open(pre("useful_theorems.json")) as f:
        useful_theorems = [LLMUsefulnessEvalTheorem.model_validate(d) for d in json.load(f)]

    
    for it_k in range(5):
        ut_k = []
        for ut in useful_theorems:
            response = ut.explanations[it_k].split("\n")[-1]
            if not "NOT" in response:
                ut_k.append(ut)
        dedup_ut_k = (await _remove_dedup(ut_k))[0]
        for thm in dedup_ut_k:
            if not thm.dedup_useful_at_k:
                thm.dedup_useful_at_k = [False] * 5
            thm.dedup_useful_at_k[it_k] = True
    useful_theorems_dedup = [thm for thm in useful_theorems if any(thm.dedup_useful_at_k)]
        
    with open(pre("useful_theorem_dedup.json"), "w") as f:
        json.dump([thm.model_dump() for thm in useful_theorems_dedup], f)
            
async def fix_all_dedup ():
    folders = os.listdir("/home/timothekasriel/minimo_org/learning/outputs")
    to_run = []
    print("To run: \n")
    for exp in folders:
        if os.path.exists(os.path.join("/home/timothekasriel/minimo_org/learning/outputs", exp, "useful_theorem_dedup.json")):
            to_run.append(fix_dedup(os.path.join("/home/timothekasriel/minimo_org/learning/outputs", exp)))
            print(exp)
    time.sleep(5)
    await asyncio.gather(*to_run)



if __name__ == "__main__":
    # asyncio.run(fix_all_dedup())
    # sys.exit(0)
    exp_folders = [
        "/home/timothekasriel/minimo/learning/outputs/line33",
        # "/home/timothekasriel/minimo/learning/outputs/line33_2",
        "/home/timothekasriel/minimo/learning/outputs/line33_3",
        "/home/timothekasriel/minimo/learning/outputs/line33_prop",
        # "/home/timothekasriel/minimo/learning/outputs/line33_group",
        "/home/timothekasriel/minimo/learning/outputs/line33.8",
        "/home/timothekasriel/minimo/learning/outputs/line33.10",
        "/home/timothekasriel/minimo_org/learning/outputs/base",
        "/home/timothekasriel/minimo_org/learning/outputs/base_2",
        "/home/timothekasriel/minimo_org/learning/outputs/base_3",
        "/home/timothekasriel/minimo_org/learning/outputs/base_prop",
        # "/home/timothekasriel/minimo_org/learning/outputs/base_group",
    ]
    OUTPUT_FOLDER = "/home/timothekasriel/minimo/learning/outputs/line2_group_2"
    print (f"Running on {OUTPUT_FOLDER}")

    # asyncio.run(remove_dedup_test(OUTPUT_FOLDER))


    asyncio.run(run_evaluation_at_k(os.path.join(OUTPUT_FOLDER, "outcomes_9.json"), OUTPUT_FOLDER, 3))
    # get_fix_evaluation_metrics(OUTPUT_FOLDER, it=9)
    # reproof_using_smt(OUTPUT_FOLDER)