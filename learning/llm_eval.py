import asyncio
import json
import os
import random
import re
import sys
import time
from typing import Awaitable
import numpy as np
from openai import AsyncOpenAI
import dotenv
from pydantic import BaseModel
import tqdm
from tqdm.asyncio import tqdm_asyncio

from convert_to_lean import convert_arith

if not dotenv.load_dotenv():
    raise ValueError("Need to set api key in .env")
print (f"OPENAI API KEY: {os.getenv('OPENAI_API_KEY')}")
client = AsyncOpenAI()
MODEL = "gpt-4.1"

class Theorem(BaseModel):
    thm_string: str
    iteration: int
    proven: bool
    logprob: float
    thm_string_simple: str
    thm_org: str | None = None
    explanations: list[str] = []
    def __str__(self) -> str:
        return f"{self.thm_string} (iteration {self.iteration}) : {self.logprob} -- {self.thm_org or ''}\n\n" + "\n#####\n".join(self.explanations)
    def __hash__(self) -> int:
        return (self.thm_string + " -- " + (self.thm_org or "")).__hash__()

    def __eq__(self, other: object) -> bool:
        return self.__hash__() == other.__hash__()
    
class LLMUsefulnessEvalResult (BaseModel):
    useful_theorems: set[Theorem] = set()
    deduplicated_theorems: set[Theorem] = set()
    proven_deduplicated: set[Theorem] = set()

    def dump_to_folder (self, folder_path: str) -> None:
        add_pre = lambda filepath : os.path.join(folder_path, filepath)
        with open(add_pre("useful_theorems.txt"), "w") as f:
            f.write("\n============\n\n".join(map(str, self.useful_theorems)))
        with open(add_pre("useful_theorem_dedup.txt"), "w") as f:
            f.write("\n============\n\n".join(map(str, self.deduplicated_theorems)))
        with open(add_pre("useful_theorem_dedup_proven.txt"), "w") as f:
            f.write("\n============\n\n".join(map(str, self.proven_deduplicated)))
        
        with open(add_pre("useful_theorems.json"), "w") as f:
            f.write(str([x.model_dump_json() for x in self.useful_theorems]))
        with open(add_pre("useful_theorem_dedup.json"), "w") as f:
            f.write(str([x.model_dump_json() for x in self.deduplicated_theorems]))
    
    def extend (self, other: "LLMUsefulnessEvalResult") -> None:
        self.useful_theorems.update(other.useful_theorems)
        self.deduplicated_theorems.update(other.deduplicated_theorems)
        self.proven_deduplicated.update(other.proven_deduplicated)
        

def _extract_theorems_from_outcomes(outcomes_filepath: str, include_hindsight: bool = False, include_unproven: bool = False) -> list[Theorem]:
    with open(outcomes_filepath) as f:
        outcomes = json.load(f)
    results = []
    for i, o in enumerate(outcomes):
        if o.get("hindsight", False) and not include_hindsight:
            continue
        if not o.get("proof", False) and not include_unproven:
            continue
        if "problem_translated" not in o.keys():
            # Old non-translated problems, pre-dsprover
            o["problem_translated"] = convert_arith(o["problem"], i, simplify=True)
        # if i == 2815:
        #     print(o)
        #     sys.exit(0)
        if o["logprob"] is None:
            o["logprob"] = 1.0
        thm_str = f"theorem problem{i} : {o['problem_translated']}"
        results.append(Theorem(thm_string=thm_str,
                                proven=bool(o["proof"]),
                                thm_org=o["problem"],
                                thm_string_simple=o["problem_translated"],
                                iteration=int(o.get("iteration", -1)),
                                logprob=float(o.get("logprob", 0.0))))
    return results

def _make_prompts (theorems: list[str]) -> list[list[dict[str, str]]]:
    prompt = """You are tasked to judge whether a given lean theorem could be considered useful for an automatic theorem prover to have among its known theorems.
This theorem prover has only access to the following axioms and known theorems:
```
Nat : type.
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
nat_ind : (p : (Nat -> Prop)) -> (p z) -> ((n : Nat) -> (p n) -> p (Nat.succ n)) -> (n : Nat) -> (p n).
```
As well as access to the `rfl` and `rewrite` commands
Here is the theorem you are to evaluate
```lean4
{}
```
Think through the problem step by step. Translate the problem into natural language, then think of what the possible uses of the theorem could be, whether it's obviously true and whether it means something.
On the last line, say either USEFUL or NOT USEFUL and nothing else.
"""
    chats = [[
        {"role": "developer", "content": "You are an expert at evaluating lean theorems"},
        {"role": "user", "content": prompt.format(theorem)}
    ] for theorem in theorems]
    return chats


async def _remove_dedup (useful_theorems: list[Theorem]) -> tuple[list[Theorem], tuple[str, str]]:
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
    outs: set[Theorem] = set()
    for res_thm in res_theorems.split('\n'):
        if '```' in res_thm:
            continue
        for ut in useful_theorems:
            if res_thm == ut.thm_string:
                outs.add(ut)
    return list(outs), (res_theorems, res)

def _extract_theorems_from_explanation_file (explanation_file: str) -> list[Theorem]:
    out = []
    with open (explanation_file) as f:
        lines = ("\n".join(f.readlines())).split("============")
    for l in lines:
        l_org = l[:]
        l = [line for line in l.split('\n') if line.strip()]
        if not l:
            continue
        theorem_name = l[0].split(" (iteration ")[0]
        theorem_name_simple = " : ".join(theorem_name.split(" : ")[1:])
        try:
            it = int((l[0].split(" (iteration ")[1])[0])
        except: 
            it = -1 # temp fix
        logprob = float(l[0].split(" : ")[-1])

        out.append(Theorem(thm_string=theorem_name,
                                       thm_string_simple=theorem_name_simple,
                                       proven=True,
                                       iteration=it,
                                       logprob=logprob,
                                       explanations = ["\n".join(l[1:])])) #TODO: fix
    return out


async def run_evaluation_at_k (outcomes_filepath: str, output_folder_path: str, k: int = 1) -> None:
    os.makedirs(output_folder_path, exist_ok = True)
    if not os.path.exists(outcomes_filepath):
        raise FileNotFoundError(f"Cannot find outcomes file")
    theorem_list: list[Theorem] = _extract_theorems_from_outcomes(outcomes_filepath, include_unproven=True)
    prompt_list = _make_prompts([t.thm_string for t in theorem_list])
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
        local_results.proven_deduplicated = set(filter(lambda x : x.proven, local_results.deduplicated_theorems))
        iteration_results[i] = [len(local_results.useful_theorems), len(local_results.deduplicated_theorems), len(local_results.proven_deduplicated)]
        out.extend(local_results)
    print(iteration_results)
    exp_name = os.path.dirname(output_folder_path)
    averages = np.average(iteration_results, 0)
    var = np.sqrt(np.var(iteration_results, 0))
    print (f"Results for {exp_name}:")
    print (f"Useful theorems: avg {averages[0]}, std {var[0]}")
    print (f"Deduplicated theorems: avg {averages[1]}, std {var[1]}")
    print (f"Proven theorems: avg {averages[2]}, std {var[2]}")
    out.dump_to_folder(output_folder_path)
    

async def remove_dedup_fix (output_folder_path: str) -> None:
    useful_theorems = _extract_theorems_from_explanation_file(os.path.join(output_folder_path, "useful_theorems.txt"))
    results = await _remove_dedup(useful_theorems)
    with open (os.path.join(output_folder_path, "useful_theorems_dedup_fix.txt"), "w") as f:
        f.write(results[1][0] + "\n\n" + results[1][1])

def get_reproof_values (output_folder_path: str) -> None:

    # It's more difficult than it would seem as the function is non-injective. So we first need to match which theorems were originally evaluated.
    assert os.path.exists(os.path.join(output_folder_path, "final_outcomes.json"))
    useful_theorems = _extract_theorems_from_explanation_file(os.path.join(output_folder_path, "useful_theorem_dedup.txt"))
    original_outcomes = _extract_theorems_from_outcomes(os.path.join(output_folder_path, "outcomes_9.json"), include_unproven=True)
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

def get_evaluation_metrics (exp_folder: str) -> None:
    add_pre = lambda file : os.path.join(exp_folder, file)
    with open(add_pre("outcomes_15.json")) as f:
        outcomes = json.load(f)
    correct_proof_count = len([o for o in outcomes if o["proof"] and not o["hindsight"]])
    with open (add_pre("generated_theorems_15.json")) as f:
        generated_theorems = json.load(f)
    total_use_count = sum([int(gt["freq_used"]) for gt in generated_theorems])
    number_of_theorems = len([1 for gt in generated_theorems if int(gt["freq_used"])])
    exp_name = os.path.basename(exp_folder)
    print (f"Results for {exp_name}:")
    print (f"Number of proven conjectures: {correct_proof_count}")
    print (f"Total theorem useage count: {total_use_count}")
    print (f"Number of useful theorems: {number_of_theorems}")
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




if __name__ == "__main__":
    exp_folders = [
        # "/home/timothekasriel/minimo/learning/outputs/line18_2",
        # "/home/timothekasriel/minimo/learning/outputs/line18_buggy",
        # "/home/timothekasriel/minimo/learning/outputs/line19_2",
        # "/home/timothekasriel/minimo/learning/outputs/line21",
        # "/home/timothekasriel/minimo/learning/outputs/line22",
        # "/home/timothekasriel/minimo/learning/outputs/line23",
        # "/home/timothekasriel/minimo/learning/outputs/line24",
        # "/home/timothekasriel/minimo/learning/outputs/line25",
        "/home/timothekasriel/minimo/learning/outputs/line27",
        "/home/timothekasriel/minimo/learning/outputs/line29"
    ]
    OUTPUT_FOLDER = "/home/timothekasriel/minimo/learning/outputs/line29"
    # count_its("/home/timothekasriel/minimo/learning/outputs/line18/useful_theorems.txt")
    # print (f"Current evaluation: {os.path.basename(OUTPUT_FOLDER)}")
    # print(get_reproof_values(OUTPUT_FOLDER))

    # asyncio.run(run_evaluation_at_k(os.path.join(OUTPUT_FOLDER, "outcomes_5.json"), OUTPUT_FOLDER, 5))
    # asyncio.run(run_evaluation_at_k(os.path.join(OUTPUT_FOLDER, "outcomes_9.json"), OUTPUT_FOLDER, 5))
    for exp_folder in exp_folders:
        get_evaluation_metrics(exp_folder)
    #     asyncio.run(run_evaluation_at_k(os.path.join(exp_folder, "outcomes_5.json"), exp_folder, 5))
    # asyncio.run(remove_dedup_fix(OUTPUT_FOLDER))
    # send_batch_evaluation("/Users/tkasriel/code/rsh/minimo/learning/outputs/orig_minimo/outcomes_arith.json", OUTPUT_FOLDER, "Original Minimo Arithmetic Evaluation")
    # list_batch_evaluations(OUTPUT_FOLDER)
    # cancel_batch("batch_68523b4c6ea081908042e6131d7bea8e")
    # get_batch_evaluations("batch_68523c3f921c8190a46b4f7f0d508d37", OUTPUT_FOLDER)
    # make_graph("/home/timothekasriel/minimo/learning/outputs/2025-06-05/11-55-28/outcomes_16.json", OUTPUT_FOLDER)
    # draw_graph("/home/timothekasriel/minimo/learning/outputs/llm_eval/graph.csv", OUTPUT_FOLDER)
    # look_at_graph("/home/timothekasriel/minimo/learning/outputs/llm_eval/graph.csv")