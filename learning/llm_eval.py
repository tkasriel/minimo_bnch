import asyncio
import json
import os
import random
import re
from openai import AsyncOpenAI
import dotenv

from convert_to_lean import convert_arith

if not dotenv.load_dotenv():
    raise ValueError("Need to set api key in .env")
print (f"OPENAI API KEY: {os.getenv('OPENAI_API_KEY')}")
client = AsyncOpenAI()
MODEL = "gpt-4.1"

def _extract_theorems_from_outcomes(outcomes_filepath: str, logprobs: bool = False, change_str: bool = True) -> list[str] | list[tuple]:
    with open(outcomes_filepath) as f:
        outcomes = json.load(f)
    results = []
    for i, o in enumerate(outcomes):
        if o.get("hindsight", False):
            continue
        if "problem_translated" not in o.keys():
            # Old non-translated problems, pre-dsprover
            o["problem_translated"] = convert_arith(o["problem"], i, flag_matters=False)
        thm_str = f"theorem problem{i} : {o['problem_translated']}" if change_str else o["problem_translated"]
        if logprobs:
            try:
                if float(o["logprob"]) < 0:
                    results.append((o["problem_translated"], float(o["logprob"])))
            except:
                continue
        else:
            results.append(thm_str)
    return results

def _make_prompts (theorems: list[str]) -> list[list[dict[str]]]:
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
As well as access to the `rfl` and `rewrite` commands.
Equality is symmetric and reflexive.
Here is the theorem you are to evaluate:
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

async def _remove_dedup (useful_theorems: list[tuple[str, str]]) -> list[tuple[str,str]]:
    prompt = """I have a set of lean theorems, some of which are very similar to each other.
Please remove the duplicates, so that I can have a list of only unique theorems.
For example, the following three theorems would be duplicates of each other:
```lean4
theorem problem1 : (v0 : Nat) -> v0 * 1 = v0
theorem problem2 : (v0 : Nat) -> (v1 : Nat) -> v1 * 1 = v1
theorem problem3 : (v0 : Nat) -> (v1 : Nat) -> (v2 : v0 = v1) -> v1 * 1 = v1
```
The inclusion of an extra variable in problem 2 doesn't change the fact that the result is exactly the same, and the different names for the variable doesn't affect the result.
Problem 3 introduces an irrelevant hypothesis, which doesn't get used in the theorem, and the conclusion is still the same.
```lean4
{}
```
Think it through step by step, and then return the list of unique theorems in the same format I gave them
"""
    chat = [
        {"role": "developer", "content": "You are an expert at evaluating lean theorems"},
        {"role": "user", "content": prompt.format("\n".join([u[0] for u in useful_theorems]))}
    ]
    completion = await client.chat.completions.create(
        model=MODEL,
        messages=chat
    )
    res = completion.choices[0].message.content
    print(res)
    assert res
    res_theorems = re.search(r'```lean(?s).*```', res).group()
    outs: list[tuple[str,str]] = []
    for res_thm in res_theorems.split('\n'):
        if '```' in res_thm:
            continue
        for ut in useful_theorems:
            if res_thm == ut[0]:
                outs.append(ut)
    return outs


async def run_evaluation (outcomes_filepath: str, generated_theorems_filepath: str, output_folder_path: str) -> None:
    os.makedirs(output_folder_path, exist_ok = True)
    if not os.path.exists(outcomes_filepath):
        raise FileNotFoundError(f"Cannot find outcomes file")
    theorem_list: list[str] = _extract_theorems_from_outcomes(outcomes_filepath)
    prompt_list = _make_prompts(theorem_list)
    promises = []
    for prompt in prompt_list:
        completion = client.chat.completions.create(
            model= MODEL,
            messages=prompt
        )
        promises.append(completion)
    results = []
    useful_theorems = []
    for p in promises:
        results.append(await p)
    for theorem, completion in zip(theorem_list, results):
        response = completion.choices[0].message.content
        if "NOT" not in response.split("\n")[-1]:
            useful_theorems.append((theorem, response))
    with open(os.path.join(output_folder_path, "useful_theorems.txt")) as f:
        for tup in useful_theorems:
            f.write(tup[0] + "\n" + tup[1] + "\n ============ \n\n")
    useful_theorems = await _remove_dedup(useful_theorems)
    with open(os.path.join(output_folder_path, "useful_theorem_dedup.txt"), "w") as f:
        for tup in useful_theorems:
            f.write(tup[0] + "\n" + tup[1] + "\n ============ \n\n")

    
    

    


def send_batch_evaluation (outcomes_filepath: str, output_folder_path: str, name: str = "Minimo evaluation") -> None:
    os.makedirs(output_folder_path, exist_ok=True)
    if not os.path.exists(outcomes_filepath):
        raise FileNotFoundError(f"{os.path.join(outcomes_filepath)}")
    batch_file = os.path.join(output_folder_path, "batch_request.jsonl")
    
    theorem_list = _extract_theorems_from_outcomes(outcomes_filepath)
    print(f"Got {len(theorem_list)} theorems")
    prompts = _make_prompts(theorem_list)

    with open(batch_file, 'w') as f:
        for i, chat in enumerate(prompts):
            request_dict = {
                "custom_id": f"minimo-eval-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4.1",
                    "messages": chat,
                    "max_tokens": 8192
                }
            }
            json.dump(request_dict, f)
            if i < len(prompts) - 1:
                f.write('\n')
    openai_file = model.files.create(file=open(batch_file, "rb"), purpose="batch")
    model.batches.create(
        input_file_id=openai_file.id,
        completion_window='24h',
        endpoint="/v1/chat/completions",
        metadata={
            "name": name
        }
    )

def list_batch_evaluations (output_folder: str):
    results = model.batches.list(limit=5)
    with open(os.path.join(output_folder, "batch_list.json"), "w") as f:
        f.write(results.to_json())

def cancel_batch (id: str):
    model.batches.cancel(batch_id=id)

def get_batch_evaluations (id: str, output_folder: str):
    batch = model.batches.retrieve(batch_id=id)
    success = batch.request_counts.completed
    useful = 0
    malformed = 0
    if batch.output_file_id:
        output = model.files.content(file_id=batch.output_file_id)
        text = output.response.text
        with open(os.path.join(output_folder, "useful_theorems.txt"), "w") as use_f:
            # with open(os.path.join(output_folder, ""))
            for res in text.strip().split("\n"):
                res_dict = json.loads(res)
                # print(res_dict)
                response = res_dict["response"]["body"]["choices"][0]["message"]["content"].split("\n")
                if "USEFUL" not in response[-1]:
                    malformed += 1
                elif "NOT" not in response[-1]:
                    useful += 1
                    use_f.write("\n".join(response))
                    use_f.write("\n----\n\n")
    
    print ("RESULTS")
    print (f"Succesful runs: {success}/{batch.request_counts.total}")
    print (f"Useful theorems: {useful}/{success}")
    print (f"Malformed results: {malformed}/{success}")
    # for res in results.:

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
            new_theorem = f"({thm_1[0]}) -> {thm_2[0]}"
            to_prove.append(new_theorem)
    # print(len(to_prove))
    res = [r for r in dsprover.prove(to_prove)]
    # with open(os.path.join(output_folder, "res_chosen.txt"), "w") as f:

    #     f.write("\n\n\n".join(res))
    # return
    logprob_tot = {}
    index = 0
    for thm_1 in thm_logprob:
        for thm_2 in thm_logprob:
            if thm_1 == thm_2: # there's a better way to do this I just dont want to do it rn
                continue
            if res[index][0]:
                if thm_2[0] not in logprob_tot.keys():
                    logprob_tot[thm_2[0]] = 0.0
                # print(logprob_tot[thm_2[0]])
                # print(res[index])
                logprob_tot[thm_2[0]] += res[index][1]
            index += 1
    logprob_avg = {k:v/(len(thm_logprob)-1) for k,v in logprob_tot.items()}
    index = 0
    for thm_1 in thm_logprob:
        for thm_2 in thm_logprob:
            if thm_1 == thm_2:
                continue
            logprob_new = res[index][1]
            logprob_old = max(logprob_avg[thm_2[0]],thm_2[1])
            if res[index][0] and logprob_new - logprob_old > 0.5:
                if thm_1 not in adj:
                    adj[thm_1[0]] = []
                adj[thm_1[0]].append((thm_2[0], logprob_new - logprob_old))
            index += 1
    with open(os.path.join(output_folder, "graph.csv"), "w") as f:
        f.write("A,B,weight\n")
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

def compare_batch_evaluation (results_id: str, outcome_filepath: str, generated_theorems_filepath: str, output_folder: str):
    outcomes = _extract_theorems_from_outcomes(outcome_filepath, change_str=False)
    batch = model.batches.retrieve(batch_id=results_id)
    useful_theorems = []
    if batch.output_file_id:
        output = model.files.content(file_id=batch.output_file_id)
        text = output.response.text
        with open(os.path.join(output_folder, "useful_probs.csv"), "w") as out:
            out.write("theorem,tot_logprob\n")
            print(len(text.strip().split("\n")), len(outcomes))
            for i,res in enumerate(text.strip().split("\n")):
                res_dict = json.loads(res)
                response = res_dict["response"]["body"]["choices"][0]["message"]["content"].split("\n")
                if "NOT" not in response[-1]:
                    useful_theorems.append(outcomes[i][0])
            ut = "[('a0 : nat) -> ('a1 : nat) -> ('a2 : (= (* (s o) (+ 'a0 o)) o)) -> (= (* (+ 'a0 o) o) (* (+ 'a0 o) o))]"
            useful_theorems.append(convert_arith(ut,0,False))
            with open(generated_theorems_filepath) as f:
                gen_theorems = json.load(f)
                for g in gen_theorems:
                    stripped = " : ".join((g["theorem"][:-1]).split(" : ")[1:])
                    # print(stripped)
                    if convert_arith(stripped, 0, flag_matters=False) in useful_theorems:
                        out.write(f"{stripped},{g['tot_improvement']}")

        



OUTPUT_FOLDER = "/home/timothekasriel/minimo/learning/outputs/llm_eval_dsprover_1"

# send_batch_evaluation("/home/timothekasriel/minimo/learning/outputs/dsprover_usefulness/outcomes_4.json", OUTPUT_FOLDER, "dsprover usefulness 1")
# send_batch_evaluation("/home/timothekasriel/minimo/learning/outputs/no_useful/outcomes_2.json", OUTPUT_FOLDER, "No Usefulness")
# list_batch_evaluations(OUTPUT_FOLDER)
# cancel_batch("batch_68523b4c6ea081908042e6131d7bea8e")
# get_batch_evaluations("batch_6863b1e072cc8190a0de0f79c7082735", OUTPUT_FOLDER)
# useful_theorems = [
#     ("theorem problem4678 : ((v0 : Nat) -> (v2 : (v0 = 1)) -> (v3 : Nat) -> (v4 : Nat) -> (1 = v0))",None),
#     ("theorem problem4679 : ((v0 : Nat) -> (v1 : Nat) -> (v2 : (v2 = 1)) -> (v4 : Nat) -> (1 = v2))",None),
#     ("theorem problem4680 : ((v0 : Nat) -> (v1 : Nat) -> (v2 : (Nat.succ v2 = 1 + 1)) -> (v4 : Nat) -> (1 + 1 = Nat.succ v2))",None),
#     ("theorem problem30 : ((v0 : Nat) ->  (v1 : Nat) -> v1 * 1 = v1)",None),
#     ("theorem problem27 : ((v0 : Nat) -> 1 * v0 = v0)",None),
#     ("theorem problem278 : ((v0 : Nat) -> 1 * (Nat.succ v0) = (Nat.succ v0))",None),
#     ("theorem problem64 : ((v0 : Nat) -> (v1 : 1 = 0) -> 1 + 1 = 0",None),
# ]
print(asyncio.run(_remove_dedup(useful_theorems)))
# compare_batch_evaluation("batch_686332a4461481908e731d62e865738f","/home/timothekasriel/minimo/learning/outputs/new_usefulness/outcomes_9.json","/home/timothekasriel/minimo/learning/outputs/new_usefulness/generated_theorems_9.json", OUTPUT_FOLDER)
# make_graph("/home/timothekasriel/minimo/learning/outputs/2025-06-05/11-55-28/outcomes_16.json", OUTPUT_FOLDER)
# draw_graph("/home/timothekasriel/minimo/learning/outputs/llm_eval/graph.csv", OUTPUT_FOLDER)
# look_at_graph("/home/timothekasriel/minimo/learning/outputs/llm_eval/graph.csv")