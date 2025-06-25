import json
import os
import random
from openai import OpenAI
import dotenv

from convert_to_lean import convert_arith

if not dotenv.load_dotenv():
    raise ValueError("Need to set api key in .env")
print (f"OPENAI API KEY: {os.getenv('OPENAI_API_KEY')}")
model = OpenAI()

def _extract_theorems_from_outcomes(outcomes_filepath: str, logprobs: bool = False) -> list[str] | list[tuple]:
    with open(outcomes_filepath) as f:
        outcomes = json.load(f)
    results = []
    for i, o in enumerate(outcomes):
        if o.get("hindsight", False):
            continue
        if "problem_translated" not in o.keys():
            # Old non-translated problems, pre-dsprover
            o["problem_translated"] = convert_arith(o["problem"], i, flag_matters=False)
        thm_str = f"theorem problem{i} : {o['problem_translated']}"
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
As well as access to the `rfl` and `rewrite` commands
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

def send_batch_evaluation (outcomes_filepath: str, output_folder_path: str, name: str = "Minimo evaluation") -> None:
    if not os.path.exists(output_folder_path):
        raise FileNotFoundError(f"{os.path.join(__path__, output_folder_path)}")
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
                    use_f.writelines(response)
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
            if thm_2[0] != '((v0 : Nat) -> (v1 : (v0 = 0)) -> (v2 : (v0 = v0)) -> (v3 : ((Nat.succ 0) = v0)) -> (v4 : (0 = v0)) -> (v5 : (0 = v0)) -> ((Nat.succ (Nat.succ 0)) = 0))':
                continue
            print(len(to_prove))
            new_theorem = f"({thm_1[0]}) -> {thm_2[0]}"
            to_prove.append(new_theorem)
    print(len(to_prove))
    res = dsprover.prove(to_prove, debug=True)
    with open(os.path.join(output_folder, "res_chosen.txt"), "w") as f:
        f.write("\n\n\n".join(res))
    return
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





OUTPUT_FOLDER = "/home/timothekasriel/minimo/learning/outputs/llm_eval"

# send_batch_evaluation("/Users/tkasriel/code/rsh/minimo/learning/outputs/orig_minimo/outcomes_arith.json", OUTPUT_FOLDER, "Original Minimo Arithmetic Evaluation")
# list_batch_evaluations(OUTPUT_FOLDER)
# cancel_batch("batch_68523b4c6ea081908042e6131d7bea8e")
# get_batch_evaluations("batch_68523c3f921c8190a46b4f7f0d508d37", OUTPUT_FOLDER)
make_graph("/home/timothekasriel/minimo/learning/outputs/2025-06-05/11-55-28/outcomes_16.json", OUTPUT_FOLDER)
# draw_graph("/home/timothekasriel/minimo/learning/outputs/llm_eval/graph.csv", OUTPUT_FOLDER)
# look_at_graph("/home/timothekasriel/minimo/learning/outputs/llm_eval/graph.csv")