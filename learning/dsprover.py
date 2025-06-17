import json
import re
import time
import tqdm
from vllm import LLM, SamplingParams
import torch
from accelerate import PartialState

from lean_interact import LeanREPLConfig, AutoLeanServer, Command
torch.manual_seed(30)

model_id = "deepseek-ai/DeepSeek-Prover-V2-7B"
model = LLM(model=model_id)
tokenizer = model.get_tokenizer()
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# distributed_state = PartialSstate()
# local_rank = distributed_state.local_process_index
# device = torch.device(f"cuda:{local_rank}")
# model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"":device}, torch_dtype=torch.bfloat16, trust_remote_code=True)
server = AutoLeanServer(LeanREPLConfig())

def extract_code (encoded_response, logprobs, debug = False) -> tuple[str | None, float]:
    code_blocks: list[list[int]] = []
    logprobs_out = []
    code_block_open = False
    intro_done = False
    for i, token in enumerate(encoded_response):
        if code_block_open:
            code_blocks[-1].append(token)
            logprobs_out[-1] += logprobs[i][token].logprob
        if token == 10897:
            code_block_open = not code_block_open
            if code_block_open:
                code_blocks.append([token])
                logprobs_out.append(logprobs[i][token].logprob)
    code_blocks = [tokenizer.decode(code_block) for code_block in code_blocks]
    paired = zip(code_blocks, logprobs_out)
    paired = list(filter(lambda x : "lean4" in x[0], paired))
    paired.sort(key=lambda x : len(x[0]))
    if len(paired) == 0:
        return None, 0.0
    return (paired[-1][0].replace("```", "").replace("lean4", "").strip(), float(paired[-1][1]))


def prove (conjectures : list[str], debug: bool = False) -> list[tuple[str | None, float]]:
    input_texts = []
    for i, conjecture in enumerate(conjectures):

        prompt = f"""Complete the following Lean 4 code:
    # ```lean4
    # import Mathlib
    # import Aesop
    # open Nat
    # theorem problem{i} : {conjecture} := by
        sorry
    # ```

    # Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
    # The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
    # If the conjecture is false, do your best to prove the conjecture anyways."""
        input_texts.append( [
            {"role": "user", "content": prompt}
        ])
    if debug:
        start_time = time.time()
    

    # texts = tokenizer.apply_chat_template(input_texts, tokenize=False, add_generation_prompt=True)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # inputs = tokenizer(texts, padding="longest", return_tensors="pt", verbose=False)

    # inputs = {key: val.to(device) for key, val in inputs.items()}
    sample_params = SamplingParams(max_tokens=8192, logprobs=0, detokenize=False)
    model_results = model.chat(input_texts, sample_params, use_tqdm=True)
    generated_tokens = [out.outputs[0].token_ids for out in model_results]
    logprobs = [out.outputs[0].logprobs for out in model_results]

    # outputs = model.generate(**inputs, max_new_tokens=8192, return_dict_in_generate=True, output_scores=True)

    if debug:
        print (f"Inference took {time.time()-start_time}s")
        start_time = time.time()

    results = []
    for generated_i, transition_i in zip(generated_tokens, logprobs):
        lean_code, logprob = extract_code(generated_i, transition_i, debug=debug)
        if not lean_code:
            results.append((None, 0.0))
            continue
        res = server.run(Command(cmd=lean_code))
        break_out = False
        for msg in res.messages:
            if msg.severity == "error":
                results.append((None, 0.0))
                break_out = True
                break
        if break_out:
            continue
        results.append((lean_code, logprob))
    # print(res)
    if debug:
        print (f"Proof verification took {time.time()-start_time}s")
    return results

if __name__ == "__main__":
    print ("Testing dsprover loop")
    path_name = "/home/timothekasriel/minimo/learning/outputs/2025-06-05/11-55-28/outcomes_0.json"
    num_conj = 32
    batch_size = 3

    with open(path_name) as f:
        outcomes = json.load(f)
    true_theorems = [o["problem_translated"] for o in outcomes][:num_conj]
    true_theorems = list(filter(None, true_theorems))
    start_time = time.time()
    # for t in true_theorems:
    #     prove([t])
    mid_time = time.time()
    # print (f"Proving {num_conj} conjectures took {mid_time - start_time}s unbatched")
    batches = [true_theorems[i:min(i+batch_size, len(true_theorems))] for i in range(0, len(true_theorems), batch_size)]
    bar = tqdm.tqdm(total=len(true_theorems))
    for batch in batches:
        prove(batch, debug=True)
        bar.update(len(batch))
        bar.close()
    # for i in range(0, len(true_theorems), batch_size):
    #     print (f"it {i}/{len(true_theorems)}")
    #     batch = true_theorems[i:min(i+batch_size, len(true_theorems))]
    #     prove(batch, debug=True)
    end_time = time.time()
    print (f"Proving {num_conj} conjectures took {mid_time - start_time}s unbatched and {end_time-mid_time}s batched")

    # results = prove(["(x : Nat) -> x * 1 = x", "(x : Nat) -> (x + 1 = 2) -> 2 * x + 1 = x + 2"], debug=True)
    # results = prove(["(x : Nat) -> x * 1 = x"], debug=True)
    # results = prove(["(x : Nat) -> (x + 1 = 2) -> 2 * x + 1 = x + 2"], debug=True)
    
    # for code, logprob in results:
    #     print (f"Received Code: \n{code}\n\nLogprob: {logprob}")