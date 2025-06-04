from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import convert_to_lean
from lean_interact import LeanREPLConfig, AutoLeanServer, Command
torch.manual_seed(30)

model_id = "deepseek-ai/DeepSeek-Prover-V2-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
server = AutoLeanServer(LeanREPLConfig())


def extract_code (encoded_response, logprobs) -> tuple[str, float] | tuple[None,None]:
    code_blocks: list[list[int]] = []
    logprobs_out = []
    code_block_open = False
    intro_done = False
    for i, token in enumerate(encoded_response):
        if code_block_open:
            code_blocks[-1].append(token)
            logprobs_out[-1] += logprobs[i]
        if token is 10897:
            code_block_open = not code_block_open
            if code_block_open:
                code_blocks.append([token])
                logprobs_out.append(logprobs[i])
    code_blocks = [tokenizer.decode(code_block) for code_block in code_blocks]
    paired = zip(code_blocks, logprobs_out)
    paired = list(filter(lambda x : "lean4" in x[0], paired))
    paired.sort(key=lambda x : len(x[0]))
    if len(paired) == 0:
        return None, None
    return (paired[-1][0].replace("lean4", "").strip(), paired[-1][1])


def prove (problem_name : str, conjecture : str) -> tuple[str | None, float | None]:
    prompt = f"""Complete the following Lean 4 code:
# ```lean4
# import Mathlib
# import Aesop
# open Nat
# theorem {problem_name} : conjecture := by
    sorry
# ```

# Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
# The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""
    chat = [
        {"role": "user", "content": prompt}
    ]
    input = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(input, max_new_tokens=8192, output_scores=True)
    transition_scores = model.compute_transition_scores (outputs.sequences, outputs.scores)
    decoded_out = tokenizer.batch_decode(outputs)[0]
    if conjecture not in decoded_out:
        return None, None
    lean_code, logprob = extract_code(outputs, transition_scores)
    if not lean_code:
        print(decoded_out)
        return None, None
    res = server.run(Command(lean_code))
    print(res)
    return lean_code, logprob