from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(30)

model_id = "deepseek-ai/DeepSeek-Prover-V2-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)


chat = [[
    {"role": "user", "content": "testing 123"},
],
    {"role": "user", "content": "testing 12345"},
]

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
inputs = [tokenizer.apply_chat_template(c, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device) for c in chat]

outputs = model.generate(**inputs, max_new_tokens=8192)
print(tokenizer.batch_decode(outputs))