import torch
from cs336_basics.modules.layers import TransformerLM
from tokenizers import Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TransformerLM(
    vocab_size=10000,
    context_length=256,
    num_layers=4,
    d_model=512,
    num_heads=16,
    d_ff=1344,
    rope_theta=10000.0,
).to(device)

checkpoint = torch.load("model_iter_20000.pt", map_location=device)
state_dict = checkpoint["model_state_dict"]
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

tokenizer = Tokenizer.from_file("tokenizer/vocab_train.json")

def encode(text):
    return tokenizer.encode(text).ids

def decode(ids):
    return tokenizer.decode(ids)

prompt = "how are you?"
input_ids = torch.tensor([encode(prompt)], device=device)

with torch.no_grad():
    output_ids = model.generate(input_ids, temperature=0.6, max_new_tokens=20, top_k=1, eos_token_id='<EOS>')

output_text = decode(output_ids[0].tolist())
print("Input:", prompt)
print("Output:", output_text)
