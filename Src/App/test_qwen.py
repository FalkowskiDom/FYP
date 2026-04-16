from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "Outputs/qwen_logdom"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

prompt = """### Instruction:
Explain Windows Event ID 403 in one short paragraph for a security analyst. Do not output JSON, XML, or raw log data.

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)