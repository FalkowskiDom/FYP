from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "Outputs/qwen_logdom"

# Loads the tokenizer for the base Qwen model.
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Loads the base Qwen model onto the available device.
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto"
)

# Loads the fine-tuned LoRA adapter onto the base model.
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

# Sets the model to evaluation mode.
model.eval()

# Creates the test prompt for the model.
prompt = """### Instruction:
Explain Windows Event ID 403 in one short paragraph for a security analyst. Do not output JSON, XML, or raw log data.

### Response:
"""

# Tokenises the prompt into model input format.
inputs = tokenizer(prompt, return_tensors="pt")

# Moves the input tensors to the same device as the model.
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generates a response without updating model weights.
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

# Converts the model output tokens back into text.
text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Prints the generated response.
print(text)