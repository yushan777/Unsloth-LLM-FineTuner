from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import subprocess
import os

# === Model settings ===
max_seq_length = 2048
dtype = None  # Auto-detect (float16/bfloat16 depending on your GPU)
load_in_4bit = False

# === Load model ===
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "models/unsloth/Meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# === Load your dataset ===
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = load_dataset("json", data_files="dataset.jsonl")["train"]
dataset = dataset.map(formatting_prompts_func, batched=True)

# ============================
# Fine-tune params
# ============================
learn_rate = 5e-5
max_epochs = 10
warmup_steps = 5
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
output_dir = "outputs"

# ============================
# calulate steps per epoch
# ============================
num_samples = len(dataset)
effective_batch_size =  per_device_train_batch_size * gradient_accumulation_steps
steps_per_epoch = (num_samples + effective_batch_size - 1) // effective_batch_size
print(f" >> Number of training samples: {num_samples}")
print(f" >> Effective batch size: {effective_batch_size}")
print(f" >> Steps per epoch: {steps_per_epoch}")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,

    args = SFTConfig(
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_steps = warmup_steps,
        num_train_epochs = max_epochs,
        learning_rate = learn_rate,        
        optim = "adamw_8bit",
        weight_decay = 0.01,
        save_steps = 1000,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
        logging_dir = os.path.join(output_dir, "logs"),
        logging_steps = 5,        
        report_to = ["tensorboard"],  # Enables TensorBoard logs
    ),
)

trainer.train()

# to resume from the laste checkpoint saved, use:
# trainer.train(resume_from_checkpoint=True)

# to resume from a specific checkpoint, use:
# trainer.train(resume_from_checkpoint="outputs/checkpoint-1500")


# === Save merged model ===
save_dir = "merged_model"
model.save_pretrained_merged(
    save_directory = save_dir,
    tokenizer = tokenizer,
    save_method = "merged_16bit",
)

print(f"✅ Merged model saved at: {save_dir}")

# === Convert to Unquantized GGUF ===
llama_cpp_path = "./llama.cpp"
convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
gguf_output = os.path.join(save_dir, "merged_model.gguf")

try:
    print(f"🚀 Converting merged model to Unquantized GGUF: {gguf_output}")
    subprocess.run([
        "python3",
        convert_script,
        save_dir,
        "--outfile", gguf_output
    ], check=True)
    print(f"✅ Unquantized GGUF model saved at: {gguf_output}")
    
    # === Quantize ===
    # Other quantization types you can choose:
    # "Q4_0"    # 4-bit quantization, older method, smaller size, lower quality
    # "Q4_1"    # 4-bit quantization, better than Q4_0
    # "Q4_K"    # 4-bit K-bit quantization, balanced quality/size
    # "Q4_K_M"  # 4-bit K multiblock quantization (very popular, good quality/size balance)
    # "Q5_0"    # 5-bit quantization, better than 4-bit, medium size
    # "Q5_1"    # 5-bit quantization with better handling of outliers
    # "Q6_K"    # 6-bit quantization, very good quality, larger size
    # "Q8_0"    # 8-bit quantization, almost no loss in quality, largest size

    # === Quantize to both Q8_0 and Q4_K_M ===
    quant_types = ["Q8_0", "Q4_K_M"]
    quant_bin = os.path.join(llama_cpp_path, "build/bin/llama-quantize")
    
    for quant_type in quant_types:
        quant_output = os.path.join(save_dir, f"merged_model_epoch-{max_epochs}_{quant_type}.gguf")
        
        print(f"🚀 Quantizing GGUF model to: {quant_output} using {quant_type}")
        subprocess.run([
            quant_bin,
            gguf_output,
            quant_output,
            quant_type
        ], check=True)
        print(f"✅ Quantized model saved at: {quant_output}")

except subprocess.CalledProcessError as e:
    print(f"❌ Process failed: {e}")

print("🎉 Training and quantization complete!")
print(f"📁 Available models in {save_dir}:")
