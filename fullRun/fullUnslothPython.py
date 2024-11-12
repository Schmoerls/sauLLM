# Umsetzung des Finetuning Prozesses als Pythondatei, um diese mit nohup in der Konsole auszuführen
from huggingface_hub import login, logout

login("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")

import os
import wandb

# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="sauLLM"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="checkpoint"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"

# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-v0.3", 
    max_seq_length = 2048, # Choose any! We auto support RoPE Scaling internally!
    dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
)

model_with_peft = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


def formatting_func_1(example):
    text = f"### Question: {example['input']}\n ### Answer: {example['output']}"
    return text

max_length = 8000 # kann verändert werden

def generate_and_tokenize_prompt(prompt):
    result = tokenizer(
        formatting_func_1(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

import datasets
from datasets import load_dataset
json_paths = ['data/bb.json', 
              'data/be.json', 
              'data/bund.json', 
              'data/he.json', 
              'data/hh.json', 
              'data/mv.json', 
              'data/rp.json', 
              'data/sh.json', 
              'data/sl.json', 
              'data/st.json', 
              'data/th.json']

dataset = load_dataset('json', data_files=json_paths, split='train')

prompt = """

### Question:
{}

### Answer:
{}
"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(example):
    exinput = example['input']
    exoutput = example['output']
    texts = []

    for i,j  in zip(exinput, exoutput):
        text = prompt.format(i,j) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = dataset.map(formatting_prompts_func, batched = True)


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

print("#" * 120)
print("Running training with...")
print(len(dataset), " training tokens")
print("#" * 120)

max_length = 8000
output_dir = "./lorasConfig4"

trainer = SFTTrainer(
    model = model_with_peft,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 10,
        gradient_accumulation_steps = 2,
        warmup_steps = 40,
        num_train_epochs = 12,
        # max_steps = 60, # Set num_train_epochs = 1 for full training runs
        learning_rate = 2e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
    	save_strategy = "steps",
    	save_steps = 100,
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
        report_to="wandb",
    ),
)


# Training

print("#" * 120)
print("Running training with...")
print(len(dataset), " training tokens")
print("#" * 120)

trainer_stats = trainer.train()
