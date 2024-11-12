# Mit dieser Pythondatei wurde das Pretraining umgesetzt und konnte über die Konsole ausgeführt werden

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


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


# MODELL
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-v0.3", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)


# PEFT-MODELL
model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",

                      "embed_tokens", "lm_head",], # Add for continual pretraining
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = True,   # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# LOAD DATA
import datasets
from datasets import load_dataset
json_paths1 = ['../fullRun/dataGpt/bb_preprocessingData_1.json', 
              '../fullRun/dataGpt/be_preprocessingData_1.json', 
              '../fullRun/dataGpt/be_preprocessingData_2.json', 
              '../fullRun/dataGpt/be_preprocessingData_3.json', 
              '../fullRun/dataGpt/be_preprocessingData_4.json', 
              '../fullRun/dataGpt/bund_preprocessingData_1.json', 
              '../fullRun/dataGpt/bund_preprocessingData_2.json', 
              '../fullRun/dataGpt/bund_preprocessingData_3.json', 
              '../fullRun/dataGpt/bund_preprocessingData_4.json', 
              '../fullRun/dataGpt/bund_preprocessingData_5.json', 
              '../fullRun/dataGpt/bund_preprocessingData_6.json',
              '../fullRun/dataGpt/bund_preprocessingData_7.json']

dataset1 = load_dataset('json', data_files=json_paths1, split='train')

# FORMATING
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
dataset1 = dataset1.map(formatting_prompts_func, batched = True)

# TRAINING
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

output_dir = "./chatGptLorasPretraining"

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset1,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 40,
        gradient_accumulation_steps = 2,

        # Use warmup_ratio and num_train_epochs for longer runs!
        # max_steps = 120,
        warmup_steps = 10,
        warmup_ratio = 0.1,
        num_train_epochs = 10,

        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate = 2e-5,
        embedding_learning_rate = 2e-6,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        save_strategy="epoch",
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
        report_to = "wandb", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()

'''
# SECOND DATA

json_paths2 = ['dataGpt/bb_preprocessingData_1.json', 
              'dataGpt/be_preprocessingData_1.json', 
              'dataGpt/be_preprocessingData_2.json', 
              'dataGpt/be_preprocessingData_3.json', 
              'dataGpt/be_preprocessingData_4.json', 
              'dataGpt/bund_preprocessingData_1.json', 
              'dataGpt/bund_preprocessingData_2.json', 
              'dataGpt/bund_preprocessingData_3.json', 
              'dataGpt/bund_preprocessingData_4.json', 
              'dataGpt/bund_preprocessingData_5.json', 
              'dataGpt/bund_preprocessingData_6.json',
              'dataGpt/bund_preprocessingData_7.json']

dataset2 = load_dataset('json', data_files=json_paths2, split='train')

from datasets import load_dataset
dataset2 = dataset2.map(formatting_prompts_func, batched = True)


from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = alpaca_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 8,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,

        # Use num_train_epochs and warmup_ratio for longer runs!
        max_steps = 120,
        warmup_steps = 10,
        # warmup_ratio = 0.1,
        # num_train_epochs = 1,

        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate = 5e-5,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.00,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()
'''
