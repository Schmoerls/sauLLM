# produce bad outputs for the reward_trainer using an old and bad model

from huggingface_hub import login
import os
import json
import torch
from unsloth import FastLanguageModel
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

login("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")
os.environ["WANDB_DISABLED"] = "true"

json_paths = ['../fullRun/dataGpt/bb_preprocessingData_1.json', 
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


def process_json_file(input_file_path, output_file_path, model, tokenizer, device, max_length=2048):
    """
    Processes each entry in a JSON file one-by-one: for each input, generates an output using the model.
    
    Args:
        input_file_path (str): Path to the input JSON file.
        output_file_path (str): Path where the output JSON will be saved.
        model (PeftModel): The loaded model for inference.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        device (torch.device): The device to run the model on.
        max_length (int): Maximum length for generated outputs.
    
    Returns:
        None
    """
    # Load the input JSON data
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # Assuming data is a list of {"input": "...", "output": "..."} dictionaries
    
    generated_data = []
    
    # Process each entry individually
    for entry in tqdm(data, desc=f"Processing {input_file_path}"):
        input_text = entry.get("input", "")
        
        # Tokenize the input
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length
        ).to(device)
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                repetition_penalty=1.2,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # Decode the output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Append to generated_data
        generated_data.append({
            "input": input_text,
            "output": generated_text
        })
    
    # Save the generated data to the output JSON file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)
    
    print(f"Saved generated data to {output_file_path}")


def create_output_files(output_directory):    
    os.makedirs(output_directory, exist_ok=True)
    
    for json_path in json_paths:
        if not os.path.exists(json_path):
            print(f"Input file does not exist: {json_path}")
            continue
        
        base_filename = os.path.splitext(os.path.basename(json_path))[0]
        output_file = os.path.join(output_directory, f"{base_filename}_generated.json")

        if os.path.exists(output_file):
            print(f"Output file already exists: {output_file}")
            continue
        
        try:
            print(f"Starting processing of {json_path}")
            process_json_file(
                input_file_path=json_path,
                output_file_path=output_file,
                model=peft_model,
                tokenizer=tokenizer,
                device=device,
                max_length=2048   
            )
            print(f"Successfully processed {json_path}")
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            continue


# Model and Tokenizer Loading
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-v0.3", 
    local_files_only=True, 
    max_seq_length=2048, 
    dtype=None, 
    load_in_4bit=True 
)

peft_model = PeftModel.from_pretrained(base_model, "../fullRun/lorasSFTTrainer/checkpoint-1000/") 

FastLanguageModel.for_inference(peft_model)
device = torch.device("cuda")

create_output_files("bad_data")
