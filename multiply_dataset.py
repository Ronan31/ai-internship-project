# Author: Ronan Cantin
# Creation Date: 13/11/23
# Description: Multiply dataset

import re
import sys
import torch
import fire
import json
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter

################################Global variables#############################################
base_model_used = ""
experiment_used = ""
tokenizer = None
model = None

################################Functions#############################################
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def Initialise_model(
        base_model,
        experiment,
        load_8bit,
):
    global tokenizer, model, base_model_used, experiment_used

    if base_model != base_model_used or experiment != experiment_used:
        base_model_used = base_model
        experiment_used = experiment
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        if device == "cuda":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(
                model,
                experiment,
                torch_dtype=torch.float16,
            )
        elif device == "mps":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
            model = PeftModel.from_pretrained(
                model,
                experiment,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                base_model, device_map={"": device}, low_cpu_mem_usage=True
            )
            model = PeftModel.from_pretrained(
                model,
                experiment,
                device_map={"": device},
            )

        # unwind broken decapoda-research config
        # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        # model.config.bos_token_id = 1

        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        # model.config.eos_token_id = tokenizer.eos_token_id

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)


def generate_response(
        instruction,
        input_data,
        temperature,
        top_p,
        top_k,
        num_beams,
        repetition_penalty,
        max_new_tokens,
        **kwargs,
):
    global tokenizer, model

    # repetition_penalty = 1.2
    # Prompt generation
    prompter = Prompter('')
    prompt = prompter.generate_prompt(instruction, input_data)
    # inputs = tokenizer(prompt, return_tensors="pt")
    # input_ids = inputs["input_ids"].to(device)
    encoding = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt", max_length=1024)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        num_beams=num_beams,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        **kwargs,
    )
    with torch.inference_mode():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)

    match = re.search(r'Response:(.*?)(?=\n\n|\Z)', output, re.DOTALL)
    if match:
        response = match.group(1).strip()
        return response
    else:
        return output


################################Main#############################################
def main(
        dataset_path,
        new_dataset_path,
        nb_multiply
):
    # Load the input dataset from the provided path
    with open(dataset_path, 'r') as file:
        data = json.load(file)

    # Initialize the model and tokenizer
    Initialise_model(base_model="./Base_models/llama-7b-hf", experiment="./Experiments/alpaca-lora-7b", load_8bit=True)

    new_dataset = []

    for i in range(nb_multiply):

        for example in data:
            
            instruction = example["instruction"]
            input_data = example["input"]
            output_data = example["output"]

            old_example = {
                "instruction": instruction,
                "input": input_data,
                "output": output_data
            }

            new_instruction_prompt = (f"Reformulate the following instruction and input in one sentence:\n"
                                      f"instruction: {instruction}\n"
                                      f"input: {input_data}")

            new_instruction = generate_response(new_instruction_prompt, "", temperature=0.1, top_p=0.2, top_k=40,
                                                num_beams=1,
                                                repetition_penalty=1.2, max_new_tokens=128)

            new_output_prompt = (f"Reformulate the following output in one sentence:\n"
                                 f"{output_data}")

            new_output = generate_response(new_output_prompt, "", temperature=0.1, top_p=0.2, top_k=40,
                                           num_beams=1,
                                           repetition_penalty=1.2, max_new_tokens=128)

            new_example = {
                "instruction": new_instruction,
                "input": "",
                "output": new_output
            }

            # Check if the example already exists in the new dataset
            exists_in_new_dataset = any(
                e["instruction"] == new_example["instruction"] and e["input"] == new_example["input"] and
                e["output"] == new_example["output"]
                for e in data
            )

            # Add the generated example to the new dataset if it doesn't exist already
            if not exists_in_new_dataset:
                print(f"Example:\n"
                      f"{old_example}\n"
                      f"New_example:\n"
                      f"{new_example}")

                new_dataset.append(old_example)
                new_dataset.append(new_example)
                
        data = new_dataset
        
    # Save the new dataset with generated responses
    with open(new_dataset_path, 'w') as file:
        json.dump(new_dataset, file, indent=2)

if __name__ == "__main__":
    fire.Fire(main)
