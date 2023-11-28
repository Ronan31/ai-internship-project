# Author: Ronan Cantin
# Creation Date: 13/11/23
# Description: Finetune, Generate, Manage


################################Import#############################################
import os
import re
import sys
import io
import fire
import torch
import transformers
import json
import tempfile
import socket
import shutil
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
from typing import List
from peft import PeftModel
from datasets import load_dataset
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

################################Global variables#############################################
base_model_used = ""
experiment_used = ""
tokenizer = None
model = None
path_new_dataset = "./Datasets/new_dataset.json"
path_database = "./database.json"
Experiments_list = ""
Datasets_list = ""

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

# FUNCTION_NAME = Initialise_model
# SUMMARY = Initialize the language model and tokenizer based on the specified parameters.
# PARAM base_model = The base language model to be used.
# PARAM experiment = The experiment name.
# PARAM load_8bit = Flag indicating whether to load the model in 8-bit mode.
# RETURN = None (Global variables are updated in-place).
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

# FUNCTION_NAME = generate_response
# SUMMARY = Generate a response based on the given instruction, input data, and generation parameters.
# PARAM instruction = The instruction for response generation.
# PARAM input_data = Additional input data for context.
# PARAM temperature = Temperature parameter for controlling randomness in generation.
# PARAM top_p = Top-p parameter for nucleus sampling.
# PARAM top_k = Top-k parameter for top-k sampling.
# PARAM num_beams = Number of beams for beam search.
# PARAM repetition_penalty = Repetition penalty for discouraging repetitive tokens.
# PARAM max_new_tokens = Maximum number of tokens to generate.
# RETURN = The generated response.
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


# FUNCTION_NAME = compute_f1
# SUMMARY = Calculate the F1 score based on prediction and ground truth text.
# PARAM prediction = The generated text for comparison.
# PARAM ground_truth = The reference text for comparison.
# RETURN = The calculated F1 score as a float value.
def compute_f1(prediction, ground_truth):
    prediction_tokens = prediction.lower().split()
    ground_truth_tokens = ground_truth.lower().split()
    common_tokens = set(prediction_tokens) & set(ground_truth_tokens)
    if len(common_tokens) == 0:
        return 0
    precision = len(common_tokens) / len(prediction_tokens)
    recall = len(common_tokens) / len(ground_truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


# FUNCTION_NAME = evaluation_model
# SUMMARY = Evaluate a model using specified parameters and a custom dataset.
# PARAM base_model = The base language model to use for evaluation.
# PARAM output_dir = The directory where fine-tuned model checkpoints are saved.
# PARAM datasets = The custom dataset for evaluation.
# PARAM use_8bit = Flag indicating whether to use 8-bit training.
# RETURN = The average F1 score for the entire custom dataset, along with the top 3 and bottom 3 examples by F1 score.
def evaluation_model(
        base_model,
        experiment,
        datasets,
):
    global tokenizer, model

    print(
        f"\nEvaluation model with params:\n"
        f"base_model: {base_model}\n"
        f"lora_weights: {experiment}\n"
        f"eval_datasets: {datasets}\n"
    )

    Initialise_model(base_model, experiment, True)

    if datasets.endswith(".json"):

        with open(datasets, 'r') as json_file:
            custom_dataset = json.load(json_file)

        f1_scores = []
        examples_with_f1 = []
        evaluation_examples = ""
        num_examples = len(custom_dataset)
        print(f"\n Num Examples = {num_examples}")

        # Loop through each example in the custom dataset
        for data_point in custom_dataset:
            # Extract data for this example
            instruction = data_point["instruction"]
            input_data = data_point.get("input", "")  # If "input" is not specified, use an empty string by default
            expected_output = data_point["output"]

            # Use the evaluate function to generate output based on instruction and input
            # output = evaluate(tokenizer, model, instruction, input_data, 0.1, 0.8, 40, 1, 64)
            output = generate_response(instruction, input_data, 0.1, 0.2, 40, 1, 1.2, 256)
            # Calculate the F1 score for this example by comparing the generated output with the expected output
            f1_score = compute_f1(output, expected_output)

            # Add the F1 score to the list of F1 scores
            f1_scores.append(f1_score)

            # Add the example with its F1 score to the list of examples
            examples_with_f1.append((data_point, f1_score, output))

            example = ""
            # Display results for this example
            example += "\n##########################################################################################################\n"
            example += f"\n- Instruction: {instruction}"
            example += f"- Input: {input_data}"
            example += f"- Expected Output: {expected_output}"
            example += f"- Generated Output: {output}"
            example += f"- F1 Score: {f1_score}\n"
            print(example)
            evaluation_examples += example
        # Calculation of the average F1 score
        average_f1_score = sum(f1_scores) / len(f1_scores)

        # Sort the examples according to their F1 scores (from highest to lowest)
        examples_with_f1.sort(key=lambda x: x[1], reverse=True)

        # Select the three examples with the highest F1 scores
        top_3_examples = examples_with_f1[:3]

        # Select the three examples with the lowest F1 scores
        bottom_3_examples = examples_with_f1[-3:]

        print("Top 3 Examples with Highest F1 Scores:")
        for example, f1_score, generated_output in top_3_examples:
            instruction_data = example["instruction"]
            input_data = example.get("input", "")
            output_data = example["output"]
            print(f"\nInstruction: {instruction_data}\n")
            print(f"Input: {input_data}\n")
            print(f"Output: {output_data}\n")
            print(f"Generated output: {generated_output}\n")
            print(f"F1 Score: {f1_score}")

        print("Bottom 3 Examples with Lowest F1 Scores:")
        for example, f1_score, generated_output in bottom_3_examples:
            instruction_data = example["instruction"]
            input_data = example.get("input", "")
            output_data = example["output"]
            print(f"\nInstruction: {instruction_data}\n")
            print(f"Input: {input_data}\n")
            print(f"Output: {output_data}\n")
            print(f"Generated output: {generated_output}\n")
            print(f"F1 Score: {f1_score}")

        print(
            "\n##########################################################################################################\n")
        print(
            f"===> Average F1 Score for the Entire Custom Dataset: {average_f1_score} i.e. {average_f1_score * 100}% ")
        print(
            "\n##########################################################################################################\n")

        return average_f1_score, top_3_examples, bottom_3_examples, evaluation_examples

    else:
        print("Unsupported Dataset")


# FUNCTION_NAME = finetune
# SUMMARY = Fine-tune a language model with provided data and hyperparameters.
# PARAM base_model = The base language model to fine-tune.
# PARAM data_path = The path to the data used for fine-tuning.
# PARAM output_dir = The directory to save fine-tuned model checkpoints.
# RETURN = Training output and statistics.
def finetune(
        # model/data params
        base_model,
        data_path,
        output_dir,
        # training hyperparams
        batch_size,
        micro_batch_size,
        num_epochs,
        learning_rate,
        cutoff_len,
        val_set_size,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        add_eos_token: bool = False,
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=1 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #    lambda self, *_, **__: get_peft_model_state_dict(
    #        self, old_state_dict()
    #    )
    # ).__get__(model, type(model))

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #    model = torch.compile(model)

    stdout_buffer = io.StringIO()
    sys.stdout = stdout_buffer

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    training_output = stdout_buffer.getvalue()

    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(training_output)
    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )
    return training_output


# FUNCTION_NAME = count_instructions
# SUMMARY = Count the number of instructions in a JSON file.
# PARAM file_path = The path to the JSON file.
# RETURN = The number of instructions.
def count_instructions(file_path):
    try:
        # Load the JSON file containing instructions
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Count the number of instructions in the file
        number_of_instructions = len(data)

        return number_of_instructions

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return 0  # Or you can choose to raise the exception again
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {file_path}: {e}")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0


# FUNCTION_NAME = finetune_eval
# SUMMARY = Perform fine-tuning and evaluation for a language model.
# PARAM base_model = The base language model for fine-tuning.
# PARAM train_dataset = The training dataset for fine-tuning.
# PARAM eval_dataset = The evaluation dataset.
# PARAM experiment = The experiment name.
# PARAM batch_size = Batch size for fine-tuning.
# PARAM micro_batch_size = Micro-batch size.
# PARAM num_epochs = Number of training epochs.
# PARAM learning_rate = Learning rate.
# PARAM cutoff_len = Cutoff length.
# PARAM val_set_size = Validation set size as a percentage.
# RETURN = Training and evaluation results.
def finetune_eval(
        # model/data params
        base_model,
        train_dataset,
        eval_dataset,
        experiment,

        # training hyperparams
        batch_size,
        micro_batch_size,
        num_epochs,
        learning_rate,
        cutoff_len,
        val_set_size,  # percent
        log,

):
    # lora hyperparams
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
    # llm hyperparams
    train_on_inputs = True  # if False, masks out inputs in loss
    add_eos_token = False
    group_by_length = False  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project = ""
    wandb_run_name = ""
    wandb_watch = ""  # options: false | gradients | all
    wandb_log_model = ""  # options: false | true
    resume_from_checkpoint = None  # either training checkpoint or final adapter
    prompt_template_name = "alpaca"  # The prompt template to use, will default to alpaca.

    output_dir = "./Experiments/" + experiment

    # Check if the directory already exists
    if experiment == "":
        print("The Experiment name was not specified.")
    elif os.path.isdir(output_dir):
        print(f"The Experiment name '{experiment}' already exists. Please choose another name or delete the old one.")
    else:
        train_dataset_size = count_instructions(train_dataset)
        if eval_dataset is not None:
            eval_dataset_size = count_instructions(eval_dataset)
        train_val_set_size = int((val_set_size / 100) * train_dataset_size)
        training_data = finetune(base_model, train_dataset, output_dir, batch_size, micro_batch_size, num_epochs,
                                 learning_rate, cutoff_len, train_val_set_size, lora_r, lora_alpha, lora_dropout,
                                 lora_target_modules, train_on_inputs, add_eos_token, group_by_length, wandb_project,
                                 wandb_run_name, wandb_watch, wandb_log_model, resume_from_checkpoint,
                                 prompt_template_name)

        # Use a regular expression to extract the values of loss and eval_loss
        loss_values = re.findall(r"'loss': ([\d.]+)", training_data)
        eval_loss_values = re.findall(r"'eval_loss': ([\d.]+)", training_data)

        # Extract 'epoch' values associated with 'loss' and 'eval_loss'.
        loss_data = re.findall(r"'loss': ([\d.]+), 'learning_rate': [\d.e-]+, 'epoch': ([\d.]+)", training_data)
        eval_loss_data = re.findall(
            r"'eval_loss': ([\d.]+), 'eval_runtime': [\d.e-]+, 'eval_samples_per_second': [\d.]+, 'eval_steps_per_second': [\d.]+, 'epoch': ([\d.]+)",
            training_data)

        # Convert values to float
        loss_values = [float(val) for val in loss_values]
        eval_loss_values = [float(val) for val in eval_loss_values]
        loss_epochs = [float(epoch) for _, epoch in loss_data]
        eval_loss_epochs = [float(epoch) for _, epoch in eval_loss_data]

        # Get the first and last values of loss and eval_loss
        first_loss = loss_values[0]
        last_loss = loss_values[-1]
        diff_loss = first_loss - last_loss

        first_eval_loss = eval_loss_values[0]
        last_eval_loss = eval_loss_values[-1]
        diff_eval_loss = first_eval_loss - last_eval_loss

        if eval_dataset is not None:
            f1_score, top_3_examples, bottom_3_examples, evaluation_examples = evaluation_model(base_model, output_dir,
                                                                                                eval_dataset)

        output = ""
        output += f"Experiment : {experiment}\n\n"
        output += f"Experiment Parameters:\n"
        output += f"base_model: {base_model}\n"
        output += f"train_dataset: {train_dataset}\n"
        output += f"train_dataset_size: {train_dataset_size}\n"
        if eval_dataset is not None:
            output += f"eval_dataset: {eval_dataset}\n"
            output += f"eval_dataset_size: {eval_dataset_size}\n"
        output += f"batch_size: {batch_size}\n"
        output += f"micro_batch_size: {micro_batch_size}\n"
        output += f"num_epochs: {num_epochs}\n"
        output += f"learning_rate: {learning_rate}\n"
        output += f"cutoff_len: {cutoff_len}\n"
        output += f"val_set_size: {train_val_set_size}\n"
        output += f"lora_r: {lora_r}\n"
        output += f"lora_alpha: {lora_alpha}\n"
        output += f"lora_dropout: {lora_dropout}\n"
        output += f"lora_target_modules: {lora_target_modules}\n"
        output += f"train_on_inputs: {train_on_inputs}\n"
        output += f"add_eos_token: {add_eos_token}\n"
        output += f"group_by_length: {group_by_length}\n\n"
        output += f"Experiment Results :\n"
        output += f"first_loss : {first_loss}\n"
        output += f"last_loss : {last_loss}\n"
        output += f"diff_loss : {diff_loss}\n"
        output += f"first_eval_loss : {first_eval_loss}\n"
        output += f"last_eval_loss : {last_eval_loss}\n"
        output += f"diff_eval_loss : {diff_eval_loss}\n"
        if eval_dataset is not None:
            output += f"F1 Score : {f1_score} i.e. {f1_score * 100}%\n"

            output += "\n\nTop 3 Examples with Highest F1 Scores:\n"
            for example, f1_score, generated_output in top_3_examples:
                instruction_data = example["instruction"]
                input_data = example.get("input", "")
                output_data = example["output"]
                output += f"\nInstruction: {instruction_data}\n"
                output += f"Input: {input_data}\n"
                output += f"Output: {output_data}\n"
                output += f"Generated output: {generated_output}\n"
                output += f"F1 Score: {f1_score}\n"

            output += "\nBottom 3 Examples with Lowest F1 Scores:\n"
            for example, f1_score, generated_output in bottom_3_examples:
                instruction_data = example["instruction"]
                input_data = example.get("input", "")
                output_data = example["output"]
                output += f"\nInstruction: {instruction_data}\n"
                output += f"Input: {input_data}\n"
                output += f"Output: {output_data}\n"
                output += f"Generated output: {generated_output}\n"
                output += f"F1 Score: {f1_score}\n"

        print(output)

        # Create a temporary folder
        temp_dir = tempfile.mkdtemp()

        # Plot the loss and eval_loss values
        plt.figure(figsize=(10, 6))
        plt.plot(loss_epochs, loss_values, label='Training Loss', marker='o')
        plt.plot(eval_loss_epochs, eval_loss_values, label='Evaluation Loss', marker='x')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Loss')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(temp_dir, 'loss_plot.png'))  # Save the plot as an image
        plt.close()

        # Get full image path
        image_path = os.path.join(temp_dir, 'loss_plot.png')

        if log == True:
            log_path = f"./Experiments_log/{experiment}"
            # Check if the directory already exists
            if os.path.exists(log_path):
                # If it exists, create a new directory with a different name
                index = 1
                while os.path.exists(log_path):
                    log_path = f"./Experiments_log/{experiment}_{index}"
                    index += 1
            os.makedirs(log_path)
            result_path = os.path.join(log_path, "Results.txt")
            with open(result_path, "w") as text:
                text.write(output)
            if eval_dataset is not None:
                evaluation_path = os.path.join(log_path, "Evaluation.txt")
                with open(evaluation_path, "w") as text:
                    text.write(evaluation_examples)
            image = Image.open(image_path)
            chemin_image = os.path.join(log_path, "loss_plot.png")
            image.save(chemin_image)

        return output, image_path

# FUNCTION_NAME = generate
# SUMMARY = Generate responses using a language model based on given parameters.
# PARAM base_model = The base language model for response generation.
# PARAM experiment = The experiment name.
# PARAM instruction = The instruction for response generation.
# PARAM input_data = Additional input data for context (optional).
# PARAM temperature = Temperature parameter for controlling randomness in generation.
# PARAM top_p = Top-p parameter for nucleus sampling.
# PARAM top_k = Top-k parameter for top-k sampling.
# PARAM num_beams = Number of beams for beam search.
# PARAM repetition_penalty = Repetition penalty for discouraging repetitive tokens.
# PARAM max_new_tokens = Maximum number of tokens to generate.
# RETURN = Generated output response.
def generate(
        base_model,
        experiment,
        instruction,
        input_data,
        temperature,
        top_p,
        top_k,
        num_beams,
        repetition_penalty,
        max_new_tokens,
):
    global tokenizer, model

    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='./Base_models/llama-7b-hf'"

    Initialise_model(base_model, experiment, True)

    output = generate_response(instruction, input_data, temperature, top_p, top_k, num_beams, repetition_penalty,
                               max_new_tokens)

    print(f"Instruction: {instruction}\n")
    if input_data != "":
        print(f"Input: {input_data}\n")

    print(f"Output Generated: {output}\n")

    return output


################################Interface functions#############################################

# FUNCTION_NAME = add_good_answer
# SUMMARY = Add a good answer example to an existing dataset.
# PARAM instruction = The instruction for the example.
# PARAM input_data = Additional input data for context.
# PARAM answer = The correct output answer.
# RETURN = None (Dataset is updated in-place).
def add_good_answer(
        instruction,
        input_data,
        answer,
):
    with open(path_new_dataset, 'r', encoding='utf-8') as fichier_json:
        dataset_existante = json.load(fichier_json)

    nouvel_exemple = {
        "instruction": instruction,
        "input": input_data,
        "output": answer
    }

    for exemple in dataset_existante:
        if exemple["instruction"] == instruction and exemple["input"] == input_data and exemple["output"] == answer:
            return

    dataset_existante.append(nouvel_exemple)

    with open(path_new_dataset, 'w', encoding='utf-8') as fichier_json:
        json.dump(dataset_existante, fichier_json, ensure_ascii=False, indent=4)

# FUNCTION_NAME = get_database
# SUMMARY = Retrieve information from the database file.
# RETURN = A tuple containing formatted strings for datasets and experiments.
def get_database():
    global path_database

    try:
        with open(path_database, "r") as file:
            data = json.load(file)

        dataset_database = [f"Name: {desc['name']}  Description: {desc['description']}" for desc in data if
                            desc['type'] == "dataset"]

        experiment_database = [f"Name: {desc['name']}   Description: {desc['description']}" for desc in data
                               if desc['type'] == "experiment"]

        return '\n'.join(dataset_database), '\n'.join(experiment_database)

    except Exception as e:
        print(f"Get_database error : {str(e)}")

# FUNCTION_NAME = update_database
# SUMMARY = Update the database with new datasets and experiments.
# PARAM dataset_list = List of new datasets.
# PARAM experiment_list = List of new experiments.
# RETURN = A tuple containing formatted strings for datasets and experiments.
def update_database(dataset_list, experiment_list):
    try:
        global path_database

        with open(path_database, "r") as file:
            data = json.load(file)

        for dataset in dataset_list:
            if not any(desc["type"] == "dataset" and desc["name"] == dataset for desc in data):
                data.append({"type": "dataset", "name": dataset, "description": ""})

        for experiment in experiment_list:
            if not any(desc["type"] == "experiment" and desc["name"] == experiment for desc in data):
                data.append({"type": "experiment", "name": experiment, "description": ""})

        data = [desc for desc in data if
                (desc["type"] == "dataset" and desc["name"] in dataset_list) or
                (desc["type"] == "experiment" and desc["name"] in experiment_list)]

        with open(path_database, "w") as file:
            json.dump(data, file)

        dataset_database, experiment_database = get_database()

        return dataset_database, experiment_database

    except Exception as e:
        print(f"Update_database error : {str(e)}")


################################Main#############################################
# FUNCTION_NAME = main
# SUMMARY = Entry point for the Gradio interface.
# PARAM share_gradio = Allows to create a public link.
def main(
        share_gradio: bool = False,
):
    global Experiments_list, Datasets_list
    # Get the machine's IP address
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    # Get lists
    Datasets_path = "./Datasets/"
    Datasets_list = [os.path.join(Datasets_path, f) for f in os.listdir(Datasets_path) if
                     os.path.isfile(os.path.join(Datasets_path, f))]

    Base_models_path = "./Base_models"
    Base_models_list = [os.path.join(Base_models_path, subdir) for subdir in next(os.walk(Base_models_path))[1]]

    Experiments_path = "./Experiments"
    Experiments_list = [os.path.join(Experiments_path, subdir) for subdir in next(os.walk(Experiments_path))[1]]

    dataset_database, experiment_database = update_database(Datasets_list, Experiments_list)

    with gr.Blocks() as interface:
        with gr.Tab(label="Finetune"):
            gr.Markdown(
                """
                🛠️🎯 Finetune 🛠️🎯\n
                This interface allows to finetune a LLM with a custom dataset and to evaluated the result with the F1-score classification metric.\n
                Based on codes : [Finetune](https://github.com/tloen/alpaca-lora) and [Evaluation](https://github.com/gururise/AlpacaDataCleaned).\n
                Author: Ronan Cantin\n
                """)
            with gr.Row():
                with gr.Column(scale=1):
                    app1_dpd_base_model = gr.Dropdown(
                        choices=Base_models_list,
                        label="base_model"
                    )
                    app1_dpd_train_dataset = gr.Dropdown(
                        choices=Datasets_list,
                        label="train_dataset"
                    )
                    app1_dpd_eval_dataset = gr.Dropdown(
                        choices=[("None", None)] + Datasets_list,
                        label="eval_dataset",
                        info="Select 'None' not to perform the evaluation"
                    )
                    app1_tb_experiment = gr.Textbox(
                        lines=1,
                        label="experiment",
                        placeholder="Test"
                    )
                    app1_slid_batch_size = gr.Slider(
                        minimum=1,
                        maximum=512,
                        value=128,
                        label="batch_size"
                    )
                    app1_slid_micro_batch_size = gr.Slider(
                        minimum=1,
                        maximum=128,
                        value=4,
                        label="micro_batch_size"
                    )
                    app1_slid_epochs = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=3,
                        step=1,
                        label="num_epochs"
                    )
                    app1_slid_learning_rate = gr.Slider(
                        minimum=0,
                        maximum=0.01,
                        value=3e-4,
                        step=1e-4,
                        label="learning_rate"
                    )
                    app1_slid_cutoff_len = gr.Slider(
                        minimum=1,
                        maximum=512,
                        value=256,
                        label="cutoff_len"
                    )
                    app1_slid_val_set_size = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=20,
                        label="val_set_size(%)"
                    )
                    app1_cb_log = gr.Checkbox(
                        value=True,
                        label="Log",
                        info="Check box to save results"
                    )
                    app1_btn_start_finetune = gr.Button("Start finetune")

                with gr.Column(scale=2):
                    app1_tb_output = gr.Textbox(
                        lines=22,
                        label="Output",
                    )
                    app1_tb_loss_graph = gr.Image(
                        type="filepath",
                        label="Loss Graph",
                    )

        def app_finetune_eval(
                # model/data params
                base_model,
                train_dataset,
                eval_dataset,
                experiment,

                # training hyperparams
                batch_size,
                micro_batch_size,
                num_epochs,
                learning_rate,
                cutoff_len,
                val_set_size,  # percent
                log,
        ):
            global path_database
            output, loss_graph = finetune_eval(base_model, train_dataset, eval_dataset, experiment, batch_size,
                                               micro_batch_size, num_epochs, learning_rate, cutoff_len, val_set_size,
                                               log)
            with open(path_database, "r") as file:
                data = json.load(file)

            if not any(desc["type"] == "experiment" and desc["name"] == experiment for desc in data):
                data.append({"type": "experiment", "name": experiment, "description": ""})

            # Écrit les descriptions mises à jour dans le fichier
            with open(path_database, "w") as file:
                json.dump(data, file)

            new_dataset_database, new_experiment_database = get_database()

            return output, loss_graph, Update_experiments_list(), Update_experiments_list(), gr.Textbox.update(
                value=new_experiment_database)

        with gr.Tab(label="Test model"):
            gr.Markdown(
                """
                🧠📝 Test Model 🧠📝\n
                This interface allows to test a model.\n
                Based on code : [Generate](https://github.com/tloen/alpaca-lora).\n
                Author: Ronan Cantin\n
                """)
            with gr.Row():
                with gr.Column(scale=1):
                    app2_dpd_base_model = gr.components.Dropdown(
                        choices=Base_models_list,
                        label="base_model",
                    )
                    app2_dpd_experiment = gr.components.Dropdown(
                        choices=Experiments_list,
                        label="experiment",
                    )
                    app2_tb_instruction = gr.components.Textbox(
                        lines=2,
                        label="Instruction",
                        placeholder="Tell me about alpacas.",
                    )
                    app2_tb_input = gr.components.Textbox(
                        lines=2,
                        label="Input",
                        placeholder="none",
                    )
                    app2_slid_temperature = gr.components.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.1,
                        label="Temperature"
                    )
                    app2_slid_top_p = gr.components.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.2,
                        label="Top p"
                    )
                    app2_slid_top_k = gr.components.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=40,
                        label="Top k"
                    )
                    app2_slid_beams = gr.components.Slider(
                        minimum=1,
                        maximum=4,
                        step=1,
                        value=1,
                        label="Beams"
                    )
                    app2_slid_repetition_penalty = gr.components.Slider(
                        minimum=0,
                        maximum=2,
                        step=0.05,
                        value=1.2,
                        label="repetition_penalty"
                    )
                    app2_slid_max_tokens = gr.components.Slider(
                        minimum=1,
                        maximum=2000,
                        step=1,
                        value=128,
                        label="Max tokens"
                    )
                    app2_btn_generate = gr.Button("Generate")

                with gr.Column(scale=2):
                    app2_tb_output = gr.components.Textbox(
                        lines=30,
                        label="Output"
                    )
                    with gr.Group():
                        app2_btn_good_result = gr.Button(value="Good result", interactive=False)
                        app2_btn_bad_result = gr.Button(value="Bad result", interactive=False)

        def app_generate(
                base_model,
                experiment,
                instruction,
                input_data,
                temperature,
                top_p,
                top_k,
                num_beams,
                repetition_penalty,
                max_new_tokens,
        ):
            answer = generate(base_model, experiment, instruction, input_data, temperature, top_p, top_k, num_beams,
                              repetition_penalty, max_new_tokens)

            return answer, gr.Button.update(interactive=True), gr.Button.update(interactive=True)

        def bad_response(
                instruction,
                input_data,
                answer
        ):
            return gr.Button.update(interactive=False), gr.Button.update(interactive=False)

        def good_response(
                instruction,
                input_data,
                answer
        ):
            add_good_answer(instruction, input_data, answer)
            return gr.Button.update(interactive=False), gr.Button.update(interactive=False)

        app2_btn_good_result.click(
            good_response,
            [app2_tb_instruction, app2_tb_input, app2_tb_output],
            [app2_btn_good_result, app2_btn_bad_result],
        )

        app2_btn_bad_result.click(
            bad_response,
            [app2_tb_instruction, app2_tb_input, app2_tb_output],
            [app2_btn_good_result, app2_btn_bad_result],
        )

        app2_btn_generate.click(
            app_generate,
            [app2_dpd_base_model, app2_dpd_experiment, app2_tb_instruction, app2_tb_input,
             app2_slid_temperature, app2_slid_top_p, app2_slid_top_k, app2_slid_beams, app2_slid_repetition_penalty,
             app2_slid_max_tokens],
            [app2_tb_output, app2_btn_good_result, app2_btn_bad_result],
        )

        with gr.Tab(label="Manage"):
            gr.Markdown(
                """
                📂📄 Manage 📂📄\n
                This interface allows to manage datasets and experiments.\n
                Author: Ronan Cantin\n
                """)
            with gr.Row():
                with gr.Column(scale=3):
                    app3_dpd_dataset = gr.Dropdown(
                        choices=[("None", None)] + Datasets_list,
                        label="Datasets"
                    )
                    app3_tb_dataset = gr.components.Textbox(
                        lines=10,
                        label="",
                        value=dataset_database,
                    )
                    app3_dpd_experiment = gr.Dropdown(
                        choices=[("None", None)] + Experiments_list,
                        label="Experiments"
                    )
                    app3_tb_Experiment = gr.components.Textbox(
                        lines=10,
                        label="",
                        value=experiment_database,
                    )
                with gr.Column(scale=1):
                    app3_tb_Rename = gr.components.Textbox(
                        lines=1,
                        label="Rename",
                    )
                    app3_btn_rename = gr.Button(value="Rename")
                    app3_tb_edit_description = gr.components.Textbox(
                        lines=1,
                        label="edit_description",
                    )
                    app3_btn_edit_description = gr.Button(value="Edit description")
                    app3_btn_delete = gr.Button(value="Delete")
                    # app3_btn_add = gr.Button(value="Add")

            def Update_datasets_list():
                global Datasets_list

                Datasets_list = [os.path.join(Datasets_path, f) for f in os.listdir(Datasets_path) if
                                 os.path.isfile(os.path.join(Datasets_path, f))]

                return gr.Dropdown.update(choices=[("None", None)] + Datasets_list)

            def Update_experiments_list():
                global Experiments_list

                Experiments_list = [os.path.join(Experiments_path, subdir) for subdir in
                                    next(os.walk(Experiments_path))[1]]
                return gr.Dropdown.update(choices=[("None", None)] + Experiments_list)

            def rename(dataset, experiment, new_name):
                global Datasets_list, Experiments_list, path_database

                if dataset in Datasets_list and experiment in Experiments_list:
                    raise "Error: an Experiment and a dataset are selected"
                else:
                    if dataset in Datasets_list:
                        choice = dataset
                        type_ = "dataset"
                    else:
                        choice = experiment
                        type_ = "experiment"

                    directory = os.path.dirname(choice)
                    new_path = os.path.join(directory, new_name)
                    try:
                        os.rename(choice, new_path)

                        with open(path_database, "r") as file:
                            data = json.load(file)

                        for desc in data:
                            if desc['type'] == type_ and desc['name'] == choice:
                                desc['name'] = new_path

                        with open(path_database, "w") as file:
                            json.dump(data, file)

                        new_dataset_database, new_experiment_database = get_database()

                        return Update_datasets_list(), Update_datasets_list(), Update_datasets_list(), Update_experiments_list(), Update_experiments_list(), gr.Textbox.update(
                            value=new_dataset_database), gr.Textbox.update(value=new_experiment_database)

                    except Exception as e:
                        print(f"Renaming error : {str(e)}")

            app3_btn_rename.click(
                rename,
                [app3_dpd_dataset, app3_dpd_experiment, app3_tb_Rename],
                [app3_dpd_dataset, app1_dpd_eval_dataset, app1_dpd_train_dataset, app3_dpd_experiment,
                 app2_dpd_experiment, app3_tb_dataset, app3_tb_Experiment]
            )

            def edit_description(dataset, experiment, new_description):
                global Datasets_list, Experiments_list, path_database

                if dataset in Datasets_list and experiment in Experiments_list:
                    raise "Error: an Experiment and a dataset are selected"
                else:
                    if dataset in Datasets_list:
                        choice = dataset
                        type_ = "dataset"
                    else:
                        choice = experiment
                        type_ = "experiment"

                    try:
                        with open(path_database, "r") as file:
                            data = json.load(file)

                        for desc in data:
                            if desc['type'] == type_ and desc['name'] == choice:
                                desc['description'] = new_description

                        with open(path_database, "w") as file:
                            json.dump(data, file)

                        new_dataset_database, new_experiment_database = get_database()

                        return gr.Textbox.update(value=new_dataset_database), gr.Textbox.update(
                            value=new_experiment_database)

                    except Exception as e:
                        print(f"Edit error : {str(e)}")

            app3_btn_edit_description.click(
                edit_description,
                [app3_dpd_dataset, app3_dpd_experiment, app3_tb_edit_description],
                [app3_tb_dataset, app3_tb_Experiment]
            )

            def delete(dataset, experiment):
                global Datasets_list, Experiments_list, path_database

                if dataset in Datasets_list and experiment in Experiments_list:
                    raise "Error: an Experiment and a dataset are selected"
                else:
                    if dataset in Datasets_list:
                        choice = dataset
                        type_ = "dataset"
                        os.remove(dataset)
                    else:
                        choice = experiment
                        type_ = "experiment"
                        shutil.rmtree(experiment)

                    try:
                        with open(path_database, "r") as file:
                            data = json.load(file)

                        data = [desc for desc in data if
                                desc["name"] != choice or desc["type"] != type_]

                        with open(path_database, "w") as file:
                            json.dump(data, file)

                        new_dataset_database, new_experiment_database = get_database()

                        return Update_datasets_list(), Update_datasets_list(), Update_datasets_list(), Update_experiments_list(), Update_experiments_list(), gr.Textbox.update(
                            value=new_dataset_database), gr.Textbox.update(value=new_experiment_database)

                    except Exception as e:
                        print(f"Deleting error : {str(e)}")

            app3_btn_delete.click(
                delete,
                [app3_dpd_dataset, app3_dpd_experiment],
                [app3_dpd_dataset, app1_dpd_eval_dataset, app1_dpd_train_dataset, app3_dpd_experiment,
                 app2_dpd_experiment, app3_tb_dataset, app3_tb_Experiment]
            )

            app1_btn_start_finetune.click(
                app_finetune_eval,
                [app1_dpd_base_model, app1_dpd_train_dataset, app1_dpd_eval_dataset, app1_tb_experiment,
                 app1_slid_batch_size, app1_slid_micro_batch_size, app1_slid_epochs, app1_slid_learning_rate,
                 app1_slid_cutoff_len, app1_slid_val_set_size, app1_cb_log],
                [app1_tb_output, app1_tb_loss_graph, app2_dpd_experiment, app3_dpd_experiment, app3_tb_Experiment],
            )

    interface.queue().launch(server_name=ip_address, share=share_gradio)


if __name__ == "__main__":
    fire.Fire(main)

