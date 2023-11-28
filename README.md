# 🚀🌒 AI-INTERNSHIP-PROJECT

The aim of this project is to create an interface for finetuning a model and testing models. 
Based on the code : [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora) & [gururise/AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned).


### Project installation

1. Clone the project locally on your machine

   ```bash
   git clone https://github.com/Ronan31/ai-intership-project.git
   ```
2. Go to the directory

3. Create a python environment

   ```bash
   virtualenv -p pyhton3 venv
   ```
4. Activate the python environment

   ```bash
   . venv/bin/activate
   ```
5. Execute requirement.txt

   ```bash
   pip install -r requirements.txt
   ```


### Launch the interface:

1. Go to the directory ai_internship_project 

2. Activate the python environment

   ```bash
   . venv/bin/activate
   ```
3. Launch the interface 

   ```bash
   LD_LIBRARY_PATH=/usr/local/cuda-11.6/targets/x86_64-linux/lib python3 interface.py
   ```
4. Open the address indicate in the browser


### Interface parts

1. Finetune

The Finetune tab allows to train a LLM with a custom dataset and to evaluate the result with F1-
score classification metric. On the left of the interface are all the parameters for the finetune:
- Base_model: The base model refers to the pre-trained architecture that you use as a starting
point for fine-tuning.
- Train_dataset: The training dataset is the collection of data on which the model is trained. This
data should be representative of the task you are trying to solve.
- Eval_dataset: The evaluation (or validation) dataset is used to assess the model's performance
during training without the model having seen it during learning. This helps detect overfitting.
- Experiment: Name of experiment.
- Batch_size: Batch size represents the number of training samples used in one iteration. A batch
is a division of the training dataset.
- Micro_batch_size: In the context of parallel or distributed training, micro batch size may refer to
the batch size used for each processing unit.
- Num_epochs: The number of epochs is the number of times the learning algorithm sees the
entire dataset during training.
- Learning_rate: The learning rate is a hyperparameter that determines the size of the steps the
model takes during optimization. It influences how quickly the model converges to a solution.
- Cutoff_len: Cutoff length may refer to a limit imposed on the maximum length of input or
output sequences.
- Val_set_size: The size of the validation set refers to the proportion of the training dataset that
you reserve for model evaluation.
- Log: Defines whether to save the results of the experiment.
On the right-hand side of the interface, you'll find a window showing a summary of the fine tune
parameters, the fine tune results and the 3 best answers and the 3 worst answers according to the F1
score. And a window with a graph of the fine tune's training and evaluation loss curves.

2. Test model

The "Test model" tab is used to test the result of the finetune, so you can use your experiment.
To do this, on the left you have the various parameters to set:
- Base_model: Basic model with which the experiment was fintuned.
- Experiment: name of experiment tested.
- Instruction: Instruction corresponds to the prompt or query you provide to the model to
generate a response or output.
- Input: Input represents the input data you provide to the model to generate output. This could
be the initial text, a question, etc.
- Temperature: Temperature is a parameter used in probabilistic text generation. A higher
temperature makes the probability distribution more balanced, producing more creative but
potentially less reliable results, while a lower temperature produces more deterministic results.
- Top_p: Top-p is a text generation technique that considers only tokens with cumulative
probability above a certain threshold (p). It helps control the diversity of generation.
4
- Top_k: Top-k is a technique similar to Top-p, but instead of relying on cumulative probability, it
focuses on the top k most probable tokens.
- Beams: The number of beams in text generation refers to the number of parallel paths the
model explores when generating a sequence. This can affect the diversity and quality of the
results.
- Repetition_penalty: Repetition penalty is a measure used to discourage the model from
generating the same sequence multiple times. It can help improve the diversity of the generated
text.
- Max_tokens: Max_tokens represents the maximum number of tokens, imposing a limit on the
total length of the generated sequence.
On the left is the result of the instruction and the two buttons are used to add the example (in the case
of a good response) to the "new_dataset.json" dataset.

3. Management

The management tab allows you to sort the different Datasets and Experiments. You can
rename, delete or edit the description associated with the experiment or dataset. Descriptions are saved
in the file “database.json” and you cannot perform an action on an Experiment and a database at the
same times.


### Important directories/files:

→ database.json
• File used to save descriptions relating to the management part of the interface.
→ Datasets
• This is where the datasets are stored. If you want to add a dataset, save it in this folder.
→ new_dataset.json
• This dataset is built with the different good answers you choose during the model tests.
→ Experiments
• This is where the finetune result is stored.
→ Experiments_log
• This folder stores the various finetune results.
→ Base_models
• This is where the base models are stored. If you want to add a base model, save it in this folder.


### Other python programs:

→ multiply_dataset.py
• Program for multiplying a dataset. The programme uses the finetuned alpaca-lora-7b model to
reformulate instructions and outputs. Parameters:
- dataset_path: the path of the dataset that is multiplied.
- new_dataset_path: path of the dataset to be created.
- nb_multiply: number of times to multiply the dataset.
Base model: [llama-7b-hf](https://huggingface.co/decapoda-research/Llama-7b-hf).
Experiment: [Alpaca-lora-7b](https://huggingface.co/tloen/alpaca-lora-7b).
   ```bash
   LD_LIBRARY_PATH=/usr/local/cuda-11.6/targets/x86_64-linux/lib python3 –dataset_path dataset.json -new_dataset_path new_dataset.json -nb_multiply 1
   ```
→ reduce_dataset.py
• Program to reduce a dataset. The programme reduces the dataset to the number of lines
mentioned. Parameters:
- dataset_path: the path of the dataset that is reduced.
- new_dataset_path: path of the dataset to be created.
- nb_lines: number of lines in the new datasets.
   ```bash
   python3 –dataset_path dataset.json -new_dataset_path new_dataset.json -nb_lines 352
   ``` 
### Author: Ronan Cantin
