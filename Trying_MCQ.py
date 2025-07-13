# Now that I know I can actually train a model,
#   this is going to try to train on multiple choice questions
# Going to try on GBaker/MedQA-USMLE-4-options dataset

# NOTE: This code runs, but I'm not sure if it calculates accuracy correctly
#       Was only able to run it on very small datasets
#       Would need someone to verify on larger dataset
# 
# To optimize for your device, may need to modify the following:
#   - Sizes of medQA_test and medQA_train datasets
#   - the max_length value within preprocess_test_function 
#       (However changing this parameter may affect results because of padding)
#   - block size for preprocess_train_function
#   - quantization_config
#   - max_memory in model
#   - training_args 

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
import torch
import numpy as np
from collections import defaultdict


# Load GBaker/MedQA-USMLE-4-options dataset
medQA = load_dataset("GBaker/MedQA-USMLE-4-options")

# Important parts of this dataset:
#   - question: The question being asked
#   - options: The multiple choice options for the question
#   - answer: The correct answer to the question
#       - answer_idx: The letter corresponding to the correct answer

# Need to do different preprocessing for the train set and the test set
#   - Will use the test set to evaluate the model
#       - Prepare the dataset so that each (prompt + one choice) is a separate evaluation example
#       - Set labels so only the answer tokens are scored (prompt is masked out).
#       - Aggregate results per original exmaple (group of choices) manually after evaluation
#   - Will use the train set to train the model
#       - Train on the question with the correct answer attached
#       - Don't use any of the other options

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Preprocess function for the test set
#medQA_test = medQA["test"].shuffle().select(range(5))  # Select a subset for testing; random
medQA_test = medQA["test"].select(range(5))

# Set up the dictionary that is needed for evaluation and the compute_metric
def preprocess_test_function(example):
    possible_answers = ["A", "B", "C", "D"]

    question = f"Question: {example["question"]} Answer: "
    choice_texts = [" " + options for options in example["options"].values()]
    answer_index = example["answer_idx"]


    results = {
        "input_ids": [],
        "labels": [],
        "question_id": [],
        "choice_index": [],
        "is_correct": [],
    }
    for i, choice in enumerate(choice_texts):
        full_text = question + choice
        input_ids = tokenizer(full_text, padding="max_length", truncation=True, max_length=512).input_ids
        labels = input_ids.copy()

        # Mask out everything before the start of the answer
        answer_start = len(tokenizer(question, truncation=True, max_length=512)["input_ids"])
        #print(answer_start)
        labels[:answer_start] = [-100] * answer_start  # Mask out the question part

        results["input_ids"].append(input_ids)
        results["labels"].append(labels)
        results["question_id"].append(example["question"])
        #print(i, end=" ")
        results["choice_index"].append(i)
        #print(answer_index, possible_answers.index(answer_index))
        results["is_correct"].append(int(i == possible_answers.index(answer_index)))

    return results

tokenized_medQA_test = medQA_test.map(
    preprocess_test_function, 
    remove_columns=medQA_test.column_names)


# Flatten all the lists so that it can be taken by compute_metric
def flatten(examples):

    result = {}
    for key, items in examples.items():
        result[key] = []
        for item in items:
            for el in item:
                result[key].append(el)

    return result

tokenized_medQA_test = tokenized_medQA_test.map(
    flatten,
    batched=True
)


# Preprocess function for the train set

# How training works
# Given a multiple choice question, you train the model to predict the correct answer
#   This involves concatenating the question with the correct answer and only training on that text

# Training Setup
# For each training sample:
#   - Your input_ids are: tokenizer("Question: ... Answer: ...")
#   - Your labels are the same, but mask the "Question: ... Answer:" part
#       with -100 so the loss is only computed on the Answer 

medQA_train = medQA["train"].shuffle().select(range(5))  # Select a subset for training; random
#medQA_train = medQA["train"].select(range(5))

block_size = 128 #Change this based on how much your GPU can handle

def preprocess_train_funciton(example):
    questions = [f"Question: {question} Answer: " for question in example["question"]]
    answers = example["answer"]
    input_ids = []
    labels = []
    for question, answer in zip(questions, answers):    
        input_text = question + answer
        input_id = tokenizer(input_text).input_ids

        label = input_id.copy()
        # Mask out everything before the start of the answer
        answer_start = len(tokenizer(question).input_ids)
        label[:answer_start] = [-100] * answer_start

        input_ids.extend(input_id)
        labels.extend(label)
    

    
    #Need to break both input_ids and labels into blocks

    # Total_length gets the length of the concatenated input_ids
    total_length = len(input_ids)

    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    #   customize this part to your needs.
    # It ensures that the total length is a multiple of block_size, dropping any
    #   remainder tokens that don't fit into a full block.
    # This avoids partial chunks at the end, unless you later choose to pad them 
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    # Split by chunks of block_size.
    # Splits each flattened list into chunks of size block_size
    # Important because the model expects inputs of a fixed size
    # k is the key (e.g., input_ids, attention_mask), and t is the list of values
    #   for that key.
    result = {
        "input_ids": [input_ids[i : i + block_size] for i in range(0, total_length, block_size)],
        "labels": [labels[i: i + block_size] for i in range(0, total_length, block_size)]
    }


    return result

tokenized_medQA_train = medQA_train.map(
    preprocess_train_funciton,
    batched=True,
    remove_columns=medQA_train.column_names,
)

# Define a compute_metrics function
def compute_metrics(eval_preds):
    #losses = eval_preds.predictions # These are loss values returned by 'prediction_loss_only=True'
    logits = eval_preds.predictions
    labels = eval_preds.label_ids

    # Each entry corresponds to (question, choice)
    # defaultdict provides a default value for a nonexistent key in the dictionary,
    #   eliminating the need for checking if the key exists before using it
    results = defaultdict(list)
    for i in range(len(tokenized_medQA_test["is_correct"])):
        qid = tokenized_medQA_test["question_id"][i]
        choice_idx = tokenized_medQA_test["choice_index"][i]
        is_correct = tokenized_medQA_test["is_correct"][i]
        # Let's use the mean logit value as a crude scoring method
        score = -np.mean(logits[i]) # negative for "lower is better"
        results[qid].append((choice_idx, score, is_correct))
    
    correct = 0
    total = 0
    for qid, choices in results.items():
        true_choices = [idx for idx, _, correct_flag in choices if correct_flag]
        true_choice = true_choices[0]
        pred_choice = min(choices, key=lambda x: x[1])[0] # pick choice with lowest (best) score
        if pred_choice == true_choice:
            correct += 1
        total += 1
    
    accuracy = correct / total
    return {"accuracy": accuracy}

# Run Evaluation with Trainer
quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16)

# Load Mistral model
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/mistral-7b-instruct-v0.3",
    device_map="auto",
    quantization_config=quantization_config,
    #torch_dtype="auto",
    trust_remote_code=False,
    max_memory={
        0: "4GiB",
        "cpu": "28GiB"
    }
)

# create LoRA configuration object
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # type of task to train on
    inference_mode=False, # set to False for training
    r=8, # dimension of the smaller matrices
    lora_alpha=32, # scaling factor
    lora_dropout=0.1 # dropout of LoRA layers
)

# Add LoraConfig to the model
model.add_adapter(lora_config, adapter_name="lora_1")

# Define training hyperparameters in TrainingArguments
training_args = TrainingArguments(
    output_dir="test",  # Where to save checkpoints & logs
    overwrite_output_dir=True, # Overwrite existing output_dir if needed

    # Training-specific
    do_train=True,
    #num_train_epochs=3, # Set according to your needs
    #per_device_train_batch_size=4,  # Match or adjust for GPU memory
    learning_rate=2e-5, # Typical starting point for transformers
    weight_decay=0.01,  # Optional: helps regularize the model

    # Evaluation
    do_eval=True,
    eval_strategy="epoch",    # or "steps" (set eval_steps if using steps)
    per_device_eval_batch_size=4,
    dataloader_drop_last=False,

    prediction_loss_only=False,

    save_strategy="epoch",  #Optional: save checkpoint after each epoch
    logging_strategy="epoch",   #Optional: log once per epoch

    # Other useful options
    #save_total_limit=2, # Limit number of saved checkpoints
    #load_best_model_at_end=True, # Optional: load best checkpoint by eval metric
    #metric_for_best_model="accuracy",   # Must match your compute_metrics output
)

# Pass the training arguments to Trainer along with the model, datasets, and data collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_medQA_train,
    eval_dataset=tokenized_medQA_test,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)
    
#print(trainer.evaluate())
print(trainer.train())
