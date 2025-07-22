# Rework of code using GBaker/MedQA-USMLE-4-options dataset
#
# Includes the choices for each of the questions
#   More of a multiple choice format
#
# For testing, instead of breaking up into four choices,
#   now just one question with the four choices included in the prompt
# To test accuracy, have the model generate an answer and check for exact match
# Also have a testing set to calculate loss for each question
#   Use this to calculate perplexity


#######################################################
#######             NOTE        #################
# Some things to keep in mind while running this script
# 
# If doing evaluation before training:
#   - Set how much data you want to test on
#       line of code looks like: medQA_test = medQA["test"].select(range(5))
#       Above preprocess_test_function()
#   - At the bottom of evaluate_model_generation() function, uncomment the line where NOTE is
#       This needs to be uncommented because before training, model keeps the
#           letter of the choice it is picking.
#       Uncommenting this section will remove the letter so it can correctly be
#           compared to the correct answer
#   - In evaluate_mode_generation() function, there are two versions of
#       model generation: One with TextIteratorStreamer and one without
#       - TextIteratorStreamer displays text to the screen as it is generating
#       - The other vesion does not
#       - Choose which one you want to use
#   - Make sure the LoraConfig is uncommented and you are not trying
#       to load it from some directory
#   - Make sure the training section at the bottom of the file is commented out
#   - Make sure the evaluation section at the bottom of the file is uncommented
# 
# If doing first round of training:
#   - Set how much data you want to train on
#       line of code looks like:  medQA_train = medQA["train"].shuffle().select(range(5))
#       Above preprocess_train_function()
#   - Set how much data you want to test on
#       line of code looks like: medQA_test = medQA["test"].select(range(5))
#       Above preprocess_test_function()
#   - Make sure the LoraConfig is uncommented and you are not trying
#       to load it from some directory
#   - Make sure the training section at the bottom of the file is uncommented
#   -NOTE: Everything below this is for if you are also using the Evaluation section while training
#   - At the bottom of evaluate_model_generation() function, comment the line where NOTE is
#       This needs to be commented because after training, the model no longer keeps the
#           letter of the choice it is picking.
#       If this section is left commented, there will be an index_out_of_bounds error thrown
#   - In evaluate_mode_generation() function, there are two versions of
#       model generation: One with TextIteratorStreamer and one without
#       - TextIteratorStreamer displays text to the screen as it is generating
#       - The other vesion does not
#       - Choose which one you want to use
#   - Make sure the evaluation section at the bottom of the file is uncommented
# 
# Anything after first round of Training:
#   - Set how much data you want to train on
#       line of code looks like:  medQA_train = medQA["train"].shuffle().select(range(5))
#       Above preprocess_train_function()
#   - Set how much data you want to test on
#       line of code looks like: medQA_test = medQA["test"].select(range(5))
#       Above preprocess_test_function()
#   - Make sure the LoraConfig is commented out and you are loading the adapter
#       from the directory you saved it in during training
#   -NOTE: Everything below this is for if you are also using the Evaluation section,
#           either while training or on it's own
#   - At the bottom of evaluate_model_generation() function, comment the line where NOTE is
#       This needs to be commented because after training, the model no longer keeps the
#           letter of the choice it is picking.
#       If this section is left commented, there will be an index_out_of_bounds error thrown
#   - In evaluate_mode_generation() function, there are two versions of
#       model generation: One with TextIteratorStreamer and one without
#       - TextIteratorStreamer displays text to the screen as it is generating
#       - The other vesion does not
#       - Choose which one you want to use
#   - Make sure the evaluation section at the bottom of the file is uncommented 


from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import Trainer, TrainingArguments, EvalPrediction
from transformers import DataCollatorWithPadding
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from copy import deepcopy
import torch
import numpy as np
import random
import threading


# Load GBaker/MedQA-USMLE-4-options dataset
medQA = load_dataset("GBaker/MedQA-USMLE-4-options")

# Important parts of this dataset:
#   - question: The question being asked
#   - options: The multiple choice options for the question
#   - answer: The correct answer to the question
#       - answer_idx: The letter corresponding to the correct answer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

#NOTE: Training set
medQA_train = medQA["train"].shuffle().select(range(20))  # Select a subset for training; random
#medQA_train = medQA["train"].select(range(5))

def preprocess_train_function(examples):
    """
    Preprocesses a QA example for mixed training:
    - Sometimes uses multiple-choice format (MC)
    - Sometimes uses open-ended question → answer format
    
    Parameters:
    - example: dict with 'question', 'choices', 'answer' (letter: 'A', 'B', ...)

    Returns:
    - dict with input_ids, attention_mask, labels
    """

    possible_input_texts = []
    prompt_lens = []

    for question, answer, choices in zip(examples["question"], examples["answer"], examples["options"]):

        choices_block = "\n".join([f"{key}. {choice}" for key, choice in choices.items()])
        prompt = f"Question:\n{question}\n\n{choices_block}\n\nAnswer:\n"
        prompt_lens.append(len(tokenizer(prompt)["input_ids"]))


        full_text = prompt + answer
        #print(full_text)

        possible_input_texts.append(full_text)


    # Tokenize
    tokenized = tokenizer(
        possible_input_texts,
        add_special_tokens=True,
        return_tensors=None,)
    
    all_input_ids = []
    attention_masks = []
    paddings = []
    for input_ids, attention in zip(tokenized["input_ids"], tokenized["attention_mask"]):
        padding = len(input_ids) % 8
        paddings.append(padding)
        input_id = [tokenizer.pad_token_id] * padding + input_ids
        attention_mask = [0] * padding + attention
        all_input_ids.append(input_id)
        attention_masks.append(attention_mask)

    tokenized["input_ids"] = all_input_ids
    tokenized["attention_mask"] = attention_masks        

    # Set labels to match input_ids, mask out padding
    tokenized["labels"] = tokenized["input_ids"][:] 
    for idx, labels in enumerate(tokenized["labels"]):
        pad = paddings[idx]
        prompt_len = prompt_lens[idx]
        labels = labels.copy()
        labels[:pad+prompt_len] = [-100] * (pad+prompt_len)
        tokenized["labels"][idx] = labels 

    return tokenized


tokenized_medQA_train = medQA_train.map(
    preprocess_train_function,
    batched=True,
    remove_columns=medQA_train.column_names,
)

#print(tokenized_medQA_train[0])
#print(tokenizer.decode(tokenized_medQA_train["input_ids"][0]))


# NOTE:Preprocess function for the test set
#medQA_test = medQA["test"].shuffle().select(range(1000))  # Select a subset for testing; random
medQA_test = medQA["test"].select(range(5))


# Set up the dictionary that is needed for evaluation and the compute_metric
# This is the preprocessing function for the eval dataset sent to trainer
#   Used to calculate perplexity and other loss based metrics
def preprocess_test_function(examples):
    """
    Preprocesses a QA example for *testing* using open-ended format.
    This dataset contains both the question and answer so model can be evaluated on loss metrics
    Labels are masked before the answer portion to evaluate loss only on answer.
    """
    #possible_answers = ["A", "B", "C", "D"]

    processed = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }

    possible_input_texts = []

    prompts = []

    for i in range(len(examples["question"])):
        prompt = f"Question:\n{examples['question'][i]}\n\nAnswer:\n"
        prompts.append(prompt)
        gold_answer_text = examples["answer"][i]

        # Combine prompt and answer
        full_text = prompt + gold_answer_text

        possible_input_texts.append(full_text)


    # Tokenize
    tokenized = tokenizer(
        possible_input_texts,
        add_special_tokens=True,
        return_tensors=None,)


    for i in range(len(prompts)):
        prompt_len = len(tokenizer(prompts[i])["input_ids"])

        token_ids = tokenized["input_ids"][i]
        attention_mask = tokenized["attention_mask"][i]
        
        pad_len = len(token_ids) % 8

        token_ids = [tokenizer.pad_token_id] * pad_len + token_ids
        attention_mask = [0] * pad_len + attention_mask

        labels = token_ids.copy()
        labels[:pad_len+prompt_len] = [-100] * (pad_len+prompt_len)

        processed["input_ids"].append(token_ids)
        processed["attention_mask"].append(attention_mask)
        processed["labels"].append(labels)

    return processed


tokenized_medQA_test = medQA_test.map(
    preprocess_test_function, 
    batched=True,
    remove_columns=medQA_test.column_names)

#print(tokenized_medQA_test[0])
#print(tokenizer.decode(tokenized_medQA_test["input_ids"][0]))


def preprocess_generation_function(examples):
    """
    Preprocesses a QA example for *testing* using open-ended format.
    Only contains the question so that the model can generate the answer
    Don't care about loss so no need for masking except for padding.
    Also returns gold answer for external metrics like accuracy or F1.
    """

    processed = {
        "input_ids": [],
        "attention_mask": [],
        "gold_answer_text": [],
    }

    prompts = []

    for i in range(len(examples["question"])):
        choices_block = "\n".join([f"{key}. {choice}" for key, choice in examples["options"][i].items()])
        prompt = f"Question:\n{examples['question'][i]}\n\n{choices_block}\n\nAnswer:\n"
        prompts.append(prompt)
        gold_answer_text = examples["answer"][i]

        processed["gold_answer_text"].append(gold_answer_text)


    # Tokenize
    tokenized = tokenizer(
        prompts,
        add_special_tokens=True,
        return_tensors=None,)
    
    processed["input_ids"] = tokenized["input_ids"]
    processed["attention_mask"] = tokenized["attention_mask"]


    return processed


tokenized_medQA_generate = medQA_test.map(
    preprocess_generation_function, 
    batched=True,
    remove_columns=medQA_test.column_names)


#print(tokenized_medQA_generate[0])
#print(tokenizer.decode(tokenized_medQA_generate["input_ids"][0]))

# Normalized match
def normalize(text):
    return text.lower().strip()

# This is the evaluation for metrics of what the model generates
def evaluate_model_generation(model, tokenizer):
    model.eval()
    eval_dataset = deepcopy(tokenized_medQA_generate)
    eval_dataset = eval_dataset.remove_columns(["gold_answer_text"])

    all_preds = []
    gold_answers = [answer.strip() for answer in tokenized_medQA_generate["gold_answer_text"]]

    for i in range(len(eval_dataset["input_ids"])):
        input_ids = torch.tensor(eval_dataset["input_ids"][i]).unsqueeze(0).to(model.device)
        attention_mask = torch.tensor(eval_dataset["attention_mask"][i]).unsqueeze(0).to(model.device)

        # NOTE: Streamer is for if you want to see what it is outputting before
        #   it is postprocessed to compare to the expected value
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = threading.Thread(target=model.generate, kwargs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 50,
            "streamer": streamer,
        })
        thread.start()

        # Collect the output from the streamer
        decoded_tokens = ""
        for token in streamer:
            print(token, end="", flush=True)  # stream to console
            decoded_tokens += token

        print()  # new line

        all_preds.append(decoded_tokens.strip())

        # NOTE: This is for if you don't care about seeing the generated output as it's generating it
        '''generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
        )
        
        new_tokens = generated_ids[0, input_ids.shape[1]:]
        preds = tokenizer.decode(new_tokens, skip_special_tokens=True)

        all_preds.append(preds)'''

    correct = []
    print()
    for i in range(len(all_preds)):
        #print(f"Prediction: {all_preds[i]}", end=" ")
        #print(f"Expected: {gold_answers[i]}")
        prediction = all_preds[i]
        parts = prediction.split("\n")
        prediction = parts[0].strip()

        # NOTE: Need to uncomment if testing before first training
        '''parts = prediction.split('.', 1)
        prediction = parts[1].strip()'''

        print(f"Prediction: {prediction}")
        print(f"Expected: {gold_answers[i]}\n")

        correct.append(int(normalize(prediction) ==  normalize(gold_answers[i])))

    accuracy = sum(correct) / len(all_preds)
    return {"accuracy": accuracy,
            "correct": correct}


def compute_per_example_loss(logits, labels):
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute loss per token
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    per_token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(shift_labels.size()) # (batch, seq_len)

    # Mask out -100s and average per example
    valid_mask = shift_labels != -100
    per_example_loss = (per_token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)

    return per_example_loss

def compute_perplexity_metrics(losses, correct):
    
    # calculate the perplexity for each example
    # shows how confident the model was generating the correct answer for each example
    perplexities = torch.exp(losses)
    
    correct_avg_perplex = None
    wrong_avg_perplex = None
    if correct is not None:
        right = []
        wrong = []
        for i in range(len(correct)):
            right.append(losses[i]) if correct[i]==1 else wrong.append(losses[i])

        right = torch.tensor(right)
        wrong = torch.tensor(wrong)

        correct_avg_perplex = torch.exp(right.mean()).item()
        wrong_avg_perplex = torch.exp(wrong.mean()).item()

    # Shows the average perplexity across all the examples
    # Should be low if the model is confident in it's choices
    average_perplexity = torch.exp(losses.mean()).item()



    return perplexities, correct_avg_perplex, wrong_avg_perplex, average_perplexity


def compute_metrics(eval_preds: EvalPrediction, gen_results=None):
    logits = eval_preds.predictions
    labels = eval_preds.label_ids

    losses = compute_per_example_loss(logits, labels)
    print(losses)

    accuracy = None
    correct = None
    if gen_results is not None:
        accuracy = gen_results["accuracy"]
        correct = gen_results["correct"]


    perplexities, correct_avg_perplex, wrong_avg_perplex, average_perplexity = compute_perplexity_metrics(losses, correct)
    print(perplexities)

    results = {
        "average_perplex": average_perplexity
    }
    if gen_results is not None:
        results["accuracy"] = accuracy
        results["correct_avg_perplex"] = correct_avg_perplex
        results["wrong_avg_perplex"] = wrong_avg_perplex

    

    return results


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


# NOTE: Use Before first training
# create LoRA configuration object
'''lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # type of task to train on
    inference_mode=False, # set to False for training
    r=8, # dimension of the smaller matrices
    lora_alpha=32, # scaling factor
    lora_dropout=0.1 # dropout of LoRA layers
)

# Add LoraConfig to the model
model = get_peft_model(model, lora_config)'''

# NOTE: After first training
model = PeftModel.from_pretrained(model, "./test2/lora_adapter")

# Need to call this anytime the adapter is loaded
# Esures:
#   - The LoRA weights are active
#   - Their gradients are enabled (requires_grad = True)
#   - You're not accidentally training a frozen model 
model.set_adapter("default")


# Define training hyperparameters in TrainingArguments
training_args = TrainingArguments(
    output_dir="test2",  # Where to save checkpoints & logs
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
    per_device_eval_batch_size=1, # For evaluation: 1 question, correct answer
    dataloader_drop_last=False,

    #prediction_loss_only=False,

    save_strategy="epoch",  #Optional: save checkpoint after each epoch
    logging_strategy="epoch",   #Optional: log once per epoch

    # Other useful options
    #save_total_limit=2, # Limit number of saved checkpoints
    per_device_train_batch_size=1,  # Trained on correct answer for each question, so none need to be batched together
    gradient_accumulation_steps=1,
    label_names=["labels"],
    fp16=True,
    fp16_full_eval=True
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



# NOTE: Train
print(trainer.train())
# Need to manually save LoRA Adapter afterwords
model.save_pretrained("./test2/lora_adapter")


# NOTE: Evaluate
# Generation phase
generation_results = evaluate_model_generation(model, tokenizer)
trainer.compute_metrics = lambda p: compute_metrics(p, gen_results=generation_results)
#print()
# loss phase
print(trainer.evaluate())

