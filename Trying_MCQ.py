# Now that I know I can actually train a model,
#   this is going to try to train on multiple choice questions
# Going to try on GBaker/MedQA-USMLE-4-options dataset

# NOTE------------------------------------------
#               NOTES FOR USE
# NOTE: At that bottom of this file there is trainer.train() and trainer.evaluate()
#   Use trainer.train() for training
#       After training make sure to save the LoRA adapter to some director 
#           to load in the future
#       Metrics will also be calculated during training 
#   Use trainer.evaluate() for evaluating the model on the metrics
#       If the adapter is not loaded in when evaluating after your first time training,
#           the base model will be evaluated.
#           This is because we are not actually changing any of the
#               parameters of the model.
#           Only changing parameters of the adapter
#       If you want to evaluate the base model, then do not load the adapter
# 
# NOTE: There are two ways to enable your model to use Low-Rank Adaptation (LoRA),
#           a method for parameter-efficient fine-tuning (PEFT)
#   - When you are training for the first time: 
#       Set up the LoraConfig and then wrap your base model in the LoraConfig
#       After traing, make sure you save your model to some directory
#       Do this so that you can load the trained adapter when you want to 
#           evaluate or continue training the model
#   - After your first time traing:
#        If you want to evaluate or continue training the model
#           from the most recent version of your trained model,
#           load the adapter from the directory you saved it in
#           and wrap your base model in it.
#           Use  PeftModel.from_pretrained()
#       Whenever continuing your training, make sure to always save
#           the model after training 

#############  Additional Notes ##############
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
from transformers import DataCollatorWithPadding
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
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
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", model_max_length=512, add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# Preprocess function for the test set
#medQA_test = medQA["test"].shuffle().select(range(1000))  # Select a subset for testing; random
medQA_test = medQA["test"].select(range(5))


# Set up the dictionary that is needed for evaluation and the compute_metric
def preprocess_test_function(examples):
    possible_answers = ["A", "B", "C", "D"]

    processed = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "question_id": [],
        "choice_index": [],
        "is_correct": [],
    }

    for i in range(len(examples["question"])):
        question = f"Question: {examples['question'][i]} Answer: "
        options = examples["options"][i]
        answer_idx = possible_answers.index(examples["answer_idx"][i])
        question_id = examples["question"][i]

        max_len = 0
        choice_inputs = []

        # First pass: tokenize all full texts and compute max_len
        for choice_letter in possible_answers:
            choice_text = " " + options[choice_letter]
            full_text = question + choice_text
            tokenized = tokenizer(full_text, add_special_tokens=True)
            choice_inputs.append(tokenized)
            max_len = max(max_len, len(tokenized["input_ids"]))


        max_len = max_len + (max_len % 8)

        # Second pass: pad inputs and build labels
        for j, tokenized in enumerate(choice_inputs):
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            padding_len = max_len - len(input_ids)
            padded_input_ids = [tokenizer.pad_token_id] * padding_len + input_ids
            padded_attention_mask = [0] * padding_len + attention_mask

            question_len = len(tokenizer(question, add_special_tokens=False)["input_ids"])
            labels = [-100] * padding_len + input_ids.copy()
            labels[:padding_len + question_len] = [-100] * (padding_len + question_len)

            processed["input_ids"].append(padded_input_ids)
            processed["attention_mask"].append(padded_attention_mask)
            processed["labels"].append(labels)
            processed["question_id"].append(question_id)
            processed["choice_index"].append(j)
            processed["is_correct"].append(int(j == answer_idx))

    return processed


tokenized_medQA_test = medQA_test.map(
    preprocess_test_function, 
    batched=True,
    remove_columns=medQA_test.column_names)



# Preprocess function for the train set

# How training works
# Given a multiple choice question, you train the model to predict the correct answer
#   This involves concatenating the question with the correct answer and only training on that text

# Training Setup
# For each training sample:
#   - Your input_ids are: tokenizer("Question: ... Answer: ...")
#   - Your labels are the same, but mask the "Question: ... Answer:" part
#       with -100 so the loss is only computed on the Answer 

medQA_train = medQA["train"].shuffle().select(range(30))  # Select a subset for training; random
#medQA_train = medQA["train"].select(range(5))

#block_size = 128 #Change this based on how much your GPU can handle

def preprocess_train_function(examples):
    possible_input_texts = [
        f"Question: {q} Answer: {a}" for q, a in zip(examples["question"], examples["answer"])
    ]

    # Tokenize
    tokenized = tokenizer(
        possible_input_texts,
        add_special_tokens=True,
        return_tensors=None,)

    # Compute labels with masked question part
    labels = []

    for i in range(len(possible_input_texts)):
        input_ids = tokenized["input_ids"][i]
        input_len = len(input_ids)
        padding = input_len % 8

        input_ids = [tokenizer.pad_token_id] * padding + input_ids
        attention_mask = [0] * padding + tokenized["attention_mask"][i]

        tokenized["input_ids"][i] = input_ids
        tokenized["attention_mask"][i] = attention_mask

        question = f"Question: {examples['question'][i]} Answer: "
        question_len = len(tokenizer(question, add_special_tokens=False)["input_ids"])

        label_ids = input_ids.copy()
        label_ids[:padding+question_len] = [-100] * (padding+question_len)
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


tokenized_medQA_train = medQA_train.map(
    preprocess_train_function,
    batched=True,
    remove_columns=medQA_train.column_names,
)

#print(tokenized_medQA_train[0])
#print(tokenizer.decode(tokenized_medQA_train[0]["input_ids"]))

def compute_results(logits):
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
    
    print(results)
    return results

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

def compute_perplexity_metrics(results, losses):
    # divide the losses into their groups of 4 (4 choices for each question)
    blocked_losses = [losses[i : i+4] for i in range(0, len(losses), 4)]
    

    # The perplexities of the correct answers
    # If the model is trained well, all loses in true_losses should
    #   be the same ones found by min_loss
    true_losses = []

    correct = 0
    total = 0
    for qid, choices in results.items():
        true_choices = [idx for idx, _, correct_flag in choices if correct_flag]
        true_choice = true_choices[0]
        true_losses.append(blocked_losses[total][true_choice])
        
        # Because the greater the loss, the greater the perplexity,
        #   can just find min loss
        # Min loss should be the correct answer if model trained correctly
        min_loss = min(blocked_losses[total]) # pick the choice with the lowest (best) loss
        pred_choice = (blocked_losses[total] == min_loss).nonzero(as_tuple=True)[0][0] # get the index of that choice
        if pred_choice == true_choice:
            correct += 1
        total += 1
    
    
    perplexity_accuracy = correct / total
    
    # If the model is trained well, correct_perplexity should be much less than all_perplexity
    # Should be predicting the correct answer with much more confidence than others
    correct_perplexity = torch.exp(torch.tensor(true_losses).mean()).item()
    all_perplexity = torch.exp(losses.mean()).item()

    return perplexity_accuracy, correct_perplexity, all_perplexity


def compute_MR_MRR(results, losses):
    # divide the losses into their groups of 4 (4 choices for each question)
    blocked_losses = [losses[i : i+4] for i in range(0, len(losses), 4)]

    results_by_question = defaultdict(list)

    index = 0
    for qid, answers in results.items():
        for i, answer in enumerate(answers):
            results_by_question[qid].append((blocked_losses[index][i], answer[2]))
        index += 1

    reciprocal_ranks = []
    ranks = []


    for qid, answers in results_by_question.items():
        # Sort by loss (lower is better)
        sorted_answers = sorted(enumerate(answers), key=lambda x: x[1][0])
        # Find rank of correct answer
        for rank, (idx, (loss, is_correct)) in enumerate(sorted_answers, start=1):
            if is_correct:
                reciprocal_ranks.append(1.0/rank)
                ranks.append(rank)
                break # Only one correct answer per question
    
    mrr = np.mean(reciprocal_ranks)
    mean_rank = np.mean(ranks)

    return mrr, mean_rank

def compute_confidence_margin(losses):
    # divide the losses into their groups of 4 (4 choices for each question)
    blocked_losses = [losses[i : i+4] for i in range(0, len(losses), 4)]

    margins = []

    for loss_block in blocked_losses:
        sorted_losses = sorted(loss_block)
        if len(sorted_losses) >= 2:
            margin= sorted_losses[1] - sorted_losses[0]
            margins.append(margin)
    
    avg_margin = float(np.mean(margins))
    std_margin = float(np.std(margins))
    median_margin = float(np.median(margins))
    num_questions = len(blocked_losses)

    return avg_margin, std_margin, median_margin, num_questions



    
def compute_accuracy(results):
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
    return accuracy



# Define a compute_metrics function
def compute_metrics(eval_preds):
    logits = eval_preds.predictions
    labels = eval_preds.label_ids


    results = compute_results(logits)

    #print(compute_per_example_loss(logits, labels))

    losses = compute_per_example_loss(logits, labels)
    print(losses)

    # Calculate accuracy based on perplexity
    # Perplexity is measuring confident is the model in perdicitng the next token
    # So for MC, "How surprising is this answer, given the quesiton?"
    # A well trained model should have the lowest perplexities for the correct answer
    # Lower perplexity --> more confident, more accurate
    # Higher perplexity --> more uncertainty, more spread-out probability
    #  
    # perplexity_accuracy - how many pedictions were correct based on min perplexity
    # correct_perplexity - The average perplexity across the correct answers
    # all_perplexity - The average perplexity across all answers
    #   If the model is trained well, correct_perplexity should be much less than all_perplexity;
    #   This is because the model should be confident about predicting the correct answers compared to the others
    perplexity_accuracy, correct_perplexity, all_perplexity = compute_perplexity_metrics(results, losses)

    # Calculate Mean Rank and Mean Reciprocal Rank
    # Based on the losses, rank each of the models predicted answer (lower loss --> better rank)
    # Mean Rank - Average position of the correct answer (Lower is better)
    #           - If always predicts correct answer with lowest loss, then Mean Rank = 1.0
    #               - Will be greater than 1 if anything else
    # MRR - On average, how close to first is the correct answer (Higher is better)
    #     - If always predicts correct answer with lowest loss, then MRR = 1.0
    #       - Will be less than 1 if anything else
    mrr, mean_rank = compute_MR_MRR(results, losses)

    # Calculate Confidence Margin
    # The difference in model confidence (based on loss) between:
    #   - The top predicted choice (lowest loss), and
    #   - The second-best choice (next lowest loss)
    # 
    # Gives insigt into how decisively the model made its choice.
    # A large margin means the model was more confident
    # Small or negative margins can flag uncertain or risky decisions
    # Does not tell anything about the correct answer.
    #   However, if it does pick the correct answer (which can be seen by other metrics)
    #       a larger confidence margin will show that the model was more confident about that correct prediction
    # 
    # avg_margin - On average, how confident the model is
    # std_margin - How variable/confident the model is across questions
    # median_margin - Typical confidence level (less affected by outliers) 
    avg_margin, std_margin, median_margin, num_questions = compute_confidence_margin(losses)

    # accuracy determined by the logits
    accuracy = compute_accuracy(results)
    
    return {
        "accuracy": accuracy,
        "perplex_accuracy": perplexity_accuracy,
        "correct_perplex" : correct_perplexity,
        "all_perplex" : all_perplexity,
        "mean_rank": mean_rank,
        "MRR": mrr,
        "avg_confidence_margin": avg_margin,
        "std_confidence_margin": std_margin,
        "median_confidence_margin": median_margin,
        "num_questions": num_questions,
    }

# Run Evaluation with Trainer
quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True)

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
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # type of task to train on
    inference_mode=False, # set to False for training
    r=8, # dimension of the smaller matrices
    lora_alpha=32, # scaling factor
    lora_dropout=0.1 # dropout of LoRA layers
)

# Add LoraConfig to the model
model = get_peft_model(model, lora_config)

# NOTE: After first training
#model = PeftModel.from_pretrained(model, "./test/lora_adapter")

# Need to call this anytime the model is reloaded for training
# Esures:
#   - The LoRA weights are active
#   - Their gradients are enabled (requires_grad = True)
#   - You're not accidentally training a frozen model 
model.set_adapter("default")


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
    per_device_eval_batch_size=4, # For evaluation: 1 question, 4 choices, so need be in batched of 4
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



# Train
print(trainer.train())
# Need to manually save LoRA Adapter afterwords
model.save_pretrained("./test/lora_adapter")


# Evaluate
#print(trainer.evaluate())

