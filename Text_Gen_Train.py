# Practice training a causal language model
# Using the Mistral Model

# The guide shows how to:
#   1. Finetune DistilGPT2 (I will be using Mistral-7B) on the r/askscience subset of the ELI5 dataset
#   2. Use the finetuned model for inference

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
import torch
import math



# Load ELI5 dataset
eli5 = load_dataset("eli5_category", split="train[:50]", trust_remote_code=True)

# spit the dataset's train split into a train and test set
eli5 = eli5.train_test_split(test_size=0.2)

#print(eli5["train"][0])

# What stuff in this set means:
#   - Only really interested in the text field
#   - For language modeling tasks, don't need labels (also known as an unsupervised task)
#       because the next word is the label

# Preprocess
# import Mistral-7B stuff
# tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

#print()

# need to extract the text subfield from its nested structure
eli5 = eli5.flatten()
#print(eli5["train"][0])

# convert the list of text to a string so they can be jointly tokenized
# Preprocessing funciton to join the list of strings for each example and tokenize the result:
def preprocess_function(examples):
    return tokenizer([" ".join(text) for text in examples["answers.text"]])

# Use map to apply this preprocessing function over the entire dataset
# Speed up map by setting batched=True to process multiple elements of the dataset at once
# Summary: Tokenize the important text and only leave the input_ids and attention_mask columns
tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    remove_columns=eli5["train"].column_names,
)

# This dataset contains the token sequences, but some of these are longer than
#   the maximum input length for the model
#  Use the second preprocessing function to:
#   - concatenate all the sequences
#   - split the concatenated sequences into shorter chunks defined by block_size

block_size = 128

def group_texts(examples):
    # examples is a list of dictionaries, each dictionary contains a list of lists of input_ids and attention_mask

    # Concatenate all texts.
    # This merges all sublists into a single flat list for each key (e.g., input_ids, attention_mask).
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

    # Total_length gets the length of the concatenated input_ids (or any key -
    #   they're assumed to be of equal length)
    total_length = len(concatenated_examples[list(examples.keys())[0]])

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
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    # Duplicate input_ids as labels
    # For causal language modeling, the model is trained to predict
    #   the next token in the sequence
    # So the input and labels are the same, just shifted during training  
    result["labels"] = result["input_ids"].copy()

    return result

lm_dataset = tokenized_eli5.map(group_texts, batched=True)

# create a batch of examples
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Train

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

# Low-Rank Adaptation (LoRA) is a very common parameter-efficient fine-tuning (PEFT) method
#   Decomposes the weight matrix into two smmaller trainable matrices.
# Start by defining a LoraConfig object with the parameters shown below

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
    output_dir="test",
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
)

# Pass the training arguments to Trainer along with the model, datasets, and data collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
    processing_class=tokenizer,
)

#trainer.train()


# Evaluate the model
# Got rid of model, am now going to try to redownload it and evaluate
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

