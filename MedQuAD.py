# Going to try on lavita/MedQuAD dataset
# A dataset of medical question and answers
# Will be used to train the model to predict longer answers
# 
# Focus on metrics that relate with long answer generation more than accuracy

from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import Trainer, TrainingArguments, EvalPrediction
from transformers import DataCollatorWithPadding
from transformers import BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from copy import deepcopy
import torch
import numpy as np
import evaluate
import threading


# Load lavita/MedQuAD dataset
# Number of records with question and answer
medQuAD = load_dataset("lavita/MedQuAD", split=["train[:16400]"])[0]

# Remove all examples that don't contain an answer
medQuAD = medQuAD.filter(lambda example: example["answer"]!=None)

# split the dataset's train split into a train and test set
# Seed is there for testing purposes
medQuAD = medQuAD.train_test_split(test_size=0.2, seed=41)

# Important parts of this dataset:
#   - question: The question being ask
#   - answer: The long response answer to the question

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

class DataCollatorForCausalLMWithCustomMasking:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.base_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")

    def __call__(self, features):
        # Separate labels so we can collate inputs and attention mask
        labels = [f["labels"] for f in features]
        for f in features:
            del f["labels"]

        # Use base collator to pad input_ids and attention_mask
        batch = self.base_collator(features)

        # Now pad labels manually
        max_len = max(len(label) for label in labels)
        padded_labels = []
        for label in labels:
            pad_len = max_len - len(label)
            padded = [-100] * pad_len + label
            padded_labels.append(padded)

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch

data_collator = DataCollatorForCausalLMWithCustomMasking(tokenizer)
#data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt", pad_to_multiple_of=8)

# NOTE: Training set
medQuAD_train = medQuAD["train"].shuffle().select(range(30))  # Select a subset for training; random
#medQuAD_train = medQA["train"].select(range(5))



def preprocess_train_function(examples):
    
    input_texts = []
    prompt_lens = []

    for question, answer in zip(examples["question"], examples["answer"]):
        prompt = f"Question:\n{question}\n\nAnswer:\n"
        prompt_lens.append(len(tokenizer(prompt)["input_ids"]))

        full_text = prompt + answer
        input_texts.append(full_text)

    # Tokenize
    tokenized = tokenizer(
        input_texts,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors=None,
        max_length=512,
        truncation=True,
    ) 
    

    masked_labels = []
    for idx, input_ids in enumerate(tokenized["input_ids"]):
        prompt_len = prompt_lens[idx]
        labels = input_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len
        masked_labels.append(labels)

    tokenized["labels"] = masked_labels
    #for input_ids, attention, labels in zip(tokenized["input_ids"], tokenized["attention_mask"], tokenized["labels"]):
    #    print(len(input_ids), len(attention), len(labels))

    return tokenized



tokenized_medQuAD_train = medQuAD_train.map(
    preprocess_train_function,
    batched=True,
    remove_columns=medQuAD_train.column_names,
)

#print(tokenized_medQuAD_train[0])
#print(tokenizer.decode(tokenized_medQuAD_train["input_ids"][0]))

'''lengths = []
for input_ids in tokenized_medQuAD_train["input_ids"]:
    lengths.append(len(input_ids))

lengths = np.array(lengths)
Q1 = np.percentile(lengths, 25)
Q2 = np.percentile(lengths, 50)
Q3 = np.percentile(lengths, 75)
mean = np.mean(lengths)
print(Q1, Q2, Q3, mean)'''

# NOTE: Testing Generation set
#medQuAD_test = medQA["test"].shuffle().select(range(1000))  # Select a subset for testing; random
medQuAD_test = medQuAD["test"].select(range(5))


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
        #choices_block = "\n".join([f"{key}. {choice}" for key, choice in examples["options"][i].items()])
        prompt = f"Question:\n{examples['question'][i]}\n\nAnswer:\n"
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


tokenized_medQuAD_generate = medQuAD_test.map(
    preprocess_generation_function, 
    batched=True,
    remove_columns=medQuAD_test.column_names)

#print(tokenized_medQuAD_generate[0])
#print(tokenizer.decode(tokenized_medQuAD_generate["input_ids"][0]))




# Normalized match
def normalize(text):
    return text.lower().strip()

# This is the evaluation for metrics of what the model generates
def evaluate_model_generation(model, tokenizer):
    model.eval()
    eval_dataset = deepcopy(tokenized_medQuAD_generate)
    eval_dataset = eval_dataset.remove_columns(["gold_answer_text"])

    all_preds = []
    gold_answers = [answer.strip() for answer in tokenized_medQuAD_generate["gold_answer_text"]]
    questions = []

    for i in range(len(eval_dataset["input_ids"])):
        input_ids = torch.tensor(eval_dataset["input_ids"][i]).unsqueeze(0).to(model.device)
        attention_mask = torch.tensor(eval_dataset["attention_mask"][i]).unsqueeze(0).to(model.device)

        question = tokenizer.decode(eval_dataset["input_ids"][i])
        questions.append(question)
        print(question)

        # NOTE: Streamer is for if you want to see what it is outputting before
        #   it is postprocessed to compare to the expected value
        '''streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = threading.Thread(target=model.generate, kwargs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 500,
            "streamer": streamer,
        })
        thread.start()

        # Collect the output from the streamer
        decoded_tokens = ""
        for token in streamer:
            print(token, end="", flush=True)  # stream to console
            decoded_tokens += token

        print()  # new line

        all_preds.append(decoded_tokens.strip())'''

        # NOTE: This is for if you don't care about seeing the generated output as it's generating it
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=500,
        )
        
        new_tokens = generated_ids[0, input_ids.shape[1]:]
        preds = tokenizer.decode(new_tokens, skip_special_tokens=True)

        all_preds.append(preds)

    #correct = []
    print()
    #for i in range(len(all_preds)):
    #    print(f"Prediction: {all_preds[i]}")
    #    print(f"Expected: {gold_answers[i]}\n")

    bleu = evaluate.load('bleu')
    bleu_scores = []
    for pred, ref in zip(all_preds, gold_answers):
        bleu_result = bleu.compute(predictions=[pred],
                                   references=[[ref]],)
                                   #tokenizer=tokenizer)
        bleu_scores.append(bleu_result)
    #print(bleu_scores)

    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=all_preds,
                                  references=gold_answers,
                                  use_aggregator=False)
    #print(rouge_results)

    meteor = evaluate.load("meteor")
    meteor_scores = []
    for pred, ref in zip(all_preds, gold_answers):
        meteor_result = meteor.compute(predictions=[pred],
                                   references=[[ref]])
        meteor_scores.append(meteor_result)
    #print(meteor_scores)


    bertscore = evaluate.load("bertscore")
    bertscore_results = bertscore.compute(predictions=all_preds,
                                            references=gold_answers,
                                            lang="en")
    #print(bertscore_results)

    results = {
        "bleu": bleu_scores,
        "rouge": rouge_results,
        "meteor": meteor_scores,
        "bertscore": bertscore_results,
        "questions": questions,
        "predictions": all_preds,
        "references": gold_answers,
    }


    return results




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

model.config.pad_token_id = tokenizer.pad_token_id


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
model = PeftModel.from_pretrained(model, "./test3/lora_adapter")

# Need to call this anytime the adapter is loaded
# Esures:
#   - The LoRA weights are active
#   - Their gradients are enabled (requires_grad = True)
#   - You're not accidentally training a frozen model 
model.set_adapter("default")


# Define training hyperparameters in TrainingArguments
training_args = TrainingArguments(
    output_dir="test3",  # Where to save checkpoints & logs
    overwrite_output_dir=True, # Overwrite existing output_dir if needed

    # Training-specific
    do_train=True,
    #num_train_epochs=3, # Set according to your needs
    per_device_train_batch_size=1,  # Match or adjust for GPU memory
    learning_rate=2e-5, # Typical starting point for transformers
    weight_decay=0.01,  # Optional: helps regularize the model

    #Evaluation
    do_eval=False, #No evaluation

    #prediction_loss_only=False,

    save_strategy="epoch",  #Optional: save checkpoint after each epoch
    logging_strategy="epoch",   #Optional: log once per epoch

    # Other useful options
    #save_total_limit=2, # Limit number of saved checkpoints
    gradient_accumulation_steps=1,
    label_names=["labels"],
    fp16=True,
    #fp16_full_eval=True
)

# Pass the training arguments to Trainer along with the model, datasets, and data collator
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_medQuAD_train,
    data_collator=data_collator,
    processing_class=tokenizer,
)



# NOTE: Train
#print(trainer.train())
# Need to manually save LoRA Adapter afterwords
#model.save_pretrained("./test3/lora_adapter")


# NOTE: Evaluate
# Generation phase
results = evaluate_model_generation(model, tokenizer)

questions = results["questions"]
predictions = results["predictions"]
references = results["references"]

bleu_scores = results["bleu"]
rouge_scores = results["rouge"]
meteor_scores = results["meteor"]
bertscores = results["bertscore"]

print()
for i in range(len(questions)):
    print(f"Question {i+1}:\n{questions[i]}\n")
    print(f"Prediction:\n{predictions[i]}\n")
    print(f"Reference:\n{references[i]}\n\n")

    print(f"BLEU Score: {bleu_scores[i]["bleu"]}")
    print(f"BLEU 1-gram: {bleu_scores[i]["precisions"][0]}")
    print(f"BLEU 2-gram: {bleu_scores[i]["precisions"][1]}")
    print(f"BLEU 3-gram: {bleu_scores[i]["precisions"][2]}")
    print(f"BLEU 4-gram: {bleu_scores[i]["precisions"][3]}")
    print(f"Prediction Length: {bleu_scores[i]["translation_length"]}")
    print(f"Reference Length: {bleu_scores[i]["reference_length"]}")
    print(f"Brevity Penalty: {bleu_scores[i]["brevity_penalty"]}\n")

    print(print(f"ROUGE1: {rouge_scores["rouge1"][i]}"))
    print(print(f"ROUGE2: {rouge_scores["rouge2"][i]}"))
    print(print(f"ROUGEL: {rouge_scores["rougeL"][i]}"))
    print(print(f"ROUGELsum: {rouge_scores["rougeLsum"][i]}\n"))

    print(f"METEOR: {meteor_scores[i]["meteor"]}\n")

    print(print(f"BERTscore Precision: {bertscores["precision"][i]}"))
    print(print(f"BERTscore Recall: {bertscores["recall"][i]}"))
    print(print(f"BERTscore f1: {bertscores["f1"][i]}\n"))

