import torch
import copy
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
data_train = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")
data_train = data_train["train"].train_test_split(test_size=0.2)
data_validate_test = data_train["test"].train_test_split(test_size=0.5)
dataset = DatasetDict({
    "train": data_train["train"],
    "validate": data_validate_test["train"],
    "test": data_validate_test["test"]
})
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    model_max_length=512, #1024 originally, but too large for my GPU. Adjust depending on performance. Note in report that gpu limitations could affect final model.
    add_eos_token=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize_dataset(examples):
    prompts = [
        f"[INST] {q} [/INST]" 
        for q in examples["Question"]
    ]
    responses = examples["Response"]
    inputs = [p + " " + r for p, r in zip(prompts,responses)]
    tokenized = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    labels = copy.deepcopy(tokenized["input_ids"])
    for i, prompt in enumerate(prompts):
        tokenized_prompt = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
        )["input_ids"]
        prompt_len = len(tokenized_prompt)
        labels[i][:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels
    return tokenized
    
dataset = dataset.map(tokenize_dataset, batched=True, remove_columns=["Question", "Complex_CoT", "Response"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

"""
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
peft_model = get_peft_model(model, lora_config)
"""
model = PeftModel.from_pretrained(model, "./test3/lora_adapter")

training_args = TrainingArguments(
    output_dir="test4",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=4,
    #eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    label_names=["labels"],
    fp16=True,
    #fp16_full_eval=True,
    weight_decay=0.01
)

train_data = dataset["train"].select(range(5000))
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=dataset["validate"].select(range(1000)),
    processing_class=tokenizer,
    data_collator=data_collator
)
print(tokenizer.decode(train_data["input_ids"][0]))
trainer.train()
model.save_pretrained("test4/lora_adapter")
tokenizer.save_pretrained("test4/lora_adapter")


"""
Notes for future improvements:
- Try playing around with batch sizes and gradient accumulation.

- Try testing different max lengths. 600 too much? Too little? In the 
  text_gen_train file, we only use 128. Problem is that for context,
  we require larger lengths

- The big bottleneck with this dataset is the complex chain of thought column.
  Since the context is so long, the max length has to be long to accomodate it.
  However, a max length that is too long dramatically slows down training time
  or causes gpu out of memory errors in the worst case.
  Could experiment with methods to reduce the size of the context. Maybe it
  would be worthwhile to fully preprocess it before even loading the dataset.
  For example, could try feeding the context into an llm and generating a
  summary of it using < x characters. This would allow us to keep the most
  important bits of context, but make sure it is short enough to allow for
  efficient training. After preprocessing, could load dataset from locally
  saved version instead of original huggingface version

- At current, some datapoints might not be useful since the response is cut
  out due to the limited max length. Without the response, there isn't much
  benefit from the training.

Keep in mind for loading trained models:
peft_model = PeftModel.from_pretrained(model, "my_tests/lora_adapter")
tokenizer = AutoTokenizer.from_pretrained("my_tests/lora_adapter")
"""
