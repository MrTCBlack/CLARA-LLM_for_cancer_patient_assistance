import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer, pipeline
from peft import PeftModel

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
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
peft_model = PeftModel.from_pretrained(model, "my_tests/lora_adapter") # comment out this line if want to test base model
tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    max_new_tokens=1500,
    do_sample=False,
    streamer=streamer
)

# actual question from dataset
#pipe("A 45-year-old man with a history of alcohol use, who has been abstinent for the past 10 years, presents with sudden onset dysarthria, shuffling gait, and intention tremors. Given this clinical presentation and history, what is the most likely diagnosis?")


# tests
pipe("A 45-year-old man with a history of alcohol use, who has been abstinent for the past 10 years, presents with sudden onset dysarthria, shuffling gait, and intention tremors. What is the most likely diagnosis for this man?")
print()
pipe("A 45-year-old man with a history of alcohol use, who has been abstinent for the past 10 years, presents with sudden onset dysarthria, shuffling gait, and intention tremors. What is this man likely to be diagnosed with?")