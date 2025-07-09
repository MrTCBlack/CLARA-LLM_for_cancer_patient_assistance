# Code for the CLARA model

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from evaluate import evaluator
import torch

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    device_map="auto",
    torch_dtype=torch.float16,
    )

def prolog():
    print("CLARA model loaded successfully.")
    while input != "quit":
        user_input = input("Enter your query: ")
        if user_input.lower() == "quit":
            break
        else:
            print("Processing your query...")
    print("Exiting CLARA model.")