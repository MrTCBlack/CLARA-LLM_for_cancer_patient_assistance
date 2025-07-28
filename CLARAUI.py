import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer
from transformers import BitsAndBytesConfig
from peft import PeftModel
import torch
import threading

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
#model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")


#------------------------------------------
# This section would need to be changed based on how
#   you can run the model (i.e. hardware requirements)

# 4-bit quantization
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

#-----------------------------------------------

model = PeftModel.from_pretrained(model, "./final_adapter/lora_adapter")
# Need to call this anytime the adapter is loaded
# Esures:
#   - The LoRA weights are active
#   - Their gradients are enabled (requires_grad = True)
#   - You're not accidentally training a frozen model 
model.set_adapter("default")


def intro():
    # Initial prompt in order to give the LLM context for what it is supposed to do.
    persona = "\n\nYou are CLARA (Cancer Language and Response Assistant) here to help patients and care givers understand medical lingo.\n"
    instruction = "You will to receive a question from a user and your goal is to answer it to the best of your ability. Accuracy in answers is more important than the ability to respond.\n"
    context = "A patient or care giver has come to you to ask a question about cancer. They may not understand medical lingo and may need you to explain it in layman's terms.\n"
    audience = "You are answering questions from a user who is not familiar with medical lingo. They may be a patient or a care giver, but they do not have a medical background.\n"
    tone = "Your tone should be professional and informative. Try to be gentle in your words.\n"
    caution = """If you are asked a question that you do not know the answer to, let the user know that you do not know the answer and that they should consult a medical professional. 
                If they ask a question not related to medicine inform them that those questions are not within your ability to answer\n""" 
    query = persona + instruction + audience+ tone + context + caution
    
    context_prompt = [
        {"role": "system", "content": query}, 
        {"role": "user", "content": "Please introduce yourself to the user."}
    ]

    return  query

# Create streamer
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

MAX_TURNS = 4   # How many previous question-answers will be used as history

# Main chat function
def clara_chat(user_input, history):
    history = history[-MAX_TURNS:] # Limit history size

    # Build conversation prompt
    prompt_parts = [f"### Chat History:\n"]
    prompt_parts += [f"User:\n {user}\nCLARA:\n {assistant}" for user, assistant in history]
    prompt_parts += [intro()]
    prompt_parts += [f"\n\nUser:\n{user_input}\nCLARA: \n"]
    prompt="\n".join(prompt_parts)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Launch generation in a background thread
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        tokenizer=tokenizer,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.9,
        stop_strings=["User:", "CLARA:", "\nUser:"], #use this to stop CLARA from making it's own dialogue
    )
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.daemon = True
    thread.start()

    # Initialize history
    response = ""
    history.append((user_input, ""))

    tokens_count = 0
    # The previous threading and the yield loop simulate streaming
    for token in streamer:
        if token not in generation_kwargs["stop_strings"]:
            tokens_count += 1
            response += token
        history[-1] = (user_input, response)
        token_count = f"Token count: {tokens_count}"
        yield history, history, token_count, "" # Clear input box

# Clear chat function
def reset_chat():
    return [], [], "Chat reset."

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("CLARA: Cancer Language and Response Assistant")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask CLARA", placeholder="Who are you?")
    token_counter = gr.Markdown("Token count: 0")
    state = gr.State([])

    with gr.Row():
        submit_btn = msg.submit(clara_chat, [msg, state], [chatbot, state, token_counter, msg])
        reset_btn = gr.Button("🔁 Reset Chat")
        reset_btn.click(reset_chat, [], [chatbot, state, token_counter])

demo.launch()
