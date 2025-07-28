import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


def intro():
    # Initial prompt in order to give the LLM context for what it is supposed to do.
    persona = "System: You are CLARA (Cancer Language and Response Assistant) here to help patients and care givers understand medical lingo.\n"
    instruction = "You will receive a question from a user and your goal is to answer it to the best of your ability. Accuracy in answers is more important than the ability to respond\n"
    context = "A patient or care giver has come to you with a question about cancer. They may not understand medical lingo and may need you to explain it in layman's terms.\n"
    audience = "You are talking to a user who is not familiar with medical lingo. They may be a patient or a care giver, but they do not have a medical background.\n"
    tone = "Your tone should be professional and informative. Try to be gentle in your words\n"
    caution = """If you are asked a question that you do not know the answer to, let the user know that you do not know the answer and that they should consult a medical professional. 
                If they ask a question not related to medicine inform them that those questions are not within your ability to answer\n"""
    query = persona + instruction + audience+ tone + context + caution
    
    context_prompt = [
        {"role": "system", "content": query}, 
        {"role": "user", "content": "Please introduce yourself to the user."}
    ]

    return  query

# Main chat function
def clara_chat(user_input, history):
    # Build conversation prompt
    prompt = intro()
    for user, assistant in history:
        prompt += f"User: {user}\nAssistant: {assistant}\n"
    prompt += f"User: {user_input}\nAssistant:"

    # Generate response
    output = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=150,
        temperature=0.9,
        return_full_text=False
    )[0]["generated_text"]

    # Extract assistant response
    response = output.strip().split("\n")[0]

    # Token count
    tokens = tokenizer.encode(response)
    token_info = f"Token count: {len(tokens)}"

    # Append and return
    history.append((user_input, response))
    print(response)  # For debugging
    return history, history, token_info, ""

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
