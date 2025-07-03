# This is an example program for a chat bot
# It is to show ofthe capabilites of what I have learned so far
#   in using the Huggingface libaries.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
import time

def prolog():
    print()
    print("Welcome to this example protype of a simple chat bot.")
    print("This is meant to just show off how the Huggingface libraries "+
          "can be used to make a simple chat bot.")
    print("It also shows off some prompt engineering techniques that can "+
          "get better output form an LLM.")
    print("Each output from the LLM is timed and the time is output"+
          " after the output from the LLM. This is just a metric.\n")
    
    print("INSTRUCTIONS:")
    print(" - Enter your question and wait for a response.")
    print(" - When you are done chatting, enter 'quit()' to quit.\n")

    print("WARNING: This is probably very slow if you are just running on a laptop, like I am.")
    print("Please be patient as you are using the chatbot.\n")

    input("Press RETURN to begin chatting")
    


def intro(pipe):

    # Initial prompt in order to give the LLM context for what it is supposed to do.
    # I don't believe that it will remember this prompt once the look starts
    # For it to rememebr, you may need to give it the context with every prompt
    persona = "You are a conversational assistant designed to answer user questions.\n"
    instruction = "You will receive a question from a user and your goal is to answer it to the best of your ability.\n"
    context = "Start by introducing yourself to the user."
    query = persona + instruction + context

    context_prompt = [
        {"role": "system", "content": query}
    ]

    print()
    start_time = time.time()
    outputs=pipe(context_prompt, do_sample=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} seconds") # Time is printed for metric sake
    print()

    return outputs



def have_conversation(pipe, outputs):
    user_input = input("Enter your question> ")
    while (user_input != "quit()"):
        print()

        # Having the previous output as part of the new prompt gives
        #   provides the LLM with memory of it's response to the previous question
        # This allows you to ask follow up questions and for the LLM to
        #   guage what it's answer should be based off of it's last output
        # However, It doesn't have memory of anything before the last response
        # Also, including the last response in the new prompt increases the amount
        #   of time it take for the LLM to process the prompt, especially if
        #   the last response was long.
        # However, this is a tradeoff for being able to have simulated conversation
        previous_output = f"Your response to my previous question was: '{outputs[0]["generated_text"]}'.\n"
        user_question = f"My next question is: '{user_input}'"
        instruction = f"Provide an answer to my question."
        user_query = previous_output + user_question + instruction
        user_prompt = [
            {"role": "user", "content": user_query}
        ]

        start_time = time.time()
        outputs=pipe(user_prompt, do_sample=True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.4f} seconds")
        print()
        user_input = input("Enter your question> ")
    print()


def outro(pipe):
    # Just some epiloge from the LLM after you have quit
    epiloge_prompt = [
        {"role": "system", "content": "The user has finished asking questions. Thank them for asking you questions and wish them a good day."}
    ]

    start_time = time.time()
    outputs=pipe(epiloge_prompt, do_sample=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} seconds")
    print()




def main():

    # Load model and tokenizer
    # For this example we will be using microsoft/Phi-3-mini-4k-instruct
    #   The link to this model can be found here: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map="cpu",   #I am using cpu here because that is what I have found runs best
                            #If you have CUDA capabilities for GPUs and are able to use them
                            #   for better performance, use device_map="auto" or device_map="cuda"
        torch_dtype="auto",
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    # Create text streamer - outputs each word to stdout as it is generated, rather than one large output at the end
    streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Create a pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=500,
        do_sample=False,
        streamer=streamer
    )


    '''chat = [
        {"role": "system", "content": "You are playing the role of a wise-cracking, nice cop from a 1970's buddy-cop commedy. Give a proper response to the user's question as that role."},
        {"role": "user", "content": "Hey, what was this criminal brought into the station for?"}
    ]'''

    prolog()

    outputs = intro(pipe)

    have_conversation(pipe, outputs)

    outro(pipe)




main()
