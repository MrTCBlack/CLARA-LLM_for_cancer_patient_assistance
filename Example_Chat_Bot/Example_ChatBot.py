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
    
    print("There are three different scenarios that the LLM has been prompted for.")
    print("They are as follows:")
    print("\tScenario 1 - Have a conversation with a assistant meant to answer your general questions.")
    print("\tScenario 2 - You are on a wildlife safari. Ask the guide about animals and the safari.")
    print("\tScenario 3 - You are playing a game of Dungeons & Dragons. The assistant is the Dungeon Master leading you through your adventure.\n")

    print("Pick one of the above scenarions\n Enter '1' for scenario 1, '2' for scenario 2, or '3' for scenario 3\n")
    inputScenario = input()
    scenario = -1
    while (scenario == -1):
        if (inputScenario.__contains__("1")):
            scenario = 1
        elif (inputScenario.__contains__("2")):
            scenario = 2
        elif (inputScenario.__contains__("3")):
            scenario = 3
        else:
            print("\tScenario 1 - Have a conversation with a assistant meant to answer your general questions.")
            print("\tScenario 2 - You are on a wildlife safari. Ask the guide about animals and the safari.")
            print("\tScenario 3 - You are playing a game of Dungeons & Dragons. The assistant is the Dungeon Master leading you through your adventure.\n")

            print("Pick one of the above scenarions\n Enter '1' for scenario 1, '2' for scenario 2, or '3' for scenario 3\n")
            inputScenario = input()
    print()

    if (scenario == 1):
        print("Thank you for choosing Scenario 1.")
    elif (scenario == 2):
        print("Thank you for choosing Scenario 2.")
    else:
        print("Thank you for choosing Scenario 3.")
    
    print("INSTRUCTIONS:")
    if (scenario == 1):
        print(" - Enter your question and wait for a response.")
    elif (scenario == 2):
        print(" - Enter your question about animals or your observations of the safari and wait for a response.")
        print(" - Responses that are not about animals or the safari will be ignored")
    else:
        print(" - Respond to the dungeon master and wait for a response.")
        print(" - Responses that are not relevant to the adventure will be ignored")
    print(" - When you are done chatting, enter 'quit()' to quit.\n")

    print("WARNING: This is probably very slow if you are just running on a laptop, like I am.")
    print("Please be patient as you are using the chatbot.\n")

    input("Press RETURN to begin chatting")
    return scenario
    


def intro(pipe, scenario):

    if (scenario == 1):
        # Initial prompt in order to give the LLM context for what it is supposed to do.
        # I don't believe that it will remember this prompt once the look starts
        # For it to rememebr, you may need to give it the context with every prompt
        persona = "You are a conversational assistant designed to answer user questions.\n"
        instruction = "You will receive a question from a user and your goal is to answer it to the best of your ability.\n"
        context = "Start by introducing yourself to the user."
        query = persona + instruction + context
    elif (scenario == 2):
        # This is a scenario for the LLM as an expert zookeeper
        persona = "You are an expert zoologist giving a safari tour. You are great at explaining intricate details about animals in an fun, concise, and understandable fashion.\n"
        instruction = "Wait for the people on the toor to ask you quesitons and then give fun and understandable answers the questions about animals.\n"
        context = "To prompt the members of the safari, make observations about the environment around you.\n"
        audience = "Your responses are designed for someone who loves animals but doesn't know a lot about them. They need your help explaining information.\n"
        tone = "Your tone should be friendly and enthusiastic.\n"
        caution = "If you are asked questions that do not relate to animals or the safari, let the asker know that you are not familiar with the subject and can not give a good answer to it.\n"
        query = persona + instruction + audience + tone + caution
    else:
        # This is a scenario where the LLM is a dungeon master leading a game of Dungeons & Dragons
        persona = "You are the Dungeon Master leading a game of Dungeons & Dragons. As Dungeon master, you tell the story of the adventure, and make sure to give the user choices that you pause and wait to hear the answer to. Those choices will influence the story you tell.\n"
        instruction = "Your job is to give the user choices to make that will influence their adventure. Give those choices and let the user respond. Do not respond for the user.\n"
        context = "You should describe the adventure as if you were telling a story that the user is the main character of. It is important that you ask the user to make decisions about the story.\n"
        audience = "Give the user chances to make decisions about the adventure by giving the user chances to make choices.\n"
        tone = "The tone should be mysterious and exciting, and the tale you tell should twists and turns.\n"
        caution = "If the user gives responses that are not relevant to the adventure, you should keep them on track while staying in character.\n"
        query = persona + instruction + context + audience + tone + caution

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



def have_conversation(pipe, outputs, scenario):
    user_input = input("Enter your question> ")
    while (user_input != "quit()"):
        print()

        

        if (scenario == 1): 
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
        elif (scenario == 2):
            persona = "You are an expert zoologist giving a safari tour. You are great at explaining intricate details about animals in an fun, concise, and understandable fashion.\n"
            caution = "If you are asked questions that do not relate to animals or the safari, let the asker know that you are not familiar with the subject and can not give a good answer to it.\n"
            system_context = persona + caution

            previous_output = f"Your response to the previous question was: '{outputs[0]["generated_text"]}'.\n"
            user_question = f"The next question from the member of the safari is: '{user_input}'"
            instruction = f"Provide an answer to the members question."
            user_query = previous_output + user_question + instruction

            user_prompt = [
                {"role": "system", "content": system_context},
                {"role": "user", "content": user_query}
            ]
        else:
            persona = "You are the Dungeon Master leading a game of Dungeons & Dragons. As Dungeon master, you tell the story of the adventure, and make sure to give the user choices that you pause and wait to hear the answer to. Those choices will influence the story you tell.\n"
            caution = "If the user gives responses that are not relevant to the adventure, you should keep them on track while staying in character.\n"
            system_context = persona + caution

            previous_output = f"You had told the user the following about their adventure: '{outputs[0]["generated_text"]}'.\n"
            user_question = f"The user made the following decision: '{user_input}'"
            instruction = f"Continue with the adventure based on the user's decision."
            user_query = previous_output + user_question + instruction

            user_prompt = [
                {"role": "system", "content": system_context},
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


def outro(pipe, scenario):
    
    if (scenario == 1):
        # Just some epiloge from the LLM after you have quit
        epiloge_prompt = [
            {"role": "system", "content": "The user has finished asking questions. Thank them for asking you questions and wish them a good day."}
        ]
    elif (scenario == 2):
        persona = "You are an expert zoologist giving a safari tour. You are great at explaining intricate details about animals in an fun, concise, and understandable fashion.\n"
        caution = "If you are asked questions that do not relate to animals or the safari, let the asker know that you are not familiar with the subject and can not give a good answer to it.\n"
        system_context = persona + caution
        epiloge_prompt = [
            {"role": "system", "content": system_context + "The safari has finished. Thank the guests for coming on the tour and wish them a good day."}
        ]
    else:
        persona = "You are the Dungeon Master leading a game of Dungeons & Dragons. As Dungeon master, you tell the story of the adventure, and make sure to give the user choices that you pause and wait to hear the answer to. Those choices will influence the story you tell.\n"
        caution = "If the user gives responses that are not relevant to the adventure, you should keep them on track while staying in character.\n"
        system_context = persona + caution
        epiloge_prompt = [
            {"role": "system", "content": system_context + "The user wants to end the adventure. Wrap up their story and thank them for joining you."}
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

    scenario = prolog()

    outputs = intro(pipe, scenario)

    have_conversation(pipe, outputs, scenario)

    outro(pipe, scenario)




main()
