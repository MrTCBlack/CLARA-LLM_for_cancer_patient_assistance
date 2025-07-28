#Chapter 6
# section: Advanced prompt Engineering
# A bunch of advanced prompt engineering methods
# Uses model from previous section, but shows off each method

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
from transformers import BitsAndBytesConfig
from transformers import HqqConfig
from transformers import TorchAoConfig
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, Int8WeightOnlyConfig
from torchao.quantization import Int4WeightOnlyConfig
from torchao.dtypes import Int4CPULayout
import time


def potential_complexity(tokenizer, pipe):
    """
    Use the complex prompt to add and/or remove parts to observe its impact
    on the generated output
    """

    # Text to summarize which we stole from https://jalammar.github.io/illustrated-transformer/ ;)
    text = """In the previous post, we looked at Attention - a ubiquitous method in modern deep learning models. Attention is a concept that helped improve the performance of neural machine translation applications. In this post, we will look at The Transformer - a model that uses attention to boost the speed with which these models can be trained. The Transformer outperforms the Google Neural Machine Translation model in specific tasks. The biggest benefit, however, comes from how The Transformer lends itself to parallelization. It is in fact Google Cloud's recommendation to use The Transformer as a reference model to use their Cloud TPU offering. So let's try to break the model apart and look at how it functions.
        The Transformer was proposed in the paper Attention is All You Need. A TensorFlow implementation of it is available as a part of the Tensor2Tensor package. Harvard's NLP group created a guide annotating the paper with PyTorch implementation. In this post, we will attempt to oversimplify things a bit and introduce the concepts one by one to hopefully make it easier to understand to people without in-depth knowledge of the subject matter.
        Let's begin by looking at the model as a single black box. In a machine translation application, it would take a sentence in one language, and output its translation in another.
        Popping open that Optimus Prime goodness, we see an encoding component, a decoding component, and connections between them.
        The encoding component is a stack of encoders (the paper stacks six of them on top of each other - there's nothing magical about the number six, one can definitely experiment with other arrangements). The decoding component is a stack of decoders of the same number.
        The encoders are all identical in structure (yet they do not share weights). Each one is broken down into two sub-layers:
        The encoder's inputs first flow through a self-attention layer - a layer that helps the encoder look at other words in the input sentence as it encodes a specific word. We'll look closer at self-attention later in the post.
        The outputs of the self-attention layer are fed to a feed-forward neural network. The exact same feed-forward network is independently applied to each position.
        The decoder has both those layers, but between them is an attention layer that helps the decoder focus on relevant parts of the input sentence (similar what attention does in seq2seq models).
        Now that we've seen the major components of the model, let's start to look at the various vectors/tensors and how they flow between these components to turn the input of a trained model into an output.
        As is the case in NLP applications in general, we begin by turning each input word into a vector using an embedding algorithm.
        Each word is embedded into a vector of size 512. We'll represent those vectors with these simple boxes.
        The embedding only happens in the bottom-most encoder. The abstraction that is common to all the encoders is that they receive a list of vectors each of the size 512 - In the bottom encoder that would be the word embeddings, but in other encoders, it would be the output of the encoder that's directly below. The size of this list is hyperparameter we can set - basically it would be the length of the longest sentence in our training dataset.
        After embedding the words in our input sequence, each of them flows through each of the two layers of the encoder.
        Here we begin to see one key property of the Transformer, which is that the word in each position flows through its own path in the encoder. There are dependencies between these paths in the self-attention layer. The feed-forward layer does not have those dependencies, however, and thus the various paths can be executed in parallel while flowing through the feed-forward layer.
        Next, we'll switch up the example to a shorter sentence and we'll look at what happens in each sub-layer of the encoder.
        Now We're Encoding!
        As we've mentioned already, an encoder receives a list of vectors as input. It processes this list by passing these vectors into a 'self-attention' layer, then into a feed-forward neural network, then sends out the output upwards to the next encoder.
        """

    # Prompt components
    persona = "You are an expert in Large Language models. You excel at breaking down complex papers into digestible summaries.\n"
    instruction = "Summarize the key findings of the paper provided.\n"
    context = "Your summary should extract the most crucial points that can help researchers quickly understand the most vital information of the paper.\n"
    data_format = "Create a bullet-point summary that outlines the method. Follow this up with a concise paragraph that encapsulates the main results.\n"
    audience = "The summary is designed for busy researchers that quickly need to grasp the newest trends in Large Language Models.\n"
    tone = "The tone should be professional and clear.\n"
    #text = "MY TEXT TO SUMMARIZE"  # Replace with your own text to summarize
    data = f"Text to summarize: {text}"

    # The full prompt - remove and add pieces to view its impact on the generated output
    query = persona + instruction + context + data_format + audience + tone + data

    messages = [
        {"role": "user", "content": query}
    ]
    print(tokenizer.apply_chat_template(messages, tokenize=False))
    print()

    # Generate the output
    start_time = time.time()
    outputs = pipe(messages)
    end_time = time.time()
    #print(outputs[0]["generated_text"])
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} seconds")


def in_context_learn(tokenizer, pipe):
    """
    To do so, we will need to differentiate between our question (user)
        and the answer that were provided by the model (assistant).
    We additionally showcase how this interaction is processed using the template:
    """

    # Use a single example of using the made-up word in a sentence
    one_shot_prompt = [
        {
            "role": "user",
            "content": "A 'Gigamuru' is a type of Japanese musical instrument. An example of a sentence that uses the word Gigamuru is:"
        },
        {
            "role": "assistant",
            "content": "I have a Gigamuru that my uncle gave me as a gift. I love to play it at home."
        },
        {
            "role": "user",
            "content": "To 'screeg' something is to swing a sword at it. An example of a sentence that uses the word screeg is:"
        }
    ]
    print(tokenizer.apply_chat_template(one_shot_prompt, tokenize=False))
    print()

    """
    The prompt illustrates the need to differentiate between the user and the assistant. 
    If we did not, it would seem as if we were talking to ourselves. 
    Using these interactions, we can generate output as follows:
    """

    # Generate the output
    start_time = time.time()
    outputs = pipe(one_shot_prompt)
    end_time = time.time()
    #print(outputs[0]["generated_text"])
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} seconds")


def chain_prompt(pipe):
    """
    This technique of chaining prompts allows the LLM to spend more time on each
        individual question instead of tackling the whole problem.
    Let us illustrate this with a small example. We first create a name and slogan for a chatbot:
    """

    # Create name and slogan for a product
    product_prompt = [
        {"role": "user", "content": "Create a name and slogan for a chatbot that leverages LLMs."}
    ]
    start_time = time.time()
    outputs = pipe(product_prompt)
    end_time = time.time()
    product_description = outputs[0]["generated_text"]
    #print(product_description)
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} seconds")
    print()


    # Then, we can use the generated output as input for the LLM to generate a sales pitch:

    # Based on a name and slogan for a product, generate a sales pitch
    sales_prompt = [
        {"role": "user", "content": f"Generate a very short sales pitch for the following product: '{product_description}'"}
    ]
    start_time = time.time()
    outputs = pipe(sales_prompt)
    end_time = time.time()
    sales_pitch = outputs[0]["generated_text"]
    #print(sales_pitch)
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.4f} seconds")




def main():
    # No Calibration Required (On-the-fly Quantization)
    # These methods are generally easier to use as they don't need a separate calibration dataset or step

    #NOTE: I originally ran all of these on the model: microsoft/Phi-3-mini-4k-instruct
    # I have only tested out the Mistral-7B-Instruct-v0.3 model with the bnb quantization

    # bitsandbytes: 8-bit and 4-bit
    # 8-bit:
    #    cpu:
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    #
    #    cuda:
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    #  
    #    auto: For me it required offloading; may not for you
    #  8-bit models can offload weights between the CPU and GPU to fit
    #   very large models into memory.
    # The weights dispatched to the CPU are stored in float32 and aren't
    #   converted to 8-bit.
    # For this to be done, need different quanization_config and device_map
    # NOTE: Can design device map, but 'auto' seems to do just as well;
    # Design device map to fit everything on GPU except for the lm_head, which is dispatched to the CPU
    '''quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
        )
    model_8bit_auto = AutoModelForCausalLM.from_pretrained(
        "mistralai/mistral-7b-instruct-v0.3",
        device_map='auto',
        quantization_config=quantization_config,
        trust_remote_code=False,
    )'''

    # 4-bit:
    #    cpu: requires nf4 on CPU, not fp4:
    # NF4 is a 4-bit data type, adapted for weights initialized from a normal distribution.
    # Should use NF4 for training 4-bit base models when using cpu
    '''nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model_4bit_cpu = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        device_map="cpu",
        quantization_config=nf4_config,
        trust_remote_code=False,
    )'''
    #
    #   cuda:
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    #  
    #   auto:
    '''quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16)'''
    

    # HQQ (Half-Quadratic Quantization)
    # NOTE: I never ran this on the Mistral-7B-Instruct-v0.3 model
    # NOTE: With the Phi-3-mini-4k-instruct model, I could only get it to run on GPU only
    #  
    # 8-bit:
    #   cpu: NOTE: It took a real long time to load, but didn't run
    #   cuda: 
    #   auto: NOTE: Was not able to load or run
    #quantization_config = HqqConfig(nbits=8, group_size=64)

    # 4-bit:
    #   cpu: NOTE: Loaded, but didn't run
    #   cuda:
    #   auto: NOTE: Was not able to load or run
    #quantization_config = HqqConfig(nbits=4, group_size=64)


    # torchao
    # NOTE: I never ran this on the Mistral-7B-Instruct-v0.3 model
    # NOTE: With the Phi-3-mini-4k-instruct model, I could not get it to work
    #  
    # int8-bit-dynamic:
    #   NOTE: Doesn't seem to work
    #quantization_config = TorchAoConfig(quant_type=Int8DynamicActivationInt8WeightConfig())

    # int8-bit-weight-only:
    #   NOTE: Doesn't seem to work
    #quantization_config = TorchAoConfig(quant_type=Int8WeightOnlyConfig())

    # int4-bit-weight-only:
    #   
    #quant_config = Int4WeightOnlyConfig(group_size=128, layout=Int4CPULayout())
    #quantization_config = TorchAoConfig(quant_type=quant_config)

    # autoquant:
    #  NOTE: Doesn't seem to work
    #quantization_config = TorchAoConfig("autoquant", min_sqnr=None)


    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/mistral-7b-instruct-v0.3",
        device_map="auto",
        quantization_config=quantization_config,
        # you shouldn't use troch_dtype if you are doing quantization
        #torch_dtype="auto",
        trust_remote_code=False,
        # This is specifically designed for my system, so you may need to adjust it
        # to fit your system's memory constraints
        # Note: If you use auto, it will automatically adjust the memory footprint
        # and use the GPU if available, otherwise it will use the CPU
        # 0 stands for the first GPU I have, which is cuda:0
        max_memory={
            0: "4GiB",
            "cpu": "28GiB"
        }
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/mistral-7b-instruct-v0.3")

    streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Create a pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=500,
        do_sample=True,
        streamer=streamer,
    )

    
    #print()
    #Use the below method to check how much memory the model is taking up
    # Check before and after different quantization methods
    #print(model.get_memory_footprint())

    # Use this method to see how the layers of the model are distributed across the devices (CPU and GPU)
    #print(model.hf_device_map)
    print()

    # The method has a large prompt that could take the LLM longer to process
    #potential_complexity(tokenizer, pipe)
    
    # This method uses in-context learning to teach the LLM a new word
    #in_context_learn(tokenizer, pipe) 

    # This method chains prompts together to allow the LLM to focus on each part of the problem
    #chain_prompt(pipe)






main()