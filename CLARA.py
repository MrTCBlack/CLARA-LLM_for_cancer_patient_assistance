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
peft_model = PeftModel.from_pretrained(model, "test3/lora_adapter") # comment out this line if want to test base model
tokenizer = AutoTokenizer.from_pretrained(model_id)
streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True)

pipe = pipeline(
    "text-generation",
    model=peft_model,
    tokenizer=tokenizer,
    return_full_text=True,
    max_new_tokens=1500,
    do_sample=False,
    streamer=streamer
)

# actual question from dataset
#pipe("Question: Given the following context, what is the most likely diagnosis? Context: A 45-year-old man with a history of alcohol use, who has been abstinent for the past 10 years, presents with sudden onset dysarthria, shuffling gait, and intention tremors.")


# tests
#pipe("A 45-year-old man with a history of alcohol use, who has been abstinent for the past 10 years, presents with sudden onset dysarthria, shuffling gait, and intention tremors. What is the most likely diagnosis for this man?")
#print()
#pipe("A 45-year-old man with a history of alcohol use, who has been abstinent for the past 10 years, presents with sudden onset dysarthria, shuffling gait, and intention tremors. What is this man likely to be diagnosed with?")


#pipe("[INST] Which condition is not targeted for prevention by the Integrated Management of Childhood Illness (IMCI) initiative? Answer in a few sentences. [/INST]")
#print()
#pipe("[INST] Which organism and condition are associated with the production of asteroid bodies and cigar-shaped globi? Answer in a few sentences. [/INST]")
#print()
#pipe("[INST] A 57-year-old woman recently diagnosed with pharyngitis and prescribed phenoxymethylpenicillin presents with fever, skin rash, flank pain, and elevated serum creatinine levels. Her urinalysis shows increased white blood cells and eosinophils, but no bacterial growth. What is the best initial step in managing her condition? Answer in a few sentences. [/INST]")
#print()
#pipe("[INST] A 66-year-old man presents with a 3-month history of fatigue and examination reveals marked pallor, multiple pinpoint, red nonblanching spots on his extremities, and significant splenomegaly. Laboratory studies show severe anemia, leukocytosis with a white cell count of 80,000/mm3, thrombocytopenia, and a blood smear showing immature cells with large, prominent nucleoli and pink, elongated, needle-shaped cytoplasmic inclusions. What is the most likely diagnosis given these findings? Answer in a few sentences. [/INST]")

#pipe("[INST] After a 45-year-old woman who has undergone gastric surgery complains of severe nausea and vomiting on the second post-operative day, her surgeon prescribes metoclopramide in addition to ondansetron. Following the addition of metoclopramide, she experiences significant relief. What mechanism of action of metoclopramide is responsible for alleviating her symptoms? [/INST]")
"""
Metoclopramide alleviates post-operative nausea and vomiting through its dual mechanism of action. 
Firstly, it works centrally by blocking dopamine D2 receptors in the chemoreceptor trigger zone of the brain, 
which helps prevent the sensation of nausea and the act of vomiting. Secondly, it has a peripheral action as a prokinetic agent, 
enhancing gastrointestinal motility by promoting faster gastric emptying and improving gut peristalsis. 
This comprehensive action on both the central and gastrointestinal systems contributes to significant relief of nausea and vomiting in a post-operative setting.
"""

pipe("[INST] You are a medical professional assisting a patient with their questions. You should relay the information in a way that is easy for the patient to understand, and as if you are speaking to them. Keep it concise, but make sure you answer all questions. What should I do after being diagnosed with leukemia? What are the safest and most effective treatments? [/INST]")


