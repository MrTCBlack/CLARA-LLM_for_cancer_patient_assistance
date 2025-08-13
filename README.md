# CLARA-LLM_for_cancer_patient_assistance
## Authors: Tyler Black, Ashlyn King, Carly Beninati, John Luong

### Overview
Cancer patients often face challenges interpreting complex medical language and accessing clear, timely information about their diagnoses and treatments. 
To address this issue, CLARA (Cancer Language and Response Assistant), a domain-specific large language model (LLM), was developed to support cancer patients and 
caregivers through a user-friendly question-and-answer interface. 
CLARA was built on the Mistral-7B-Instruct model and was trained and fine-tuned on high-quality biomedical and oncology datasets using open-source tools, 
including multiple Hugging Face libraries. 
CLARA shows promising results on several text generation metrics, and reviewed well on internal user testing to assess clarity, trustworthiness, and emotional sensitivity. 
CLARA was developed to supplement — not replace — professional medical guidance, empowering patients by improving access to reliable and understandable cancer-related information. 
The final CLARA prototype demonstrates the efficacy of using LLMs to deliver personalized and easy-to-understand responses and
addresses limitations commonly seen in general-purpose LLMs, such as hallucinations and limited domain adaptation. 

### Specifications
* __Assumption__: You are using a Windows computer 
* Python 3.12.6
* You will need to create a Hugging Face account and agree to mistralai requirements to use the mistralai/Mistral-7B-Instruct-v0.3 at https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3

### Utilities
#### Model
mistralai/Mistral-7B-Instruct-v0.3 (https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
#### Datasets
lavita/MedQuAD (https://huggingface.co/datasets/lavita/MedQuAD)
#### Evaluation Metrics
- BLEU (https://huggingface.co/spaces/evaluate-metric/bleu)
- ROUGE (https://huggingface.co/spaces/evaluate-metric/rouge)
- METEOR (https://huggingface.co/spaces/evaluate-metric/meteor)
- BERTScore (https://huggingface.co/spaces/evaluate-metric/bertscore)
#### Quantization Method
bitsandbytes (https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)
#### Parameter-Efficient Fine-Tuning (PEFT) Method
Low-Rank Adaptation (LoRA) (https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)

### Files
#### CLARA_setup.ps1
This powershell script contains the setup and installs needed to properly run the CLARA model.
It should be run before running any of the other scripts to ensure that the proper python packages and libraries are installed.
#### run_CLARA.ps1
This powershell script can be run to boot up the CLARA model and open up a webpage that can be used to interact with CLARA as a chatbot.
#### MedQuAD.py
Python script used for training CLARA on data from the lavita/MedQuAD dataset.
#### Evaluation_Script.py
Python script used for evaluating the performance of CLARA and the model during training on a testing set from the lavita/MedQuAD dataset.
#### CLARAUI.py
Python script for running the CLARA model as an oncology focussed assistant.
Using Gradio to make the user-friendly question-and-answer interface, a user can easily interact with the CLARA model via a web browser.

### How to use
1. In a terminal, to run the ```CLARA_setup.ps1``` script to setup and install the appropriate libraries, run the following command:

    ```.\CLARA_setup.ps1```
  
2. In the terminal, to run the ```run_CLARA.ps1``` script to boot up CLARA and open a webpage to interact with CLARA, run the following command:

   ```.\run_CLARA.ps1```

