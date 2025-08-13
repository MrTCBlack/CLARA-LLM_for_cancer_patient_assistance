# Run this script for the installs needed to run CLARA
# Python version is 3.12.6

# Create a python virtual environment
python -m venv .env

# Activate the virtual environment
.env/Scripts/Activate.ps1

# Get latest version of pip
python.exe -m pip install --upgrade pip

# install torch requirements
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126

# install pip requirements
pip install --upgrade datasets transformers evaluate peft bitsandbytes numpy sentencepiece protobuf hf_xet

# requirements for evaluation
pip install --upgrade rouge_score bert_score

# requirements for ui
pip install --upgrade gradio