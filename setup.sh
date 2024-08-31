sudo apt update
sudo apt install python3-pip -y
sudo apt install python3.10-venv -y
python3 -m venv venv
source venv/bin/activate
pip install llama-toolchain
pip install setuptools
llama download --source meta --model-id Meta-Llama3.1-8B-Instruct
pip install torch fairscale fire blobfile
python -c "import torch; print([(i, torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())]);"