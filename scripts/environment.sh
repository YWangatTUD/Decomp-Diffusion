conda create -n "lsd" python=3.11 -y && conda activate lsd && conda install pip -y && python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && python -m pip install -U diffusers transformers tensorboard matplotlib einops accelerate xformers scikit-learn scipy distinctipy