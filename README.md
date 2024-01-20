1. create the conda environment
   
   conda create -n "lsd" python=3.11 -y && conda activate lsd && conda install pip -y && python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && python -m pip install -U diffusers transformers tensorboard matplotlib einops accelerate xformers scikit-learn scipy distinctipy

2. train the model
   
   CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes=2 --main_process_port 29500 train_lsd.py \
--enable_xformers_memory_efficient_attention --dataloader_num_workers 4 --learning_rate 2e-5 --mixed_precision fp16 \
--num_validation_images 32 --val_batch_size 32 --max_train_steps 500000 --checkpointing_steps 25000 --checkpoints_total_limit 2 \
--gradient_accumulation_steps 1 --seed 42 --encoder_lr_scale 1.0 --train_split_portion 0.9 \
--output_dir ~/Projects/latent-decomposed-diffusion/lsd/celebahq/ --backbone_config configs/celebahq/backbone/config.json \
--latent_encoder_config configs/celebahq/latent_encoder/config.json --unet_config configs/celebahq/unet/config.json \
--scheduler_config configs/celebahq/scheduler/scheduler_config.json --dataset_root /space/ywang86/celebahq_data128x128/ \
--dataset_glob '**/*.jpg' --train_batch_size 32 --resolution 128 --validation_steps 5000 --tracker_project_name latent_decomposed_diffusion
