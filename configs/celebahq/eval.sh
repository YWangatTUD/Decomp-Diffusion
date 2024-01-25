CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 eval.py \
--mixed_precision fp16 \
--seed 42 \
--batch_size 32 --num_validation_images 32 \
--output_dir ~/Projects/latent-slot-diffusion-decomp/lsd/celebahq/gen_images \
--scheduler_config configs/celebahq/scheduler/scheduler_config.json --dataset_root /space/ywang86/celebahq_data128x128/ \
--dataset_glob '**/*.jpg' --resolution 128 --vit_input_resolution 128 \
--ckpt_path ~/Projects/latent-decomposed-diffusion/checkpoint-175000
