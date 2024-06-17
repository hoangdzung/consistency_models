CUDA_VISIBLE_DEVICES=4  python image_sample.py --batch_size 256 --training_mode consistency_distillation --sampler onestep --model_path ../cd_imagenet64_lpips.pt.1 --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples 10000 --resblock_updown True --use_fp16 True --weight_schedule uniform --save_dir stable_imn_64_10000 --load_from_file 

CUDA_VISIBLE_DEVICES=4 python image_sample_pick_prior.py --training_mode edm --batch_size 64 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path ../edm_imagenet64_ema.pt --attention_resolutions 32,16,8  --class_cond True --dropout 0.1 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples 50000 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --weight_schedule karras

CUDA_VISIBLE_DEVICES=4 python backup.py --training_mode edm --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path ../edm_imagenet64_ema.pt --attention_resolutions 32,16,8  --class_cond True --dropout 0.1 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples 2000 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --weight_schedule  karras --save_dir priors_10000_k15/seed_4 --seed 4 --batch_size 32 --num_neighbors 15 --angle 1.0

CUDA_VISIBLE_DEVICES=4 python backup.py --batch_size 256 --training_mode consistency_distillation --sampler onestep --model_path ../cd_imagenet64_lpips.pt.1 --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples 40000 --resblock_updown True --use_fp16 True --weight_schedule uniform --save_dir priors_std_200000_k1/seed_4 --seed 4 --num_neighbors 1 --angle 1.0

