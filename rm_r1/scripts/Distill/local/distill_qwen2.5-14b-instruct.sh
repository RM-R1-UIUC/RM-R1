
device=0,1,2,3 # GPUs to be trained on, minimum requirement: 1 gpu

deepspeed --include localhost:$device --module openrlhf.cli.train_sft \
   --save_path your_save_path \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 128 \
   --micro_train_batch_size 1 \
   --pretrain Qwen/Qwen2.5-14B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 12288 \
   --zero_stage 3 \
   --learning_rate 5e-6 \
   --dataset gaotang/RM-R1-Distill-SFT \
   --apply_chat_template \
   --input_key context_messages \
   --output_key winner \
   --flash_attn \
   --gradient_checkpointing \
   --packing_samples \
   --adam_offload \
   # --use_wandb your_wandb \ # Optinal 
   # --wandb_project RM-R1 \  # Optinal 
   # --wandb_run_name Qwen2.5-7b-instruct-distilled \ # Optinal 