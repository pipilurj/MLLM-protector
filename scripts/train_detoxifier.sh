deepspeed --include=localhost:0,1 --master_port 30006 llava/train/correction_model_withquestion.py \
          --model_name=lmsys/vicuna-7b-v1.5 \
          --per_device_train_batch_size=4 \
          --learning_rate 2e-5 \
          --num_train_epochs 4 \
          --deepspeed scripts/zero2.json \
          --train_subset -1 \
          --weight_decay 0 \
          --lora_rank 128