# !/bin/bash
model=$1
mkdir -p /home/qlianab/TianyangHan/code/dpo/res/llava1.5_13b-lora32-lr2e-6-shargpt4_6w_llava_6w_coco_6w-1e/pope/answers/
python -m llava.eval.model_vqa_loader \
    --model-path "$model" \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /home/qlianab/TianyangHan/data/coco/val2014 \
    --answers-file "/home/qlianab/TianyangHan/code/dpo/res/llava1.5_13b-lora32-lr2e-6-shargpt4_6w_llava_6w_coco_6w-1e/pope/answers/llava1.5_13b-lora32-lr2e-6-shargpt4_6w_llava_6w_coco_6w-1e.jsonl" \
    --temperature 0 \
    --conv-mode vicuna_v1



python llava/eval/eval_pope.py \
    --annotation-dir /home/qlianab/TianyangHan/workout/LLaVA/playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file "/home/qlianab/TianyangHan/code/dpo/res/llava1.5_13b-lora32-lr2e-6-shargpt4_6w_llava_6w_coco_6w-1e/pope/answers/llava1.5_13b-lora32-lr2e-6-shargpt4_6w_llava_6w_coco_6w-1e.jsonl"
#!/bin/bash

# python -m llava.eval.model_vqa_loader \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --image-folder /home/qlianab/TianyangHan/data/coco/val2014 \
#     --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python llava/eval/eval_pope.py \
#     --annotation-dir ./playground/data/eval/pope/coco \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --result-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl