#!/bin/bash

python llava/eval/robustness_eval \
    --model-path liuhaotian/llava-v1.5-13b \
    --harm_detector path-to-harmdetector \
    --question-file path-to-question-file \
    --image-folder path-to-image-folder \
    --answers-file path-to-output-file \
    --temperature 0 \
    --conv-mode vicuna_v1
