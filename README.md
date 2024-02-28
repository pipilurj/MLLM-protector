![ex1](src/llava_protector.png)
# MLLM-Protector: Ensuring MLLM’s Safety without Hurting Performance

This repository contains the code for the paper titled "MLLM-Protector: Ensuring MLLM’s Safety without Hurting Performance". [[Link to our paper](https://arxiv.org/abs/2401.02906)]

## Install Packages

```

conda create -n mllm_protector python=3.10 -y

conda activate mllm_protector

pip install -e .

```

## Train Harm Detector

```
bash scripts/train_harm_detector.sh
```

## Train Detoxifier

```
bash scripts/train_detoxifier.sh
```

## Inference
```
bash scripts/v1_5/eval/robust_eval.sh
```

## Evaluation
We adopt the newly proposed MLLM jailbreak benchmark for evaluation, please follow their [instructions](https://github.com/isXinLiu/MM-SafetyBench) for setting up the evaluation bench. Thanks for the great work!
## Acknowledgement
The project is built on top of the amazing multimodal large language model [LLaVA](https://github.com/haotian-liu/LLaVA). 
Thanks for these great work!


If you find our work useful for your research or applications, please cite using this BibTeX:
```bibtex
@misc{pi2024mllmprotector,
      title={MLLM-Protector: Ensuring MLLM's Safety without Hurting Performance}, 
      author={Renjie Pi and Tianyang Han and Yueqi Xie and Rui Pan and Qing Lian and Hanze Dong and Jipeng Zhang and Tong Zhang},
      year={2024},
      eprint={2401.02906},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
