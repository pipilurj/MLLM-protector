![ex1](src/llava_protector.png)
# MLLM-Protector: Ensuring MLLM’s Safety without Hurting Performance

This repository contains the code and data for the paper titled "MLLM-Protector: Ensuring MLLM’s Safety without Hurting Performance".

[Paper](coming soon), [Dataset](coming soon), [Model Parameters](coming soon)

[comment]: <> (## Install Packages)

[comment]: <> (```)

[comment]: <> (conda create -n gllava python=3.10 -y)

[comment]: <> (conda activate gllava)

[comment]: <> (pip install -e .)

[comment]: <> (```)

[comment]: <> (## Data Preparation)

[comment]: <> ([comment]: <> &#40;Download the COCO dataset from [huggingface] &#40;To be published&#41;.&#41;)

[comment]: <> (Download our dataset &#40;To be published&#41;.)

[comment]: <> (Place the data under playground/data.)

[comment]: <> (Here is the data structure:)

[comment]: <> (```)

[comment]: <> (playground/data/)

[comment]: <> (├── geo3k/)

[comment]: <> (├── geoqa_plus/)

[comment]: <> (├── test/)

[comment]: <> (├── alignment.json)

[comment]: <> (├── qa_tuning.json)

[comment]: <> (├── test_question.jsonl)

[comment]: <> (├── test_answers.jsonl)

[comment]: <> (```)

[comment]: <> (## First Stage Alignment)

[comment]: <> (This stage enables the model to better interpret the content of geometric figures.)

[comment]: <> (```)

[comment]: <> (bash scripts/run_alignment.sh)

[comment]: <> (```)

[comment]: <> (## Second Stage Alignment)

[comment]: <> (This stage equips the model with stronger ability for solving geometry problems.)

[comment]: <> (```)

[comment]: <> (bash scripts/run_qa.sh)

[comment]: <> (```)

[comment]: <> (## Evaluation)

[comment]: <> (Generate responses from the model.)

[comment]: <> (```)

[comment]: <> (bash scripts/v1_5/eval_multi.sh /)

[comment]: <> (                path-to-model /)

[comment]: <> (                playground/data/test_questions.jsonl /)

[comment]: <> (                path-to-output /)

[comment]: <> (                ../data /)

[comment]: <> (                num_gpus /)

[comment]: <> (                temperature)

[comment]: <> (```)

[comment]: <> (Run automatic evaluation to calculate the accuracy.)

[comment]: <> (```)

[comment]: <> (python scripts/geo_acc_calculate.py /)

[comment]: <> (             --ground_truth_file playground/data/test_answers.jsonl /)

[comment]: <> (             --predictions_file path-to-output)

[comment]: <> (```)

[comment]: <> (This github repo will be updated soon, stay tuned!)

[comment]: <> (## Acknowledgement)

[comment]: <> (The project is built on top of the amazing [LLaVA]&#40;https://github.com/haotian-liu/LLaVA&#41; repository. Thanks for their great work!)


[comment]: <> (If you find our code and dataset helpful to your research, please consider citing us with this BibTeX:)

[comment]: <> (```bibtex)

[comment]: <> (@misc{gao2023gllava,)

[comment]: <> (      title={G-LLaVA: Solving Geometric Problem with Multi-Modal Large Language Model}, )

[comment]: <> (      author={Jiahui Gao and Renjie Pi and Jipeng Zhang and Jiacheng Ye and Wanjun Zhong and Yufei Wang and Lanqing Hong and Jianhua Han and Hang Xu and Zhenguo Li and Lingpeng Kong},)

[comment]: <> (      year={2023},)

[comment]: <> (      eprint={2312.11370},)

[comment]: <> (      archivePrefix={arXiv},)

[comment]: <> (      primaryClass={cs.CL})

[comment]: <> (})

[comment]: <> (```)