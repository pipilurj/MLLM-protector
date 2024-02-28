import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model_adapt
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model import *
import transformers
from PIL import Image
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import readline
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)

    # model = LlavaLlamaForCausalLMAdapt.from_pretrained(
    #     args.model_path,
    # )
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     args.model_path,
    #     model_max_length=2048,
    #     padding_side="right",
    #     use_fast=False,
    # )
    # mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    # mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    # if mm_use_im_patch_token:
    #     tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    # if mm_use_im_start_end:
    #     tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    # model.resize_token_embeddings(len(tokenizer))
    # model.get_model().load_mm_projector_adaptor()
    # vision_tower = model.get_vision_tower()
    # if not vision_tower.is_loaded:
    #     vision_tower.load_model()
    # vision_tower.to(device="cuda", dtype=torch.float16)
    # image_processor = vision_tower.image_processor
    tokenizer, model, image_processor, context_len = load_pretrained_model_adapt(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
    model.get_model().load_mm_projector_adaptor()
    readline.parse_and_bind("")

    while True:
        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        try:
            img_name = input("Enter image path (or 'q' to quit): ")
            if img_name.lower() == 'q':
                break
            inp = input(f"{roles[0]}: ")
        except EOFError:
            continue
        try:
            image = load_image(img_name)
            # Similar operation in model_worker.py
            image_tensor = process_images([image], image_processor, args)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                # image_tensor = [image.to(model.device, dtype=torch.bfloat16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        except:
            image = image_tensor = None
        # image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)
        print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=args.do_sample,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs
        print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()
    main(args)
