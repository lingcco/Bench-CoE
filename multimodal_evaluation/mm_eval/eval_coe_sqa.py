import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import qianfan

from Bunny.bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from Bunny.bunny.conversation import conv_templates, SeparatorStyle
from Bunny.bunny.model.builder import load_bunny_pretrained_model
from Bunny.bunny.util.utils import disable_torch_init
from Bunny.bunny.util.mm_utils import tokenizer_image_token, get_model_name_from_path

from TinyLLaVA_Factory.tinyllava.utils import *
from TinyLLaVA_Factory.tinyllava.data import *
from TinyLLaVA_Factory.tinyllava.model import *

from PIL import Image
import math
import random


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_tinyllava_model(model_path):
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    return model, tokenizer, image_processor

def load_bunny_model(model_path, model_type):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_bunny_pretrained_model(model_path, None, model_name, model_type)
    return model, tokenizer, image_processor


def eval_model(args):
    # Model
    disable_torch_init()
    
    tinyllava_model, tinyllava_tokenizer, tinyllava_image_processor = load_tinyllava_model(args.model_path1)
    tinyllava_model.to(device='cuda')
    bunny3b_model, bunny3b_tokenizer, bunny3b_image_processor = load_bunny_model(args.model_path2, "phi-2")
    bunny3b_model.to(device='cuda')
    
    tinyllava_text_processor = TextPreprocess(tinyllava_tokenizer, "phi")
    tinyllava_data_args = tinyllava_model.config
    tinyllava_image_processor = ImagePreprocess(tinyllava_image_processor, tinyllava_data_args)
    
    os.environ["QIANFAN_ACCESS_KEY"] = "YOUR QIANFAN_ACCESS_KEY"  
    os.environ["QIANFAN_SECRET_KEY"] = "YOUR QIANFAN_SECRET_KEY"  
    chat_comp = qianfan.ChatCompletion()

    categories = ["natural science", "language science", "social science"]
    category_stats = {cat: {'correct': 0, 'total': 0} for cat in categories}
    
    #model_path = os.path.expanduser(args.model_path)
    #model_name = get_model_name_from_path(model_path)
    #tokenizer, model, image_processor, context_len = load_bunny_pretrained_model(model_path, args.model_base, model_name,"phi-3")

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    instruction = (
            "Please determine the category of this text. Respond with only one word: natural science, language science, or social science. "
            "No explanations or additional information."
        )
    
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs
        
        question_input = question['value'].split("\n(A)")[0]
        input_text = f"{question_input} {instruction}"
        response = chat_comp.do(model="ERNIE-Speed-128K", messages=[{
            "role": "user",
            "content": input_text  
        }])["body"]['result']
        #print("$$$$$response:", response)

        detected_category = None
        for category in categories:
            if category in response.lower():
                detected_category = category
                break
        #print("#####################detected_category:", detected_category)

        if detected_category is None:
            detected_category = random.choice(categories)
            
        if detected_category == "language science":
            if 'image' in line:
                image_file = line["image"]
                image = Image.open(os.path.join(args.image_folder, image_file))
                image_tensor = bunny3b_image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                images = image_tensor.unsqueeze(0).to(dtype=bunny3b_model.dtype, device='cuda', non_blocking=True)

                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                cur_prompt = '<image>' + '\n' + cur_prompt
            else:
                images = None

            if args.single_pred_prompt:
                qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
                cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, bunny3b_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

            with torch.inference_mode():
                output_ids = bunny3b_model.generate(
                    input_ids,
                    images=images,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=1024,
                    use_cache=True
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = bunny3b_tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
        
        else:
            if 'image' in line:
                image_file = line["image"]
                image = Image.open(os.path.join(args.image_folder, image_file))
                image_sizes = [image.size]
                image = tinyllava_image_processor(image)
                images = image.unsqueeze(0).half().cuda()
                cur_prompt = '<image>' + '\n' + cur_prompt
            else:
                images = None
                image_sizes = None

            if args.single_pred_prompt:
                cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."
            msg = Message()
            msg.add_message(cur_prompt)

            result = tinyllava_text_processor(msg.messages, mode='eval')
            input_ids = result['input_ids']
            prompt = result['prompt']
            input_ids = input_ids.unsqueeze(0).cuda()

            with torch.inference_mode():
                output_ids = tinyllava_model.generate(
                    input_ids,
                    images=images,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=1024,
                    use_cache=True,
                    pad_token_id = tinyllava_tokenizer.pad_token_id

                )
            outputs = tinyllava_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": "coe",
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path1", type=str, default=None)
    parser.add_argument("--model-path2", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--single-pred-prompt", action="store_true")

    args = parser.parse_args()

    eval_model(args)
