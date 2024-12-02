# name: CoE/eval_hellaswag_vllm_rule
# author：Mr.Zhao
# date：2024/10/27 
# time: 下午11:13 
# description:
# name: CoE/eval_bbh_vllm_rule
# author：Mr.Zhao
# date：2024/10/24
# time: 下午3:22
# description:

import os
import csv
import json
import argparse
import gc
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from dataclasses import dataclass
import logging
import sys
import argparse

import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import pandas as pd
from difflib import SequenceMatcher


# 设置设备

# device = None
#
# # 加载分类模型和tokenizer
# cls_model_path = None
# cls_tokenizer = None
# cls_model = None
#
# max_model_length = None
# max_new_tokens = None

model_pool = {

    'glm': ['../coe/model/glm-4-9b',
            '../coe/model/glm-4-9b',
            '../coe/model/glm-4-9b',
            '../coe/model/glm-4-9b',
            '../coe/model/glm-4-9b'],
    'qwen': ['../coe/model/Qwen2-7B-Instruct',
             '../coe/model/Qwen2-7B-Instruct',
             '../coe/model/Qwen2-7B-Instruct',
             '../coe/model/Qwen2-7B-Instruct',
             '../coe/model/Qwen2-7B-Instruct'],
    'gemma': ['../coe/model/gemma-2-9b-it',
              '../coe/model/gemma-2-9b-it',
              '../coe/model/gemma-2-9b-it',
              '../coe/model/gemma-2-9b-it',
              '../coe/model/gemma-2-9b-it'],
    'math': ['../coe/model/Mathstral-7B-v0.1',
             '../coe/model/Mathstral-7B-v0.1',
             '../coe/model/Mathstral-7B-v0.1',
             '../coe/model/Mathstral-7B-v0.1',
             '../coe/model/Mathstral-7B-v0.1'],
    'llama': ['../coe/model/Llama-3-Smaug-8B',
              '../coe/model/Llama-3-Smaug-8B',
              '../coe/model/Llama-3-Smaug-8B',
              '../coe/model/Llama-3-Smaug-8B',
              '../coe/model/Llama-3-Smaug-8B'],
    'coe': ['../coe/model/glm-4-9b',
            '../coe/model/Qwen2-7B-Instruct',
            '../coe/model/gemma-2-9b-it',
            '../coe/model/Mathstral-7B-v0.1',
            '../coe/model/Llama-3-Smaug-8B']

}

# CoE
# model_lib = ['../coe/model/glm-4-9b',
#              '../coe/model/Qwen2-7B-Instruct',
#              '../coe/model/gemma-2-9b-it',
#              '../coe/model/Mathstral-7B-v0.1',
#              '../coe/model/Llama-3-Smaug-8B']

# model_lib = ['../coe/model/glm-4-9b',
#              '../coe/model/glm-4-9b',
#              '../coe/model/glm-4-9b',
#              '../coe/model/glm-4-9b',
#              '../coe/model/glm-4-9b']

# model_lib = ['../coe/model/Qwen2-7B-Instruct',
#              '../coe/model/Qwen2-7B-Instruct',
#              '../coe/model/Qwen2-7B-Instruct',
#              '../coe/model/Qwen2-7B-Instruct',
#              '../coe/model/Qwen2-7B-Instruct']

# model_lib = ['../coe/model/gemma-2-9b-it',
#              '../coe/model/gemma-2-9b-it',
#              '../coe/model/gemma-2-9b-it',
#              '../coe/model/gemma-2-9b-it',
#              '../coe/model/gemma-2-9b-it']


# model_lib = ['../coe/model/Mathstral-7B-v0.1',
#              '../coe/model/Mathstral-7B-v0.1',
#              '../coe/model/Mathstral-7B-v0.1',
#              '../coe/model/Mathstral-7B-v0.1',
#              '../coe/model/Mathstral-7B-v0.1']

# model_lib = ['../coe/model/Llama-3-Smaug-8B',
#              '../coe/model/Llama-3-Smaug-8B',
#              '../coe/model/Llama-3-Smaug-8B',
#              '../coe/model/Llama-3-Smaug-8B',
#              '../coe/model/Llama-3-Smaug-8B']

# 加载prompt
# def load_prompt(task_name):
#     prompt_path = f"../data/BIG-Bench-Hard-main/cot-prompts/{task_name}.txt"  # 设置prompt文件的正确路径
#     with open(prompt_path, 'r') as file:
#         lines = file.readlines()
#
#     # 假设警告信息总是位于第一行，并且以'-----'行作为实际prompt的开始
#     start_index = next(i for i, line in enumerate(lines) if '-----' in line) + 1
#     prompt = ''.join(lines[start_index:])  # 从'-----'后开始的行组成prompt
#     return prompt
# def load_prompt(task_name):
#     prompts = {
#         'validation': "Based on the context provided, choose the most plausible ending for the activity. The answer is a number between 0 and 3. The answer is: ",
#         'general': "Given the situation, select the most likely event that will follow. Your answer must be between 0 and 3. The answer is: "
#     }
#     # 返回对应任务的 prompt，并在末尾明确答案格式和范围
#     return prompts.get(task_name, prompts['general'])
def load_prompt():
    prompts = "Based on the context provided below, choose the index of the most plausible ending for the activity."
    # 返回对应任务的 prompt，并在末尾明确答案格式和范围
    return prompts


# 数据预处理函数
def preprocess(item):
    processed_data = []
    prompt = load_prompt()
    category_id, model_id = cls_question(item['ctx'])
    # 提取 `ctx` 和 `endings`，使用 `label` 作为目标
    processed_data.append({
        'input': item['ctx'],  # 使用 `ctx` 作为输入上下文
        'options': item['endings'],
        'target': item['label'],  # 使用 `label` 作为目标
        'task': item['activity_label'],  # 添加任务名称
        'prompt': prompt,  # 添加 prompt
        'category_id': category_id,
        'model_id': model_id  # 模型分类 ID
    })
    return processed_data


def load_hellaswag_dataset(path='../coe/data/hellaswag/validation/validation.parquet'):
    dataset = []
    data = pd.read_parquet(path)
    for index, row in tqdm(data.iterrows()):
        example = preprocess(row)
        dataset.extend(example)
    return dataset


class MyModel:

    def __init__(self, model_lib):
        self.model_lib = model_lib
        self.tokenizers = {}
        self.sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens,
                                              stop=["Q:"], top_k=1)
        self.device = device
        # 加载每个LLM对应的tokenizer
        for i, model_id in enumerate(model_lib):
            if model_id in self.tokenizers:
                continue
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.tokenizers[model_id] = tokenizer

    def generate(self, test_data):
        response_batch = []
        test_df = pd.DataFrame(test_data)
        grouped = test_df.groupby('model_id')

        model_questions = {}
        model_questions_ids = {}

        for model_id, group in grouped:
            logging.info(f"Loading LLM model: {model_id}")

            # 初始化模型相关存储
            if model_id not in model_questions:
                model_questions[model_id] = []
                model_questions_ids[model_id] = []

            tokenizer = self.tokenizers[model_lib[model_id]]

            for index, sample in tqdm(group.iterrows()):
                # combined_input = f"{sample['prompt']}\nActivity: {sample['task']}\nContext: {sample['input']}\nOptions: {sample['options'][0]}"
                combined_input = f"{sample['prompt']}\nActivity: {sample['task']}\nContext: {sample['input']}\nOptions: \n1. {sample['options'][0]}\n2. {sample['options'][1]}\n3. {sample['options'][2]}\n4. {sample['options'][3]}\nAnswer: "

                # print(combined_input)
                # 截断 combined_input 以防止死循环
                inputs = tokenizer(combined_input, return_tensors="pt").to(self.device)
                while len(inputs["input_ids"][0]) >= max_model_length - max_new_tokens:
                    combined_input = combined_input[:-50]  # 每次减少一些字符
                    inputs = tokenizer(combined_input, return_tensors="pt").to(self.device)

                model_questions[model_id].append(combined_input)
                model_questions_ids[model_id].append(index)

        temp_response_dict = {}
        for model_id, prompts in model_questions.items():
            logging.info(f"Loading LLM model: {model_id}")
            llm = LLM(model=model_lib[model_id], gpu_memory_utilization=float(args.gpu_util),
                      tensor_parallel_size=1,  # 单GPU
                      max_model_len=max_model_length,
                      trust_remote_code=True)
            logging.info(
                f"Memory used after loading model: {round(torch.cuda.memory_allocated() / 1024 / 1024 / 1024)} GB")

            with torch.no_grad():
                # 批量生成答案
                outputs = llm.generate(prompts, self.sampling_params)
                for prompt_id, output in enumerate(outputs):
                    generated_text = output.outputs[0].text
                    # print(generated_text, flush=True)

                    original_index = model_questions_ids[model_id][prompt_id]  # 获取对应的原始索引
                    temp_response_dict[original_index] = generated_text

            del llm
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            logging.info(f"Unloaded LLM model: {model_id}")
            logging.info(
                f"Memory used after unloading model: {round(torch.cuda.memory_allocated() / 1024 / 1024 / 1024)} GB")

        # 按原始顺序返回结果
        for i in range(len(test_df)):
            response_batch.append(
                [temp_response_dict[i], test_df.loc[i, 'options'], test_df.loc[i, 'target'], test_df.loc[i, 'task'],
                 test_df.loc[i, 'category_id']])

        return response_batch


def extract_answer(text):
    # pattern = r"answer is \(([^)]+)\)"
    pattern = r"answer is: ([^\.,!?]+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_again(text)


def extract_again(text):
    pattern = r"answer is: ([^\.,!?]+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None


# def calculate_accuracy(response_batch, threshold=0.8):
#     # 初始化任务分类统计数据
#     task_accuracy = {}
#     category_accuracy = {}
#     total_correct = 0
#     total_count = 0
#
#     # 遍历每一个生成的响应
#     for response in response_batch:
#         generated_text, options, target, task, category = response
#         print(generated_text, flush=True)
#         extracted_answer = generated_text.strip()
#         correct_option = options[int(target)]
#
#         # 检查 generated_text 是否复述了多个选项
#         multiple_options_mentioned = sum([opt in extracted_answer for opt in options]) > 1
#
#         # 如果复述了多个选项，则判为错误
#         if multiple_options_mentioned:
#             is_correct = False
#         else:
#             # 只考虑与目标选项的相似度
#             is_correct = (
#                     extracted_answer == correct_option or
#                     correct_option in extracted_answer
#             )
#
#         # 更新任务分类的统计数据
#         if task not in task_accuracy:
#             task_accuracy[task] = {'correct': 0, 'total': 0}
#         if category not in category_accuracy:
#             category_accuracy[category] = {'correct': 0, 'total': 0}
#
#         task_accuracy[task]['total'] += 1
#         category_accuracy[category]['total'] += 1
#
#         if is_correct:
#             task_accuracy[task]['correct'] += 1
#             category_accuracy[category]['correct'] += 1
#
#         # 更新总体统计数据
#         total_count += 1
#         if is_correct:
#             total_correct += 1
#
#     # 计算每个任务的准确率和总体准确率
#     for task, stats in task_accuracy.items():
#         task_accuracy[task]['accuracy'] = stats['correct'] / stats['total']
#
#     for category, stats in category_accuracy.items():
#         category_accuracy[category]['accuracy'] = stats['correct'] / stats['total']
#
#     total_accuracy = total_correct / total_count
#
#     return task_accuracy, category_accuracy, total_accuracy


# def calculate_accuracy(response_batch):
#     # 初始化任务分类统计数据
#     task_accuracy = {}
#     category_accuracy = {}
#     total_correct = 0
#     total_count = 0
#
#     # 遍历每一个生成的响应
#     for response in response_batch:
#         generated_text, options, target, task, category = response
#         print(generated_text, flush=True)
#         # 提取生成文本中的答案
#         extracted_answer = generated_text
#         # 判断提取的答案是否与目标一致
#         is_correct = (extracted_answer == options[int(target)] or options[int(target)] in extracted_answer)
#
#         # 更新任务分类的统计数据
#         if task not in task_accuracy:
#             task_accuracy[task] = {'correct': 0, 'total': 0}
#         if category not in category_accuracy:
#             category_accuracy[category] = {'correct': 0, 'total': 0}
#
#         task_accuracy[task]['total'] += 1
#         category_accuracy[category]['total'] += 1
#
#         if is_correct:
#             task_accuracy[task]['correct'] += 1
#             category_accuracy[category]['correct'] += 1
#         # 更新总体统计数据
#         total_count += 1
#         if is_correct:
#             total_correct += 1
#
#     # 计算每个任务的准确率和总体准确率
#     for task, stats in task_accuracy.items():
#         task_accuracy[task]['accuracy'] = stats['correct'] / stats['total']
#
#     for category, stats in category_accuracy.items():
#         category_accuracy[category]['accuracy'] = stats['correct'] / stats['total']
#
#     total_accuracy = total_correct / total_count
#
#     return task_accuracy,category_accuracy,total_accuracy
def calculate_accuracy(response_batch):
    # 初始化任务分类统计数据
    task_accuracy = {}
    category_accuracy = {}
    total_correct = 0
    total_count = 0

    # 遍历每一个生成的响应
    for response in response_batch:
        generated_text, options, target, task, category = response
        # print(generated_text, flush=True)
        # 提取生成文本中的答案
        extracted_answer = generated_text
        if extracted_answer!='':
            # 判断提取的答案是否与目标一致
            if '.' in extracted_answer:
                extracted_answer = extracted_answer.split('.')[0]
            if '\n' in extracted_answer:
                extracted_answer = extracted_answer.split('\n')[0]
            if ',' in extracted_answer:
                extracted_answer = extracted_answer.split(',')[0]
            if len(extracted_answer) > 1:
                extracted_answer = extracted_answer[0]
            if extracted_answer == '':
                is_correct = False
            else:
                is_correct = int(extracted_answer) == (int(target) + 1)
        else:
            is_correct = False
        # 更新任务分类的统计数据
        if task not in task_accuracy:
            task_accuracy[task] = {'correct': 0, 'total': 0}
        if category not in category_accuracy:
            category_accuracy[category] = {'correct': 0, 'total': 0}

        task_accuracy[task]['total'] += 1
        category_accuracy[category]['total'] += 1

        if is_correct:
            task_accuracy[task]['correct'] += 1
            category_accuracy[category]['correct'] += 1
        # 更新总体统计数据
        total_count += 1
        if is_correct:
            total_correct += 1

    # 计算每个任务的准确率和总体准确率
    for task, stats in task_accuracy.items():
        task_accuracy[task]['accuracy'] = stats['correct'] / stats['total']

    for category, stats in category_accuracy.items():
        category_accuracy[category]['accuracy'] = stats['correct'] / stats['total']

    total_accuracy = total_correct / total_count

    return task_accuracy, category_accuracy, total_accuracy


def cls_question(question):
    inputs = cls_tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = cls_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    question_cls_id = logits.argmax().item()
    # 在这里将学科转为编号
    # subject_mapping = {'biology': 0,
    #                    'business': 1,
    #                    'chemistry': 2,
    #                    'computer science': 3,
    #                    'economics': 4,
    #                    'engineering': 5,
    #                    'health': 6,
    #                    'history': 7,
    #                    'law': 8,
    #                    'math': 9,
    #                    'philosophy': 10,
    #                    'physics': 11,
    #                    'psychology': 12,
    #                    'other': 13
    #                    }
    # 在此处增加规则映射
    # subject_id_mapping={
    #     0 : 2,
    #     1 : 2,
    #     2 : 2,
    #     3 : 2,
    #     4 : 0,
    #     5 : 3,
    #     6 : 2,
    #     7 : 0,
    #     8 : 2,
    #     9 : 1,
    #     10 : 2,
    #     11 : 2,
    #     12 : 2,
    #     13 : 4
    # }
    subject_id_mapping={
        0 : 2,
        1 : 2,
        2 : 2,
        3 : 2,
        4 : 0,
        5 : 3,
        6 : 2,
        7 : 0,
        8 : 2,
        9 : 1,
        10 : 2,
        11 : 2,
        12 : 2,
        13 : 4
    }
    # subject_id_mapping = {
    #     0: 4,
    #     1: 4,
    #     2: 4,
    #     3: 4,
    #     4: 4,
    #     5: 4,
    #     6: 4,
    #     7: 1,
    #     8: 4,
    #     9: 4,
    #     10: 4,
    #     11: 4,
    #     12: 4,
    #     13: 1
    # }
    return question_cls_id, subject_id_mapping[question_cls_id]


# 主函数
def main():
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    log_file_path = f"./log/hellaswag_{time_str}_summary.txt"  # 例如，生成文件名如 "log_2024-10-28_14-30-00.txt"

    print(f"model loading start.", flush=True)
    model = MyModel(model_lib)
    # if not os.path.exists(save_result_dir):
    #     os.makedirs(save_result_dir)

    print(f"dataset loading start.", flush=True)
    test_data = load_hellaswag_dataset()
    print(f"dataset loading finish.", flush=True)
    # print(test_data, flush=True)

    response = model.generate(test_data)

    task_accuracy, category_accuracy, total_accuracy = calculate_accuracy(response)

    print("-------------------------------------------------")
    print(task_accuracy, flush=True)
    print("-------------------------------------------------")
    print(category_accuracy, flush=True)
    print("-------------------------------------------------")
    print("average_accuracy: ", total_accuracy, flush=True)

    with open(log_file_path, "w") as log_file:
        log_file.write("-------------------------------------------------\n")
        log_file.write(f"Task Accuracy: {task_accuracy}\n")
        log_file.write("-------------------------------------------------\n")
        log_file.write(f"Category Accuracy: {category_accuracy}\n")
        log_file.write("-------------------------------------------------\n")
        log_file.write(f"Average Accuracy: {total_accuracy}\n")
        log_file.write("-------------------------------------------------\n")


if __name__ == "__main__":
    global device, cls_model_path, cls_tokenizer, cls_model, max_model_length, max_new_tokens, model_lib

    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--global_record_file", "-grf", type=str,
                        default="eval_record_collection.csv")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--model", "-m", type=str, default="coe")
    parser.add_argument("--gpu_id", "-g", type=int, default=0)

    args = parser.parse_args()

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    # os.environ['VLLM_TARGET_DEVICE'] = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    # 加载分类模型和tokenizer
    cls_model_path = 'bert_base_mmlu_pro_10'
    cls_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    cls_model = BertForSequenceClassification.from_pretrained(cls_model_path).to(device).eval()

    max_model_length = 4096
    max_new_tokens = 2048

    model_lib = model_pool[args.model]

    main()
