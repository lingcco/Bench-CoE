# name: CoE/eval_bbh_vllm_rule
# author：Mr.Zhao
# date：2024/10/24 
# time: 下午3:22 
# description:

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
import os
import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import pandas as pd


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载分类模型和tokenizer
cls_model_path = 'bert_base_mmlu_pro_10'
cls_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
cls_model = BertForSequenceClassification.from_pretrained(cls_model_path).to(device).eval()

max_model_length = 4096
max_new_tokens = 2048

# CoE
model_lib = ['../model/glm-4-9b',
             '../model/Qwen2-7B-Instruct',
             '../model/gemma-2-9b-it',
             '../model/Mathstral-7B-v0.1',
             '../model/Llama-3-Smaug-8B']

# model_lib = ['../model/glm-4-9b',
#              '../model/glm-4-9b',
#              '../model/glm-4-9b',
#              '../model/glm-4-9b',
#              '../model/glm-4-9b']

#
# model_lib = ['../model/Qwen2-7B-Instruct',
#              '../model/Qwen2-7B-Instruct',
#              '../model/Qwen2-7B-Instruct',
#              '../model/Qwen2-7B-Instruct',
#              '../model/Qwen2-7B-Instruct']

# model_lib = ['../model/gemma-2-9b-it',
#              '../model/gemma-2-9b-it',
#              '../model/gemma-2-9b-it',
#              '../model/gemma-2-9b-it',
#              '../model/gemma-2-9b-it']

# model_lib = ['../model/Mathstral-7B-v0.1',
#              '../model/Mathstral-7B-v0.1',
#              '../model/Mathstral-7B-v0.1',
#              '../model/Mathstral-7B-v0.1',
#              '../model/Mathstral-7B-v0.1']

# model_lib = ['../model/Llama-3-Smaug-8B',
#              '../model/Llama-3-Smaug-8B',
#              '../model/Llama-3-Smaug-8B',
#              '../model/Llama-3-Smaug-8B',
#              '../model/Llama-3-Smaug-8B']

# 数据预处理函数
def preprocess(data, task_name):
    processed_data = []
    prompt = load_prompt(task_name)
    #增加task分类
    for item in data:
        category_id, model_id = cls_question(item['input'])
        processed_data.append({'input': item['input'],
                               'target': item['target'],
                               'task':task_name,
                               'prompt': prompt,
                               'category_id': category_id,
                               'model_id': model_id})
    return processed_data

# 加载BBH数据集
def load_bbh_dataset(path='../data/BIG-Bench-Hard-main/bbh'):
    dataset = []
    # 路径下所有的文件
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            task_name = filename[:-5]
            file_path = os.path.join(path, filename)  # 设置BBH数据集的正确路径
            with open(file_path, 'r') as file:
                data = json.load(file)['examples']
                dataset.extend(preprocess(data, task_name))
    return dataset

# 加载prompt
def load_prompt(task_name):
    prompt_path = f"../data/BIG-Bench-Hard-main/cot-prompts/{task_name}.txt"  # 设置prompt文件的正确路径
    with open(prompt_path, 'r') as file:
        lines = file.readlines()

    # 假设警告信息总是位于第一行，并且以'-----'行作为实际prompt的开始
    start_index = next(i for i, line in enumerate(lines) if '-----' in line) + 1
    prompt = ''.join(lines[start_index:])  # 从'-----'后开始的行组成prompt
    return prompt



class MyModel:

    def __init__(self, model_lib):
        self.model_lib = model_lib
        self.tokenizers = {}
        self.sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens,
                                              stop=["Q:"], top_k=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载每个LLM对应的tokenizer
        for i, model_id in enumerate(model_lib):
            if model_id in self.tokenizers:
                continue
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.tokenizers[model_id] = tokenizer

    # 这里需要进行大量修改
    # def generate(self, test_data):
    #
    #     response_batch = []
    #     test_df = pd.DataFrame(test_data)
    #     grouped = test_df.groupby('model_id')
    #
    #     torch.cuda.empty_cache()
    #     torch.cuda.ipc_collect()
    #     gc.collect()
    #
    #     for model_id, group in grouped:
    #
    #         # print(model_id,flush=True)
    #         logging.info(f"Loading LLM model: {model_id}")
    #         llm = LLM(model=model_lib[model_id], gpu_memory_utilization=float(args.gpu_util),
    #                   tensor_parallel_size=1,
    #                   max_model_len=max_model_length,
    #                   trust_remote_code=True)
    #         logging.info(f"Memory used after loading model: {round(torch.cuda.memory_allocated() / 1024 / 1024 / 1024)} GB")
    #
    #         # 加载当前group的所有数据
    #         for index, sample in group.iterrows():
    #             # prompt_length_ok = False
    #             combined_input = f"{sample['prompt']}\n\nQ: {sample['input']}"
    #             # print(combined_input, flush=True)
    #             tokenizer = self.tokenizers[model_lib[model_id]]
    #             inputs = tokenizer(combined_input, return_tensors="pt").to(self.device)
    #             # 这里可能有问题
    #             # while not prompt_length_ok:
    #             #     inputs = tokenizer(combined_input, return_tensors="pt").to(self.device)
    #             #     length = len(inputs["input_ids"][0])
    #             #     if length < max_model_length - max_new_tokens:
    #             #         prompt_length_ok = True
    #
    #             with torch.no_grad():
    #                 outputs = llm.generate(inputs, self.sampling_params)
    #                 for output in outputs:
    #                     generated_text = output.outputs[0].text
    #                     print(generated_text, flush=True)
    #                     response_batch.append([generated_text, sample['target'], sample['task']])
    #
    #         del llm
    #         torch.cuda.empty_cache()
    #         torch.cuda.ipc_collect()
    #         gc.collect()
    #         logging.info(f"Unloaded LLM model: {model_id}")
    #         logging.info(f"Memory used after unloading model: {round(torch.cuda.memory_allocated() / 1024 / 1024 / 1024)} GB")
    #
    #     return response_batch


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

            # 处理每个分组中的样本
            # for index, sample in tqdm(group.iterrows()):
            #     prompt_length_ok = False
            #     combined_input = f"{sample['prompt']}\n\nQ: {sample['input']}"
            #
            #     # 检查提示长度是否合适
            #     while not prompt_length_ok:
            #         inputs = tokenizer(combined_input, return_tensors="pt").to(self.device)
            #         length = len(inputs["input_ids"][0])
            #         if length < max_model_length - max_new_tokens:
            #             prompt_length_ok = True
            #
            #     model_questions[model_id].append(combined_input)
            #     model_questions_ids[model_id].append(index)
            for index, sample in tqdm(group.iterrows()):
                combined_input = f"{sample['prompt']}\n\nQ: {sample['input']}"

                # 截断 combined_input 以防止死循环
                inputs = tokenizer(combined_input, return_tensors="pt").to(self.device)
                while len(inputs["input_ids"][0]) >= max_model_length - max_new_tokens:
                    combined_input = combined_input[:-50]  # 每次减少一些字符
                    inputs = tokenizer(combined_input, return_tensors="pt").to(self.device)

                model_questions[model_id].append(combined_input)
                model_questions_ids[model_id].append(index)

        temp_response_dict = {}
        print("prompts are generated.", flush=True)
        for model_id, prompts in model_questions.items():
            logging.info(f"Loading LLM model: {model_id}")
            llm = LLM(model=model_lib[model_id], gpu_memory_utilization=float(args.gpu_util),
                      tensor_parallel_size=1,  # 单GPU
                      max_model_len=max_model_length,
                      trust_remote_code=True)
            logging.info(f"Memory used after loading model: {round(torch.cuda.memory_allocated() / 1024 / 1024 / 1024)} GB")

            with torch.no_grad():
                # 批量生成答案
                outputs = llm.generate(prompts, self.sampling_params)
                for prompt_id, output in enumerate(outputs):
                    generated_text = output.outputs[0].text
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
            response_batch.append([temp_response_dict[i], test_df.loc[i, 'target'], test_df.loc[i, 'task'], test_df.loc[i, 'category_id']])

        return response_batch


# 为了减少模型加载，需要先按model_id对数据分组，然后再按照model_id逐个加载模型
        # for i, sample in enumerate(test_data):
        #     k = args.ntrain
        #     prompt_length_ok = False
        #     prompt = sample['prompt']
        #     input_question = sample['input']
        #     combined_input = f"{prompt}\n\nQ: {input_question}"
        #     model_id = sample['model_id']
        #     target = sample['target']
        #     task = sample['task']
        #     tokenizer = self.tokenizers[model_id]
        #     while not prompt_length_ok:
        #
        #         inputs = tokenizer(combined_input, return_tensors="pt").to(self.device)
        #         length = len(inputs["input_ids"][0])
        #         if length < max_model_length - max_new_tokens:
        #             prompt_length_ok = True
        #         k -= 1
        #
        #
        #
        #     logging.info(f"memory used2: {round(torch.cuda.memory_allocated() / 1024 / 1024 / 1024)} GB")
        #
        #     with torch.no_grad():
        #         outputs = llm.generate(inputs, self.sampling_params)
        #         for prompt_id, output in enumerate(outputs):
        #             generated_text = output.outputs[0].text
        #             response_batch.append([generated_text,target,task])
        #
        #     del llm
        #     torch.cuda.empty_cache()
        #     torch.cuda.ipc_collect()
        #     gc.collect()
        #     logging.info(f"Unloaded LLM model: {model_id}")
        #     logging.info(f"memory used3: {round(torch.cuda.memory_allocated() / 1024 / 1024 /1024)} GB")


def extract_answer(text):
    # pattern = r"answer is \(([^)]+)\)"
    pattern = r"answer is ([^\.,!?]+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_again(text)

def extract_again(text):
    pattern = r"answer is ([^\.,!?]+)"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None


def calculate_accuracy(response_batch):
    # 初始化任务分类统计数据
    task_accuracy = {}
    category_accuracy = {}
    total_correct = 0
    total_count = 0

    # 遍历每一个生成的响应
    for response in response_batch:
        generated_text, target, task, category = response
        # 提取生成文本中的答案
        extracted_answer = extract_answer(generated_text)
        # 判断提取的答案是否与目标一致
        is_correct = (extracted_answer == target)

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

    return task_accuracy,category_accuracy,total_accuracy

# 评估模型性能
# def evaluate_model(data):
#     correct = 0
#     for item in data:
#         # 在此处利用item进行编码
#
#         answer = generate_answer(item['input'])
#         if answer == item['target']:
#             correct += 1
#     accuracy = correct / len(data)
#     return accuracy

# 生成答案的函数
# def generate_answer(question):
#     # 这里可以添加使用prompt生成答案的代码
#     return "True"  # 假设模型的答案

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
    # subject_id_mapping = {
    #     0: 4,
    #     1: 0,
    #     2: 0,
    #     3: 0,
    #     4: 0,
    #     5: 0,
    #     6: 1,
    #     7: 0,
    #     8: 4,
    #     9: 0,
    #     10: 0,
    #     11: 0,
    #     12: 0,
    #     13: 0
    # }

    subject_id_mapping = {
        0: 4,
        1: 2,
        2: 2,
        3: 4,
        4: 2,
        5: 2,
        6: 1,
        7: 4,
        8: 1,
        9: 2,
        10: 3,
        11: 3,
        12: 2,
        13: 2
    }

    return question_cls_id, subject_id_mapping[question_cls_id]


# def args_generate_path(input_args):
#     scoring_method = "CoT"
#     model_name = input_args.model.split("/")[-1]
#     subjects = args.selected_subjects.replace(",", "-").replace(" ", "_")
#     return [model_name, scoring_method, subjects]

# 主函数
def main():

    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    log_file_path = f"./log/{time_str}_summary.txt"  # 例如，生成文件名如 "log_2024-10-28_14-30-00.txt"

    print(f"model loading start.",flush=True)
    model = MyModel(model_lib)
    # if not os.path.exists(save_result_dir):
    #     os.makedirs(save_result_dir)

    print(f"dataset loading start.", flush=True)
    test_data = load_bbh_dataset()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--global_record_file", "-grf", type=str,
                        default="eval_record_collection.csv")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--model", "-m", type=str, default="coe/coe_test")

    args = parser.parse_args()
    # os.makedirs(args.save_dir, exist_ok=True)
    # global_record_file = args.global_record_file
    # save_result_dir = os.path.join(
    #     args.save_dir, "/".join(args_generate_path(args))
    # )
    # file_prefix = "-".join(args_generate_path(args))
    # timestamp = time.time()
    # time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    # file_name = f"{file_prefix}_{time_str}_summary.txt"
    # summary_path = os.path.join(args.save_dir, "summary", file_name)
    # os.makedirs(os.path.join(args.save_dir, "summary"), exist_ok=True)
    # os.makedirs(save_result_dir, exist_ok=True)
    # save_log_dir = os.path.join(args.save_dir, "log")
    # os.makedirs(save_log_dir, exist_ok=True)
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
    #                     handlers=[logging.FileHandler(os.path.join(save_log_dir,
    #                                                                file_name.replace("_summary.txt",
    #                                                                                  "_logfile.log"))),
    #                               logging.StreamHandler(sys.stdout)])

    main()
