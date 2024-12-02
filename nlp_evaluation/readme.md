# NLP Project Documentation

## Environment Setup
- `environment.yml`: Contains a list of Python dependencies and their versions, essential for setting up the development environment.

## Code
### Evaluation Scripts
The scripts in the `/coe evaluation` directory are designed to evaluate various aspects of trained models:
- `eval_bbh_vllm_query.py`: Evaluates the performance of query-level Bench-CoE on the Big Bench Hard dataset.
- `eval_bbh_vllm_subject.py`: Evaluates the performance of subject-level Bench-CoE on the Big Bench Hard dataset.
- `eval_hellaswag_vllm_query.py`: Evaluates the performance of query-level Bench-CoE on the hellaswag dataset.
- `eval_mmlu_pro_vllm_query.py`: Evaluates the performance of query-level Bench-CoE on the MMLU Pro dataset.
- `eval_mmlu_pro_vllm_subject.py`: Evaluates the performance of subject-level Bench-CoE on the MMLU Pro dataset.
- `eval_winogrand_vllm_query.py`: Evaluates the performance of query-level Bench-CoE on the Winogrande dataset.


## Instruction

### 1. Setting Up the Conda Environment
Begin by installing and setting up a Conda environment tailored for your project. This ensures that all dependencies are managed and isolated effectively.

### 2. Downloading Pre-trained BERT Router Models
Clone the following repositories to download the pre-trained weights for various BERT router models from Hugging Face:
```bash
git clone https://huggingface.co/LambdaZ27/subject_bert_mmlu_pro
git clone https://huggingface.co/LambdaZ27/query_bert_mmlu_pro
git clone https://huggingface.co/LambdaZ27/query_bert_hellaswag
git clone https://huggingface.co/LambdaZ27/query_bert_winogrande
```

### 3. Downloading and Setting Up Large Models as Sub-models
Download the desired large models to be used as sub-models to your local environment. Modify the paths in your code to point to these model files accordingly.

### 4. Downloading Datasets from Hugging Face
Obtain the relevant datasets from Hugging Face, ensuring compatibility with your model's training or testing requirements.

### 5. Running the Test Scripts
Execute the appropriate test files to evaluate the models and verify their performance.

