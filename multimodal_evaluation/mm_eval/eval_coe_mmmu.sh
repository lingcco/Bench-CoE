python -m eval_coe_mmmu \
    --model1_path /mnt/data/zxj/checkpoints/TinyLLaVA-Phi-2-SigLIP-3.1B \
    --model2_path /mnt/data/zxj/checkpoints/Bunny-v1_1-4B \
    --bert_model_path /mnt/data/zxj/checkpoints/trained_model/bert_mmmu \
    --image-folder /mnt/data/zxj/data/MMMU/all_images \
    --question-file /mnt/data/zxj/data/MMMU/anns_for_eval.json \
    --answers-file /root/coe/zxj/mmmu_eval/answers/coe.jsonl \

python convert_answer_to_mmmu.py \
    --answers-file /root/coe/zxj/mmmu_eval/answers/coe.jsonl \
    --answers-output /root/coe/zxj/mmmu_eval/answers/coe.json

cd /mnt/data/zxj/data/MMMU/eval

python main_eval_only.py --output_path /root/coe/zxj/mmmu_eval/answers/coe.json