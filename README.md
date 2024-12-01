<h2 align="center"><a>Bench-CoE</a><h5 align="center">

## Test CoE Model:

Multimodal: test CoE model by lmms-eval

1. Install the environment based on lmms-eval and the selected model.

2. Train the router(Bert) using the MMMU dataset.

3. Place 'coemodel.py' in the '.lmms_eval/models' directory and add 'coemodel' to the 'AVAILABLE_MODELS' list in '__init__.py'

4. Modify the paths for the models and the router in the code.

5. Run the command and select the corresponding task to test coemodel:
   'CUDA_VISIBLE_DEVICES=0 python3 -m accelerate.commands.launch --num_processes=8 -m lmms_eval --model coemodel
    --model_args pretrained="None" --tasks name --batch_size 1 --log_samples --log_samples_suffix coemodel --output_path ./logs/'

   Note: As long as the selected model and task are supported by lmm-eval, you only need to make the necessary modifications in 'coemodel.py'.

Multimodal: test CoE model by yourself(case):

1. Install the environment based on TinyLLaVA_Factory and Bunny, prepare the corresponding dataset.

2. Train the router(Bert) using the MMMU dataset.

3. Run the command to test coemodel on the MMMU dataset:
   'CUDA_VISIBLE_DEVICES=0 bash path/to/eval_coe_mmmu.sh'

   Note: When using this method, if you need to add a model, make modifications according to the loading method of the corresponding model.