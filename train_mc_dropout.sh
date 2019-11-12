model_name='MC-Dropout'
mode='mc'
mc_samples=15

python train.py --MODEL_PATH trained/$mode --MODEL_NAME $model_name --DOWN_DROP '0.4,0.4,0.4,0.4' --UP_DROP '0.4,0.4,0.4,0.4'