model_name='Ensemble'
mode='ensemble'
ensemble_size=15

for I in $(seq 1 $ensemble_size)
do
    python train.py --MODEL_PATH trained/$mode/$model_name --MODEL_NAME $I --DOWN_DROP '0,0,0,0' --UP_DROP '0,0,0,0'
done