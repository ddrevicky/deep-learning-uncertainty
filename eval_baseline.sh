model_name='Ensemble'
mode='ensemble'
ensemble_size=1

python evaluate.py --MODE $mode --SAMPLES $ensemble_size --MODEL_NAME $model_name