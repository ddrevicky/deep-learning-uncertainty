model_name='Ensemble'
mode='ensemble'
ensemble_size=15

python generate_predictions.py --MODE $mode --MODEL_PATH "trained/$mode/$model_name"
python evaluate.py --MODE $mode --SAMPLES $ensemble_size --MODEL_NAME $model_name