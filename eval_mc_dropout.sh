model_name='MC-Dropout'
mode='mc'
mc_samples=15

python generate_predictions.py --MODE $mode --SAMPLES $mc_samples --MODEL_PATH "trained/$mode/$model_name.pth"
python evaluate.py --MODE $mode --SAMPLES $mc_samples --MODEL_NAME $model_name