
import pandas as pd
import numpy as np
import glob
import os 


"""
Get the average and standard deviation of the MCC scores for all CV-folds of each model in the enhancer annotation task
"""
folder = '../downstream_tasks/enhancer_annotation/'
for model in os.listdir(folder):
    dfs = glob.glob(f'{folder}/{model}/*/best_model_metrics.csv', recursive = True)
    mcc = []
    if len (dfs) == 0:
        continue
    for df in dfs:
        
        df = pd.read_csv(df, header ='infer')
        mcc.append(df.test_mcc.values[0])

    with open(f'{folder}/{model}/summed_metrics.txt', 'w') as f:
        f.write(f'Model : {model:<25}  Runs : {len(mcc)}, Mean : {np.mean(mcc):.4f} Std dev : {np.std(mcc):.4f}')
    print(f'Model : {model:<25}  Runs : {len(mcc)}, Mean : {np.mean(mcc):.4f} Std dev : {np.std(mcc):.4f}')