
import pandas as pd
import numpy as np
import glob
import os 
import sys 
try:
    task = sys.argv[1]
except IndexError:
    print("Must specify a task as the first commandline argument, e.g.:")
    print('    `python list_metrics.py.py enhancer_annotation`')
    exit()
"""
Get the average and standard deviation of the MCC scores for all CV-folds of each model in the enhancer annotation task
"""
print(f'Listing metrics for {task.upper()}'.replace('_', ' '))
par_dir = os.path.dirname(os.path.dirname(__file__))

folder = f'{par_dir}/downstream_tasks/{task}/'
for model in os.listdir(folder):
    dfs = glob.glob(f'{folder}/{model}/**/best_model_metrics.csv', recursive = True)
    metric = []
    if len (dfs) == 0:
        continue
    for df in dfs:
        
        df = pd.read_csv(df, header ='infer')
        test_col = [n for n in df.columns if n.startswith('test')][1]
        test_metric = test_col.split('_')[-1].upper()
        metric.append(df[test_col].values[0])

    with open(f'{folder}/{model}/summed_metrics.txt', 'w') as f:
        f.write(f'Model: {model:<25} | N runs: {len(metric):>2} | {test_metric}  Mean: {np.mean(metric):.4f} | Std. dev: {np.std(metric):.4f}')
    print(f'Model: {model:<25} | N runs: {len(metric):>2} | {test_metric}  Mean: {np.mean(metric):.4f} | Std. dev: {np.std(metric):.4f}')