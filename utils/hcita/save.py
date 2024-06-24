import os
from pathlib import Path

import numpy as np
import torch

def save_top_N_model( epoch, ckpt, store_path, filename, num_top, min_score, score_key, score_dict):
    """
    ckpt        : define by yolov5
    num_top     : number of models will be stored
    min_score   : minimum score. if score less then min_score, ckpt will not be stored.
    score_key   : name = 'p', 'r', 'map50', 'map5095', 'avg'
    score_dict  : dictionary of name and score. key = p, r, map50, map50-95, avg
    """
    
    if score_key != 'avg' and score_dict[score_key] < min_score:
        return

    if score_key == 'p':
        store_path = store_path / "precision"
    elif score_key == 'r':
        store_path = store_path / "recall"
    elif score_key == 'avg':
        store_path = store_path / "avg"
    # elif score_key == 'map50':
    #     store_path = store_path / "amp50"
    
    Path(store_path).mkdir(parents=True, exist_ok=True)
    
    sub_name = ""
    for key, val in score_dict.items():
        sub_name += f"{key}-{score_dict[key]:.4}_"
    sub_name =f"{epoch}_{sub_name[:-1]}"
    archive_path = store_path / f"{filename}_{sub_name}.pt"
    torch.save(ckpt, str(archive_path) )
    

    file_path_list = [os.path.join(store_path, f) for f in os.listdir(store_path) if os.path.isfile(os.path.join(store_path, f)) and f.endswith('.pt')]
    if len(file_path_list) > num_top:
        file_scores = []
        if score_key == 'avg':
            file_pscores = []
            file_rscores = []


        for f in file_path_list:
            if os.name == 'posix':
                params = os.path.splitext(f)[0].split('/')[-1].split('_')
            elif os.name == 'nt':
                params = os.path.splitext(f)[0].split('\\')[-1].split('_')

            if score_key == 'avg':
                pscore = [s for s in params if 'p' in s]
                pscore = pscore[0].split('-')[-1]
                rscore = [s for s in params if 'r' in s]
                rscore = rscore[0].split('-')[-1]
                try:
                    pscore = float(pscore)
                except ValueError as e:
                    print(f"There is no precision score in file{f}: \n {e}")
                try:
                    rscore = float(rscore)
                except ValueError as e:
                    print(f"There is no recall score in file{f}: \n {e}")

                file_pscores.append(pscore)
                file_rscores.append(rscore)
            else:
                score = [s for s in params if score_key in s]
                score = score[0].split('-')[-1]
                try:
                    score = float(score)
                except ValueError as e:
                        print(f"{f} does not have correct file name format: \n {e}")
                file_scores.append(score)

        if score_key == 'avg':
            file_scores = [(p+float(r))*0.5 for p,r in zip(file_pscores, file_rscores)]
        score_index = np.argsort(file_scores)[::-1]
        rm_idx = score_index[num_top:]
        for idx in rm_idx:
            try:
                os.remove(file_path_list[idx])
            except OSError as e:
                print(e)