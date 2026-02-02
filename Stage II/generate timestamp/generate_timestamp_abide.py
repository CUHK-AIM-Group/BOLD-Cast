from datetime import datetime, timedelta
import os
import torch
import numpy as np
import pandas as pd
import glob, sys

folder_path = 'dataset/ABIDE_roi/'
file_pattern = os.path.join(folder_path, "*_cc200.npy")
file_list = sorted(glob.glob(file_pattern))

if not file_list:
    print(f"No _cc200.npy file in {folder_path}")
    sys.exit()

current_time = datetime(2004, 1, 1, 0, 0, 0)
step = timedelta(milliseconds = 2000)
processed_count = 0

cg_score = pd.read_csv('dataset/ABIDE.csv')
cg_score = pd.DataFrame(cg_score)

for file in file_list:
    # timestamp
    temp = np.load(file)
    if len(temp.shape) != 2:
        print(f"warning: {temp.shape} of {os.path.basename(file)} is not correct.")
        num_timepoints = 0
    else:
        num_timepoints = temp.shape[1]

    if num_timepoints > 169:
        timestamps = []
        for i in range(170):
            formatted_time = current_time.strftime("%Y/%m/%d %H:%M:%S")
            timestamps.append(formatted_time)
            current_time += step

        # sex
        sample_index = cg_score[cg_score.subject_id.values == int(file[-19:-14])].index.to_list()
        cg_score_sample = cg_score.loc[sample_index]
        gt = cg_score_sample.SEX.values
        if gt == 1: # male
            sex = [1, 0]
        else: # female
            sex = [0, 1]

        # ASD
        gt = cg_score_sample.DX_GROUP.values
        if gt == 2: # healthy
            gt = [1, 0]
        else: # ASD
            gt = [0, 1]

        corr = torch.corrcoef(torch.tensor(temp[:,:170]))
        timestamps = list(timestamps)
        np.savez('dataset/abide_input/ts/' + file[-19:-14] +'.npz', corr = np.asarray(corr),
                 fMRI = temp[:,:170], text = timestamps, sex = sex, ASD = gt)
