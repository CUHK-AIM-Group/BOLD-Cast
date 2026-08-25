from datetime import datetime, timedelta
import os
import torch
import numpy as np
import pandas as pd
import glob, sys

folder_path = 'dataset/UKB_roi/'
file_pattern = os.path.join(folder_path, "*_cc200.npy")
file_list = sorted(glob.glob(file_pattern))

if not file_list:
    print(f"No _cc200.npy file in {folder_path}")
    sys.exit()

current_time = datetime(2017, 7, 1, 0, 0, 0)
step = timedelta(milliseconds = 735)
processed_count = 0

cg_score = pd.read_csv('dataset/UKB1.csv')
cg_score = pd.DataFrame(cg_score)
age = np.zeros((len(file_list),1))
i = 0
for file in file_list:
    # timestamp
    temp = np.load(file)
    if len(temp.shape) != 2:
        print(f"warning: {temp.shape} of {os.path.basename(file)} is not correct.")
        num_timepoints = 0
    else:
        num_timepoints = temp.shape[1]

    if num_timepoints > 450:
        # timestamps = []
        # for i in range(490):
        #     formatted_time = current_time.strftime("%Y/%m/%d %H:%M:%S")
        #     timestamps.append(formatted_time)
        #     current_time += step

        # sex
        sample_index = cg_score[cg_score.subject_id.values == int(file[16:23])].index.to_list()
        cg_score_sample = cg_score.loc[sample_index]
        gt1 = cg_score_sample.sex.values
        gt2 = cg_score_sample.birth.values
        if gt1 == 1: # male
            sex = [1, 0]
        else: # female
            sex = [0, 1]
        age[i,:] = 2004 - gt2
        i = i + 1
        np.savetxt('dataset/age.csv', age, delimiter=',')
        # corr = torch.corrcoef(torch.tensor(temp[:,:490]))
        # timestamps = list(timestamps)
        # np.savez('dataset/ukb_input/ts/' + file[16:23] +'.npz', corr = np.asarray(corr),
        #          fMRI = temp[:,:490], text = timestamps, sex = sex)