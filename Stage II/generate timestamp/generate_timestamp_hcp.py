from datetime import datetime, timedelta
import os
import torch
import numpy as np
import pandas as pd
import glob, sys

folder_path = 'dataset/HCP1200_roi/'
file_pattern = os.path.join(folder_path, "*_cc200.npy")
file_list = sorted(glob.glob(file_pattern))

if not file_list:
    print(f"No _cc200.npy file in {folder_path}")
    sys.exit()

current_time = datetime(2002, 4, 1, 0, 0, 0)
step = timedelta(milliseconds = 700)
processed_count = 0

cg_score = pd.read_csv('dataset/HCP.csv')
cg_score = pd.DataFrame(cg_score)

for file in file_list:
    # timestamp
    temp = np.load(file)
    if len(temp.shape) != 2:
        print(f"warning: {temp.shape} of {os.path.basename(file)} is not correct.")
        num_timepoints = 0
    else:
        num_timepoints = temp.shape[1]

    if num_timepoints > 0:
        timestamps = []
        for i in range(num_timepoints):
            formatted_time = current_time.strftime("%Y/%m/%d %H:%M:%S")
            timestamps.append(formatted_time)
            current_time += step

    # sex
    sample_index = cg_score[cg_score.Subject.values == int(file[20:26])].index.to_list()
    cg_score_sample = cg_score.loc[sample_index]
    gt = cg_score_sample.Gender.values
    if gt == 'M':
        sex = [1, 0]
    else:
        sex = [0, 1]
    # cog
    ReadEng = cg_score_sample.ReadEng_Unadj.values
    ProcSpeed = cg_score_sample.ProcSpeed_Unadj.values
    Flanker = cg_score_sample.Flanker_Unadj.values
    PicSeq = cg_score_sample.PicSeq_Unadj.values
    SCPT = cg_score_sample.SCPT_SEN.values

    corr = torch.corrcoef(torch.tensor(temp))
    timestamps = list(timestamps)
    np.savez('dataset/hcp_input/ts/' + file[20:26] +'.npz', corr = np.asarray(corr),
             fMRI = temp, text = timestamps, sex = sex, ReadEng=ReadEng, ProcSpeed= ProcSpeed,
             Flanker=Flanker, PicSeq=PicSeq,SCPT=SCPT)
