import os
import numpy as np


def minmax_normalize_3d(arr):

    epsilon = 1e-8
    mean_vals = np.mean(arr, axis=1, keepdims=True)
    std_vals = np.std(arr, axis=1, keepdims=True)
    normalized_arr = (arr - mean_vals) / (std_vals +  epsilon)

    return normalized_arr

folder_path = 'dataset/ukb/train/'
files = os.listdir(folder_path)
x = np.zeros((len(files), 190, 100))
y = np.zeros((len(files), 190, 100))
adj = np.zeros((len(files), 190, 190))
i = 0
print(len(files))
for file in files:
    ts = np.load(folder_path + file)
    ts = minmax_normalize_3d(ts['fMRI'])
    x[i,:,:] = ts[:, :100]
    y[i,:,:] = ts[:, 100:200]
    adj[i,:,:] = np.corrcoef(ts)
    print("adj max:", np.max(adj[i,:,:]))
    i = i + 1

np.savez('data/ukb/train.npz', adj = adj, x = x, y = y)
