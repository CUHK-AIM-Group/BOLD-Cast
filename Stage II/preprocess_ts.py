import argparse
import torch
from tqdm import tqdm
from models.Preprocess import Model
import os
from data_processor.data_loader_hcp import Dataset_Preprocess
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BOLD-Cast Preprocess')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--llm_ckp_dir', type=str, default='gpt2', help='llm checkpoints dir')
    parser.add_argument('--dataset', type=str, default='ukb_task',
                        help='dataset to preprocess, options:[UKB, HCP-A, HCP-D, HCP-YA, ABIDE]')
    args = parser.parse_args()
    print(args.dataset)

    label_len = 81
    pred_len = 81
    seq_len = label_len + pred_len

    model = Model(args)
    path = 'dataset/ukb_input/ts/test/'
    filenames = os.listdir(path)
    for f in filenames:
        data_set = Dataset_Preprocess(root_path=path, data_path=f, size=[seq_len, label_len, pred_len])
        data_loader = DataLoader(
            data_set,
            batch_size=32,
            shuffle=False,
        )

        save_dir_path = 'dataset/abide_input/sp/test/'
        output_list = []
        for idx, data in tqdm(enumerate(data_loader)):
            output = model(data)
            output_list.append(output.detach().cpu())
        result = torch.cat(output_list, dim=0)
        torch.save(result, save_dir_path + f'/{f}.pt')
