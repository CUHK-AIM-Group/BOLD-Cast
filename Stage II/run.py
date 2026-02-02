import argparse
import random
import numpy as np
import torch
from task.long_term_forecasting import Exp_Long_Term_Forecast


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    parser = argparse.ArgumentParser(description='BOLD-Cast')

    # basic config
    parser.add_argument('--task_name', type=str,  default='long_term_forecast', help='task name')
    parser.add_argument('--is_training', type=int,  default=1, help='status')
    parser.add_argument('--model_id', type=str,  default='test', help='model id')
    parser.add_argument('--model', type=str, default='BOLDCast',
                        help='model name, BOLDCast')

    # data loader
    parser.add_argument('--data', type=str, default='Dataset', help='dataset type')
    parser.add_argument('--root_path', type=str, default='dataset/ukb_input/ts/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ukb.csv', help='data file')
    parser.add_argument('--test_data_path', type=str, default='ukb.csv', help='test data file used in zero shot forecasting')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--drop_last',  action='store_true', default=True, help='drop last batch in data loader')
    parser.add_argument('--val_set_shuffle', action='store_false', default=False, help='shuffle validation set')
    parser.add_argument('--drop_short', action='store_true', default=True, help='drop too short sequences in dataset')

    # forecasting task
    parser.add_argument('--his_len', type=int, default=81, help='label length')
    parser.add_argument('--pre_len', type=int, default=81, help='token length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--llm_ckp_dir', type=str, default='gpt2', help='llm checkpoints dir')
    parser.add_argument('--mlp_hidden_dim', type=int, default=128, help='mlp hidden dim')
    parser.add_argument('--mlp_hidden_layers', type=int, default=0, help='mlp hidden layers')
    parser.add_argument('--mlp_activation', type=str, default='tanh', help='mlp activation')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--batch_size_val', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing lr', default=False)
    parser.add_argument('--tmax', type=int, default=100, help='tmax in cosine anealing lr')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--mix_embeds', action='store_true', help='mix embeds', default=False)
    parser.add_argument('--test_dir', type=str, default='./test', help='test dir')
    parser.add_argument('--test_file_name', type=str, default='checkpoint.pth', help='test file')
    parser.add_argument('--dtype', type=str, default='float8', help='test file')

    # GPU
    parser.add_argument('--gpu_line', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_lLama', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--visualize', action='store_true', help='visualize', default=False)

    args = parser.parse_args()
    Exp = Exp_Long_Term_Forecast
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)      # 生成模型exp# set experiments
            setting = '{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.his_len + args.pre_len,
                args.his_len,
                args.pre_len,
                args.learning_rate,
                args.batch_size,
                args.weight_decay,
                args.mlp_hidden_dim,
                args.mlp_hidden_layers,
                args.cosine,
                args.mix_embeds,
                args.des, ii)
            if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:     #将参数打印出来
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)      # 开始训练
    else:
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.label_len,
            args.token_len,
            args.learning_rate,
            args.batch_size,
            args.weight_decay,
            args.mlp_hidden_dim,
            args.mlp_hidden_layers,
            args.cosine,
            args.mix_embeds,
            args.des, ii)
        exp = Exp(args)  # set experiments
        exp.test(setting)
        torch.cuda.empty_cache()
