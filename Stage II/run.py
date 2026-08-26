import argparse
import random
from pathlib import Path

import numpy as np
import torch

from task.long_term_forecasting import Long_Term_Forecast


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE2_ROOT = Path(__file__).resolve().parent

DATASETS = ('UKB', 'HCP-YA', 'HCP-D', 'HCP-A', 'ABIDE')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if v.lower() in ('0', 'false', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value.')


def build_parser():
    parser = argparse.ArgumentParser(description='BOLD-Cast Stage II')

    # Basic configuration
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=str2bool, default=True)
    parser.add_argument('--model', type=str, default='BOLDCast')
    parser.add_argument('--dataset', type=str, choices=DATASETS, default='UKB')

    # Unified dataset / output paths. Stage II always reads the outer dataset/.
    parser.add_argument('--dataset_root', type=str, default=str(PROJECT_ROOT / 'dataset'))
    parser.add_argument('--best_model_path', type=str, default=None)
    parser.add_argument('--result_path', type=str, default=str(STAGE2_ROOT / 'results'))
    parser.add_argument('--llm_ckp_dir', type=str, default=str(STAGE2_ROOT / 'gpt2'))

    # Forecast dimensions. These are explicit Stage-II hyperparameters and are
    # never inferred from dataset name, time_len, or atlas configuration.
    parser.add_argument('--seq_len', type=int, default=162)
    parser.add_argument('--label_len', type=int, default=81)
    parser.add_argument('--pred_len', type=int, default=81)

    # BOLDCast
    parser.add_argument('--mlp_hidden_dim', type=int, default=256)
    parser.add_argument('--mlp_hidden_layers', type=int, default=0)
    parser.add_argument('--mlp_activation', type=str, default='gelu')
    parser.add_argument('--mix_embeds',action='store_true',default=False)

    # Generic forecasting/model parameters used by the included baselines
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--freq', type=str, default='s')
    parser.add_argument('--use_norm', type=str2bool, default=True)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--enc_in', type=int, default=190)
    parser.add_argument('--dec_in', type=int, default=190)
    parser.add_argument('--c_out', type=int, default=190)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=4)
    parser.add_argument('--d_layers', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true')

    # ForecastGrapher / graph models
    parser.add_argument('--subgraph_size', type=int, default=3)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--z', type=int, default=32)
    parser.add_argument('--node_dim', type=int, default=10)
    parser.add_argument('--adj_path', type=str, default=None)
    parser.add_argument('--gcn_depth', type=int, default=2)
    parser.add_argument('--propalpha', type=float, default=0.3)
    parser.add_argument('--conv_channel', type=int, default=32)
    parser.add_argument('--skip_channel', type=int, default=32)
    parser.add_argument('--individual', action='store_true', default=False)

    # PatchTST / related models
    parser.add_argument('--head_dropout', type=float, default=0.0)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--class_strategy', type=str, default='projection')

    # FourierGNN
    parser.add_argument('--feature_size', type=int, default=140)
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)

    # GPT4TS
    parser.add_argument('--is_gpt', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--gpt_layers', type=int, default=3)
    parser.add_argument('--freeze', type=int, default=1)

    # SimMTM parameters that are actually referenced by the repository
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--positive_nums', type=int, default=3)

    # Optimization
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--des', type=str, default='Exp')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lradj', type=str, default='type1')

    # GPU
    parser.add_argument('--use_gpu', type=str2bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--local_rank', type=int, default=0)

    # Used only by classification branches in several baseline models.
    parser.add_argument('--num_class', type=int, default=2)

    return parser


def resolve_args(args):
    # Stage II owns its forecasting dimensions directly. Dataset selection only
    # determines where data are read from; it does not overwrite seq/label/pred
    # lengths or channel dimensions supplied through argparse.
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    input_root = dataset_root / f'{args.dataset}_input'
    args.root_path = str(input_root / 'ts')
    args.sp_path = str(input_root / 'sp')

    if args.best_model_path is None:
        args.best_model_path = str(STAGE2_ROOT / 'checkpoints' / f'{args.model}_{args.dataset}_best_model.pth')
    else:
        args.best_model_path = str(Path(args.best_model_path).expanduser().resolve())

    args.result_path = str(Path(args.result_path).expanduser().resolve())
    args.llm_ckp_dir = str(Path(args.llm_ckp_dir).expanduser().resolve())

    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        args.device_ids = [int(device_id) for device_id in args.devices.split(',')]
        args.gpu = args.device_ids[0]
    else:
        args.device_ids = [args.gpu]

    return args


def make_setting(args, ii):
    return '{}_{}_{}_sl{}_ll{}_pl{}_lr{}_bt{}_wd{}_hd{}_hl{}_mix{}_{}_{}'.format(
        args.task_name,
        args.model,
        args.dataset,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.learning_rate,
        args.batch_size,
        args.weight_decay,
        args.mlp_hidden_dim,
        args.mlp_hidden_layers,
        args.mix_embeds,
        args.des,
        ii,
    )


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args = resolve_args(build_parser().parse_args())

    print('Args in experiment:')
    print(args)
    print(f'ts input : {args.root_path}')
    print(f'sp input : {args.sp_path}')
    print(f'checkpoint: {args.best_model_path}')
    print(f'mix_embeds: {args.mix_embeds}')

    Path(args.best_model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result_path).mkdir(parents=True, exist_ok=True)

    Exp = Long_Term_Forecast
    setting = None

    if args.is_training:
        for ii in range(args.itr):
            setting = make_setting(args, ii)
            exp = Exp(args)
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)
            if args.use_gpu:
                torch.cuda.empty_cache()
    else:
        ii = 0
        setting = make_setting(args, ii)
        exp = Exp(args)

    exp.test(setting)
    if args.use_gpu:
        torch.cuda.empty_cache()
