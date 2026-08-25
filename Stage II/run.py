import argparse
import random
import numpy as np
import torch
from task.long_term_forecasting import Long_Term_Forecast



if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    parser = argparse.ArgumentParser(description='BOLD-Cast')

    # basic config
    parser.add_argument('--task_name', type=str,  default='long_term_forecast', help='task name, options:[pretrain or finetune for SimMTM, long_term_forecast for others]')
    parser.add_argument('--is_training', type=bool,  default=True, help='status')
    parser.add_argument('--model', type=str,  default='BOLDCast', help='model name, options: [BOLDCast, BrainTransformer, DLinear, ForecastGrapher, FourierGNN, One_Fit_All, iTransformer, LightTS, MSGNet, PatchTST, SimMTM, TSMixer]')

    # data & model loader
    parser.add_argument('--data', type=str,  default="boldcast", help='options:boldcast for BOLDCast, others for others')
    parser.add_argument('--dataset', type=str,  default="ukb", help='options:ukb, hcpya,hcpd,hcpa,abide')
    parser.add_argument('--root_path', type=str, default='D:/jy/project/EaseScan/dataset/ukb_input/ts/', help='root path of the data file')
    parser.add_argument('--best_model_path', type=str, default='checkpoints/BOLDCast_ukb_best_model.pth', help='location of model checkpoints')
    parser.add_argument('--result_path', type=str, default='results/', help='location of model checkpoints')
    parser.add_argument('--llm_ckp_dir', type=str, default='gpt2', help='llm checkpoints dir')

    #Bold-Cast
    parser.add_argument('--mlp_hidden_dim', type=int, default=256, help='mlp hidden dim')
    parser.add_argument('--mlp_hidden_layers', type=int, default=0, help='mlp hidden layers')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=162, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=81, help='start token length')
    parser.add_argument('--pred_len', type=int, default=81, help='prediction sequence length')

    # SimMTM
    parser.add_argument('--temperature', type=float, default=0.2, help='temperature')
    parser.add_argument('--lm', type=int, default=3, help='average masking length')
    parser.add_argument('--positive_nums', type=int, default=3, help='masking series numbers')
    parser.add_argument('--rbtp', type=int, default=1,
                        help='0: rebuild the embedding of oral series; 1: rebuild oral series')
    parser.add_argument('--masked_rule', type=str, default='geometric',
                        help='geometric, random, masked tail, masked head')
    parser.add_argument('--mask_rate', type=float, default=0.5, help='mask ratio')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='s',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]')

    # Formers
    parser.add_argument('--use_norm', type=bool, default=True, help='use norm and denorm')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=190, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=190, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=190, help='output size')
    parser.add_argument('--d_model', type=int, default=768, help='dimension of model')   #  768 for GPT4TS 760 for else
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')  # 4 for
    parser.add_argument('--d_layers', type=int, default=4, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--embed_type', type=int, default=0, help='0: default, 1: value embedding + temporal embedding + positional embedding, 2: value embedding + temporal embedding, 3: value embedding + positional embedding, 4: value embedding')

    # ForecastGrapher
    parser.add_argument('--num_nodes', type=int, default=7, help='to create Graph')
    parser.add_argument('--subgraph_size', type=int, default=3, help='neighbors number')
    parser.add_argument('--tanhalpha', type=float, default=3, help='')
    parser.add_argument('--k', type=int, default=3, help='the number of GNN block')
    parser.add_argument('--z', type=int, default=32, help='scaler')
    parser.add_argument('--node_dim', type=int, default=10, help='node embbed to dim dimentions')
    parser.add_argument('--adj_path', type=str, default=None, help='for static adj')
    parser.add_argument('--gcn_depth', type=int, default=2, help='')
    parser.add_argument('--gcn_dropout', type=float, default=0.3, help='')
    parser.add_argument('--propalpha', type=float, default=0.3, help='')
    parser.add_argument('--conv_channel', type=int, default=32, help='')
    parser.add_argument('--skip_channel', type=int, default=32, help='')

    # Dlinear
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # PatchTST
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')

    # iTransformer
    parser.add_argument('--exp_name', type=bool, required=False, default='MTSF', help='experiemnt name, options:[MTSF, partial_train]')
    parser.add_argument('--channel_independence', type=bool, default=True, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)')
    parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

    # FourierGNN
    parser.add_argument('--feature_size', type=int, default='140', help='feature size')
    parser.add_argument('--embed_size', type=int, default=128, help='hidden dimensions')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden dimensions')
    parser.add_argument('--exponential_decay_step', type=int, default=5)
    parser.add_argument('--validate_freq', type=int, default=1)

    # GTP4TS
    parser.add_argument('--is_gpt', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--gpt_layers', type=int, default=3)
    parser.add_argument('--freeze', type=int, default=1)

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--batch_size_val', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing lr', default=False)
    parser.add_argument('--mix_embeds', action='store_true', help='mix embeds', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)



    Exp = Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)      # set experiments
            setting = '{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_{}_{}'.format(
                args.task_name,
                # args.model_id,
                args.model,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.learning_rate,
                args.batch_size,
                args.weight_decay,
                args.mlp_hidden_dim,
                args.mlp_hidden_layers,
                args.cosine,
                args.mix_embeds,
                args.des, ii)
            # if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            model = exp.train(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_{}_{}'.format(
            args.task_name,
            # args.model_id,
            args.model,
            args.seq_len,
            args.label_len,
            args.pred_len,
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

    exp.test(setting)
    torch.cuda.empty_cache()