from data_processor.data_factory import data_provider
from task.basic import Basic
import torch
import torch.nn as nn
from torch import optim
import os
import warnings
import numpy as np
import random
from models.TimeSeriesTransformer import TimeSeriesTransformer
warnings.filterwarnings('ignore')


class Long_Term_Forecast(Basic):
    def __init__(self, args):
        super(Long_Term_Forecast, self).__init__(args)
        
    def _build_model(self):
        # 如果是TimeSeriesTransformer，需要特殊处理
        if self.args.model == 'BrainTransformer':
            model = TimeSeriesTransformer(
                input_size=self.args.enc_in,
                dec_seq_len=self.args.label_len,  # 使用label_len作为解码器输入长度
                batch_first=True,
                out_seq_len=self.args.pred_len,
                max_seq_len=self.args.seq_len,
                dim_val=self.args.d_model,
                n_encoder_layers=self.args.e_layers,
                n_decoder_layers=self.args.d_layers,
                n_heads=self.args.n_heads,
                dropout_encoder=self.args.dropout,
                dropout_decoder=self.args.dropout,
                dropout_pos_enc=self.args.dropout,
                dim_feedforward_encoder=self.args.d_ff,
                dim_feedforward_decoder=self.args.d_ff,
                num_predicted_features=self.args.c_out
            )
        else:
            model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, f):
        data_set, data_loader = data_provider(self.args, flag, f)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print(n, p.dtype, p.shape)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def train(self, setting):
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.StepLR(model_optim, step_size=5, gamma = 0.5)
        vali_loss_best = 1e8

        for epoch in range(self.args.train_epochs):
            loss_train = torch.tensor(0., device="cuda")
            count = torch.tensor(0., device="cuda")
            self.model.train()
            filenames = os.listdir(self.args.root_path + 'train/')
            rate = 0.1
            picknumber1 = int(len(filenames) * rate)
            sample1 = random.sample(filenames, picknumber1)
            for f in sample1:
                train_data, train_loader = self._get_data(flag='train', f = f)
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                    if self.args.model == 'BOLDCast':
                        model_optim.zero_grad()
                        self.model = self.model.to(self.args.gpu)
                        batch_x = batch_x.float().to(self.args.gpu)
                        batch_y = batch_y.float().to(self.args.gpu)
                        batch_x_mark = batch_x_mark.float().to(self.args.gpu)
                        output = self.model(batch_x, batch_x_mark)

                    elif self.args.model == 'BrainTransformer':
                        model_optim.zero_grad()
                        self.model = self.model.to(self.args.gpu)
                        batch_x = batch_x.float().to(self.args.gpu)
                        batch_y = batch_y.float().to(self.args.gpu)
                        src = batch_x
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.gpu)
                        tgt = dec_inp[:, :self.args.label_len, :]
                        output = self.model(src=src, tgt=tgt)

                    else:
                        model_optim.zero_grad()
                        self.model = self.model.to(self.args.gpu)
                        batch_x = batch_x.float().to(self.args.gpu)
                        batch_y = batch_y.float().to(self.args.gpu)
                        batch_x_mark = batch_x_mark.float().to(self.args.gpu)
                        batch_y_mark = batch_y_mark.float().to(self.args.gpu)
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.gpu)
                        output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    output = output[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.gpu)
                    loss = torch.mean(torch.abs(output - batch_y))

                    count += 1
                    loss.backward()
                    model_optim.step()
                    loss_train += loss

            print('Epoch: {:.1f}'.format(epoch))
            print('====> train_loss = {:.4f}'.format(loss_train/len(sample1)))

            # val
            vali_loss_all = 0
            rate = 1.0
            filenames_v = os.listdir(self.args.root_path + 'val')
            picknumber2 = int(len(filenames_v) * rate)
            sample2 = random.sample(filenames_v, picknumber2)
            self.model.eval()
            with torch.no_grad():
                for f_v in sample2:
                    ind = 0
                    vali_data, vali_loader = self._get_data(flag='val', f = f_v)
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                        if self.args.model == 'BOLDCast':
                            output = self.model(batch_x.float().to(self.args.gpu), batch_x_mark.float().to(self.args.gpu))

                        elif self.args.model == 'BrainTransformer':
                            batch_x = batch_x.float().to(self.args.gpu)
                            batch_y = batch_y.float().to(self.args.gpu)
                            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.gpu)
                            output = self.model(src=batch_x, tgt=dec_inp[:, :self.args.label_len, :])

                        else:
                            batch_x = batch_x.float().to(self.args.gpu)
                            batch_y = batch_y.float().to(self.args.gpu)
                            batch_x_mark = batch_x_mark.float().to(self.args.gpu)
                            batch_y_mark = batch_y_mark.float().to(self.args.gpu)
                            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.gpu)
                            output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        output = output[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                        if ind == 0:
                            output_all = output
                            y_all = batch_y
                            ind = ind + 1
                        else:
                            output_all = torch.vstack((output_all, output))
                            y_all = torch.vstack((y_all, batch_y))

                    y_all = y_all.detach().cpu() * (vali_data.max.T - vali_data.min.T) + vali_data.min.T
                    output_all = output_all.detach().cpu() * (vali_data.max.T - vali_data.min.T) + vali_data.min.T
                    vali_loss = torch.mean(torch.abs(output_all - y_all))
                    vali_loss_all = vali_loss + vali_loss_all

                vali_loss_all = vali_loss_all / len(sample2)
                if vali_loss_all < vali_loss_best:
                    vali_loss_best = vali_loss_all
                    cnt_wait = 0
                    torch.save(self.model.state_dict(), 'checkpoints/'+ self.args.model +'_'+ self.args.dataset +'_best_model.pth')
                    print('====> val_loss_best = {:.4f}'.format(vali_loss_best))
                else:
                    cnt_wait += 1
                    scheduler.step()
                    print("lr = {:.8f}".format(model_optim.param_groups[0]['lr']))
                    print('====> val_loss_best = {:.4f}, vali_loss_all= {:.4f}'.format(vali_loss_best, vali_loss_all))
                    if cnt_wait == self.args.patience:
                        print("Early stopped!")
                        break
        return self.model


    def test(self, setting):
        best_model_path = self.args.best_model_path
        self.model.load_state_dict(torch.load(best_model_path), strict=False)
        test_loss_all = 0

        filenames_t = os.listdir(self.args.root_path + 'test')
        rate = 1.0
        picknumber2 = int(len(filenames_t) * rate)  # 按照rate比例从文件夹中取一定数量的文件
        sample2 = random.sample(filenames_t, picknumber2)  # 随机选取picknumber数量的样本
        xxx = 0
        metrics = np.zeros((len(sample2), 7))
        with torch.no_grad():
            for f_t in sample2:
                ind = 0
                test_data, test_loader = self._get_data(flag='test', f=f_t)
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    if self.args.model == 'BOLDCast':
                        output = self.model(batch_x.float().to(self.args.gpu), batch_x_mark.float().to(self.args.gpu))

                    elif self.args.model == 'BrainTransformer':
                        batch_x = batch_x.float().to(self.args.gpu)
                        batch_y = batch_y.float().to(self.args.gpu)
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.gpu)
                        output = self.model(src=batch_x, tgt=dec_inp[:, :self.args.label_len, :])

                    else:
                        batch_x = batch_x.float().to(self.args.gpu)
                        batch_y = batch_y.float().to(self.args.gpu)
                        batch_x_mark = batch_x_mark.float().to(self.args.gpu)
                        batch_y_mark = batch_y_mark.float().to(self.args.gpu)
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.args.gpu)
                        output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    output = output[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                    if ind == 0:
                        output_all = output
                        y_all = batch_y
                        ind = ind + 1
                    else:
                        output_all = torch.vstack((output_all, output))
                        y_all = torch.vstack((y_all, batch_y))

                y_all = y_all.detach().cpu() * (test_data.max.T - test_data.min.T) + test_data.min.T
                output_all = output_all.detach().cpu() * (test_data.max.T - test_data.min.T) + test_data.min.T

                sigma_p = (output_all).std(axis=0)
                sigma_g = (y_all).std(axis=0)
                mean_p = output_all.mean(axis=0)
                mean_g = y_all.mean(axis=0)
                index = (sigma_g != 0);
                correlation = ((output_all - mean_p) * (y_all - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
                correlation = (correlation[index]).mean();

                test_loss = torch.sqrt(torch.mean((output_all - y_all) ** 2))
                test_loss_all = test_loss + test_loss_all
                test_loss = test_loss / len(filenames_t)
                test_loss = test_loss
                metrics[xxx, 0] = torch.mean(torch.abs(output_all - y_all))
                metrics[xxx, 1] = torch.median(torch.abs(output_all - y_all))
                metrics[xxx, 2] = torch.sqrt(torch.mean((output_all - y_all) ** 2))
                metrics[xxx, 3] = 100 * torch.mean(torch.abs((output_all - y_all) / y_all))
                metrics[xxx, 4] = correlation
                fc1 = torch.corrcoef(output_all[0,:,:].T)
                fc2 = torch.corrcoef(y_all[0,:,:].T)
                metrics[xxx, 5] = torch.corrcoef(torch.stack([fc1.ravel(), fc2.ravel()]))[0,1]
                metrics[xxx, 6] = torch.linalg.norm(fc1 - fc2, ord='fro')

                print(
                    '====> test_loss = {:.4f}, mse = {:.4f}, medae = {:.4f}, rmse = {:.4f}, mape = {:.4f}%, mpcc = {:.4f}, fc_pcc = {:.4f}, FN = {:.4f}'.format(
                        test_loss, metrics[xxx, 0], metrics[xxx, 1], metrics[xxx, 2], metrics[xxx, 3], metrics[xxx, 4], metrics[xxx, 5], metrics[xxx, 6]))
                xxx = xxx + 1
                dir_path = self.args.result_path + self.args.model + '/' + self.args.dataset
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                np.savez(dir_path + '/' +'output_all_' + f_t[:-4] + '.npz', output_all=output_all, y_all=y_all)
            metrics = torch.asarray(metrics)
            np.savetxt(dir_path + '/metircs.csv', metrics, delimiter=',')
        return

