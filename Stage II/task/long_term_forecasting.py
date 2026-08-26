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
        # Resolve one canonical device once. Do not move the model lazily inside
        # the first training batch: small datasets may legitimately have zero
        # batches if sampling/drop_last is misconfigured.
        self.device = torch.device(
            f"cuda:{self.args.gpu}" if self.args.use_gpu else "cpu"
        )
        self.model = self.model.to(self.device)

        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.StepLR(model_optim, step_size=5, gamma=0.5)
        vali_loss_best = float('inf')
        cnt_wait = 0

        for epoch in range(self.args.train_epochs):
            self.model.train()
            loss_train = 0.0
            train_steps = 0

            # Use every subject assigned to the train split. The previous hard-
            # coded rate=0.1 produced zero subjects when the split contained <10
            # files (e.g. int(9 * 0.1) == 0).
            filenames = sorted(
                f for f in os.listdir(os.path.join(self.args.root_path, 'train'))
                if f.endswith('.npz')
            )
            if not filenames:
                raise RuntimeError(
                    f"No training NPZ files found in {os.path.join(self.args.root_path, 'train')}"
                )

            for f in filenames:
                train_data, train_loader = self._get_data(flag='train', f=f)
                for _, batch in enumerate(train_loader):
                    model_optim.zero_grad()

                    if self.args.model == 'BOLDCast':
                        batch_x, batch_y, batch_x_mark, batch_y_mark, latent_embed = batch
                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float().to(self.device)
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        latent_embed = latent_embed.float().to(self.device)
                        output = self.model(batch_x, batch_x_mark, latent_embed)

                    elif self.args.model == 'BrainTransformer':
                        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float().to(self.device)
                        src = batch_x
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat(
                            [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                        ).float().to(self.device)
                        tgt = dec_inp[:, :self.args.label_len, :]
                        output = self.model(src=src, tgt=tgt)

                    else:
                        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float().to(self.device)
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat(
                            [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                        ).float().to(self.device)
                        output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    output = output[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = torch.mean(torch.abs(output - batch_y))

                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite training loss for subject {f}. "
                            "Check fMRI normalization and model inputs."
                        )

                    loss.backward()
                    model_optim.step()
                    loss_train += loss.item()
                    train_steps += 1

            if train_steps == 0:
                raise RuntimeError(
                    "No training batches were produced. Set DataLoader drop_last=False "
                    "or reduce batch_size for short subject sequences."
                )

            print('Epoch: {:.1f}'.format(epoch))
            print('====> train_loss = {:.4f}'.format(loss_train / train_steps))

            # Validation: all subjects, no parameter updates.
            vali_loss_all = 0.0
            vali_subjects = 0
            filenames_v = sorted(
                f for f in os.listdir(os.path.join(self.args.root_path, 'val'))
                if f.endswith('.npz')
            )
            if not filenames_v:
                raise RuntimeError(
                    f"No validation NPZ files found in {os.path.join(self.args.root_path, 'val')}"
                )

            self.model.eval()
            with torch.no_grad():
                for f_v in filenames_v:
                    output_chunks = []
                    y_chunks = []
                    vali_data, vali_loader = self._get_data(flag='val', f=f_v)

                    for _, batch in enumerate(vali_loader):
                        if self.args.model == 'BOLDCast':
                            batch_x, batch_y, batch_x_mark, batch_y_mark, latent_embed = batch
                            batch_x = batch_x.float().to(self.device)
                            batch_y = batch_y.float().to(self.device)
                            batch_x_mark = batch_x_mark.float().to(self.device)
                            latent_embed = latent_embed.float().to(self.device)
                            output = self.model(batch_x, batch_x_mark, latent_embed)

                        elif self.args.model == 'BrainTransformer':
                            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                            batch_x = batch_x.float().to(self.device)
                            batch_y = batch_y.float().to(self.device)
                            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                            dec_inp = torch.cat(
                                [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                            ).float().to(self.device)
                            output = self.model(
                                src=batch_x,
                                tgt=dec_inp[:, :self.args.label_len, :],
                            )

                        else:
                            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                            batch_x = batch_x.float().to(self.device)
                            batch_y = batch_y.float().to(self.device)
                            batch_x_mark = batch_x_mark.float().to(self.device)
                            batch_y_mark = batch_y_mark.float().to(self.device)
                            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                            dec_inp = torch.cat(
                                [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                            ).float().to(self.device)
                            output = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        output = output[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        output_chunks.append(output.detach().cpu())
                        y_chunks.append(batch_y.detach().cpu())

                    if not output_chunks:
                        raise RuntimeError(
                            f"No validation batches were produced for subject {f_v}."
                        )

                    output_all = torch.cat(output_chunks, dim=0)
                    y_all = torch.cat(y_chunks, dim=0)
                    y_all = y_all * (vali_data.max.T - vali_data.min.T) + vali_data.min.T
                    output_all = output_all * (vali_data.max.T - vali_data.min.T) + vali_data.min.T
                    vali_loss = torch.mean(torch.abs(output_all - y_all))

                    if not torch.isfinite(vali_loss):
                        raise FloatingPointError(
                            f"Non-finite validation loss for subject {f_v}."
                        )

                    vali_loss_all += vali_loss.item()
                    vali_subjects += 1

            vali_loss_all /= vali_subjects
            if vali_loss_all < vali_loss_best:
                vali_loss_best = vali_loss_all
                cnt_wait = 0
                os.makedirs(os.path.dirname(self.args.best_model_path), exist_ok=True)
                torch.save(self.model.state_dict(), self.args.best_model_path)
                print('====> val_loss_best = {:.4f}'.format(vali_loss_best))
            else:
                cnt_wait += 1
                scheduler.step()
                print("lr = {:.8f}".format(model_optim.param_groups[0]['lr']))
                print(
                    '====> val_loss_best = {:.4f}, vali_loss_all= {:.4f}'.format(
                        vali_loss_best, vali_loss_all
                    )
                )
                if cnt_wait == self.args.patience:
                    print("Early stopped!")
                    break

        return self.model


    def test(self, setting):
        if not hasattr(self, 'device'):
            self.device = torch.device(
                f"cuda:{self.args.gpu}" if self.args.use_gpu else "cpu"
            )
        self.model = self.model.to(self.device)
        best_model_path = self.args.best_model_path
        self.model.load_state_dict(
            torch.load(best_model_path, map_location=self.device), strict=False
        )
        test_loss_all = 0

        filenames_t = os.listdir(os.path.join(self.args.root_path, 'test'))
        rate = 1.0
        picknumber2 = int(len(filenames_t) * rate)  # 按照rate比例从文件夹中取一定数量的文件
        sample2 = random.sample(filenames_t, picknumber2)  # 随机选取picknumber数量的样本
        xxx = 0
        metrics = np.zeros((len(sample2), 7))
        with torch.no_grad():
            for f_t in sample2:
                ind = 0
                test_data, test_loader = self._get_data(flag='test', f=f_t)
                for i, batch in enumerate(test_loader):
                    if self.args.model == 'BOLDCast':
                        batch_x, batch_y, batch_x_mark, batch_y_mark, latent_embed = batch
                        output = self.model(
                            batch_x.float().to(self.device),
                            batch_x_mark.float().to(self.device),
                            latent_embed.float().to(self.device),
                        )

                    elif self.args.model == 'BrainTransformer':
                        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float().to(self.device)
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                        output = self.model(src=batch_x, tgt=dec_inp[:, :self.args.label_len, :])

                    else:
                        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float().to(self.device)
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
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
                dir_path = os.path.join(self.args.result_path, self.args.model, self.args.dataset)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                np.savez(os.path.join(dir_path, 'output_all_' + f_t[:-4] + '.npz'), output_all=output_all, y_all=y_all)
            metrics = torch.asarray(metrics)
            np.savetxt(os.path.join(dir_path, 'metrics.csv'), metrics, delimiter=',')
        return

