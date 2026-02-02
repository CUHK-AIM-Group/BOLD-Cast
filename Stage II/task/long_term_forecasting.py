from data_processor.data_factory import data_provider
from task.basic import Basic
import torch
import torch.nn as nn
from torch import optim
import os
import warnings
import numpy as np
import random
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        self.device = self.args.gpu_lLama
        model = model.to(self.device)

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

    def _select_criterion(self):
        criterion = nn.MSELoss('sum')
        return criterion

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            if not os.path.exists(path):
                os.makedirs(path)
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
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    output = self.model(batch_x, batch_x_mark)
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
                        output = self.vali(batch_x, batch_x_mark)
                        if ind == 0:
                            output_all = output.detach().cpu()
                            y_all = batch_y
                            ind = ind + 1
                        else:
                            output_all = torch.vstack((output_all, output.detach().cpu()))
                            y_all = torch.vstack((y_all, batch_y))

                    y_all = y_all * (vali_data.max.T - vali_data.min.T) + vali_data.min.T
                    output_all = output_all * (vali_data.max.T - vali_data.min.T) + vali_data.min.T
                    vali_loss = torch.mean(torch.abs(output_all - y_all))
                    vali_loss_all = vali_loss + vali_loss_all


                vali_loss_all =  vali_loss_all / len(sample2)
                if vali_loss_all < vali_loss_best:
                    vali_loss_best = vali_loss_all
                    cnt_wait = 0
                    torch.save(self.model.state_dict(), 'saved_model/best_model_abide.pth')
                    print('====> val_loss_best = {:.4f}'.format(vali_loss_best))
                else:
                    cnt_wait += 1
                    scheduler.step()
                    print("lr = {:.8f}".format(model_optim.param_groups[0]['lr']))
                    print('====> val_loss_best = {:.4f}, vali_loss_all= {:.4f}'.format(vali_loss_best, vali_loss_all))
                    if cnt_wait == self.args.patience:
                        print("Early stopped!")
                        break

        # 修改后的测试部分代码
        best_model_path = 'saved_model/best_model_abide.pth'
        self.model.load_state_dict(torch.load(best_model_path), strict=False)
        filenames_t = os.listdir(self.args.root_path + 'test')
        rate = 1.0

        picknumber2 = int(len(filenames_t) * rate)
        sample2 = random.sample(filenames_t, picknumber2)
        metrics = np.zeros((len(sample2), 7))
        self.model.eval()
        xxx = 0
        with torch.no_grad():
            for f_t in sample2:
                # 加载整个测试序列
                test_data, test_loader = self._get_data(flag='test', f=f_t)

                # 获取整个序列数据 - 创建原始数据的完整副本
                data_x = test_data.data_x.numpy().copy() if isinstance(test_data.data_x,
                                                                       torch.Tensor) else test_data.data_x.copy()
                data_stamp = test_data.data_stamp.numpy().copy() if isinstance(test_data.data_stamp,
                                                                               torch.Tensor) else test_data.data_stamp.copy()

                # 保存原始数据的完整副本（用于最终的真实目标）
                original_data = data_x.copy()

                T = data_x.shape[0]  # 序列总长度
                n_vars = data_x.shape[1]  # 变量数

                # 确保预测长度为 T - (self.args.his_len + self.args.pre_len)
                total_length = T - (self.args.his_len + self.args.pre_len)
                total_predictions = np.zeros((total_length, n_vars))

                # 真实目标值应该取自原始数据，而不是被替换的data_x
                total_targets = original_data[(self.args.his_len + self.args.pre_len) - self.args.pre_len: T - self.args.pre_len]

                # 自回归预测循环
                for start_idx in range(0, total_length, self.args.pre_len):
                    end_idx = start_idx + (self.args.his_len + self.args.pre_len)
                    pred_end = min(start_idx + self.args.pre_len, total_length)

                    # 计算实际需要预测的长度
                    pred_length = pred_end - start_idx

                    # 当前输入窗口 (确保长度始终为(self.args.his_len + self.args.pre_len))
                    current_x = data_x[start_idx:end_idx]
                    current_stamp = data_stamp[start_idx:end_idx]

                    # 转换为张量
                    batch_x = torch.FloatTensor(current_x).unsqueeze(0).to(self.device)
                    batch_x_mark = torch.FloatTensor(current_stamp).unsqueeze(0).to(self.device)

                    # 模型预测
                    output = self.model(batch_x, batch_x_mark)
                    output = output.squeeze(0).detach().cpu().numpy()

                    # 取实际需要的预测结果
                    preds = output[:pred_length]

                    # 存储预测结果
                    total_predictions[start_idx:start_idx + pred_length] = preds

                    # 自回归反馈 - 仅替换输入序列中的预测部分
                    # 保留原始目标值不变！
                    if start_idx + self.args.pre_len < total_length:
                        # 更新输入序列的预测位置
                        replace_start = end_idx  # 预测结束后位置
                        replace_end = min(replace_start + pred_length, T)
                        data_x[replace_start:replace_end] = preds[:replace_end - replace_start]

                # 反归一化 - 使用原始归一化参数
                if isinstance(test_data.max, torch.Tensor):
                    max_val = test_data.max.detach().cpu().numpy().T
                    min_val = test_data.min.detach().cpu().numpy().T
                else:
                    max_val = test_data.max.T
                    min_val = test_data.min.T

                y_all1 = total_targets * (max_val - min_val) + min_val
                output_all1 = total_predictions * (max_val - min_val) + min_val

                # 计算指标
                y_all1_tensor = torch.FloatTensor(y_all1[:int(1*self.args.pre_len),:])
                output_all1_tensor = torch.FloatTensor(output_all1[:int(1*self.args.pre_len),:])

                mape_mask = torch.abs(y_all1_tensor) > 0.2
                test_loss = torch.mean(torch.abs(output_all1_tensor - y_all1_tensor))

                # 计算其他指标
                sigma_p = (output_all1_tensor).std(dim=0)
                sigma_g = (y_all1_tensor).std(dim=0)
                mean_p = output_all1_tensor.mean(dim=0)
                mean_g = y_all1_tensor.mean(dim=0)

                valid_mask = (sigma_g != 0) & (sigma_p != 0)
                covariance = torch.mean((output_all1_tensor - mean_p) * (y_all1_tensor - mean_g), dim=0)
                correlation = covariance[valid_mask] / (sigma_p[valid_mask] * sigma_g[valid_mask])

                metrics[xxx, 0] = torch.mean(torch.abs(output_all1_tensor - y_all1_tensor))
                metrics[xxx, 1] = torch.median(torch.abs(output_all1_tensor - y_all1_tensor))
                metrics[xxx, 2] = torch.sqrt(torch.mean((output_all1_tensor - y_all1_tensor) ** 2))

                # 安全计算MAPE
                valid_mask = torch.abs(y_all1_tensor) > 0.2
                if torch.any(valid_mask):
                    jj = torch.abs(
                        (output_all1_tensor[valid_mask] - y_all1_tensor[valid_mask]) / y_all1_tensor[valid_mask])
                    jj = torch.where(torch.isnan(jj), torch.zeros_like(jj), jj)
                    metrics[xxx, 3] = 100 * torch.mean(jj)
                else:
                    metrics[xxx, 3] = 0  # 避免除零错误

                metrics[xxx, 4] = torch.max(correlation)
                # metrics[xxx,5] = int(f_t[:5])
                # metrics[xxx,6] = test_data.label

                print(
                    f'====> test_loss = {test_loss.item():.4f}, mse = {metrics[xxx, 0]:.4f}, medae = {metrics[xxx, 1]:.4f}, '
                    f'rmse = {metrics[xxx, 2]:.4f}, mape = {metrics[xxx, 3]:.4f}%, correlation = {metrics[xxx, 4]:.4f}')
                print(f'预测长度: {len(output_all1)}, 真实长度: {len(y_all1)}')

                xxx += 1
                np.savez('output_all' + f_t[:6] + '.npz', output_all=output_all1, y_all=y_all1)

        np.savetxt('metircs.csv', metrics, delimiter=',')
        return self.model

        # test
        best_model_path = 'saved_model/best_model.pth'
        self.model.load_state_dict(torch.load(best_model_path), strict=False)
        filenames_t = os.listdir(self.args.root_path + 'test')

        metrics = np.zeros((len(filenames_t),7))
        self.model.eval()
        xxx = 0
        with torch.no_grad():
            for f_t in filenames_t:
                ind = 0
                test_data, test_loader = self._get_data(flag='test', f=f_t)
                for i, (batch_x, batch_y, batch_x_mark) in enumerate(test_loader):
                    output = self.vali(batch_x, batch_x_mark)
                    if ind == 0:
                        output_all1 = output[ 0, :, :].detach().cpu()
                        y_all1 = batch_y[ 0, :, :]
                        ind = ind + 1
                    else:
                        output_all1 = torch.vstack((output_all1,output[ 0, :, :].detach().cpu()))
                        y_all1 = torch.vstack((y_all1, batch_y[ 0, :, :]))

                mape_mask = torch.abs(y_all1) > 0.2
                y_all1 = y_all1 * (test_data.max.T - test_data.min.T) + test_data.min.T
                output_all1 = output_all1 * (test_data.max.T - test_data.min.T) + test_data.min.T

                test_loss = torch.mean(torch.abs(output_all1 - y_all1))
                sigma_p = (output_all1).std(axis=0)
                sigma_g = (y_all1).std(axis=0)
                mean_p = output_all1.mean(axis=0)
                mean_g = y_all1.mean(axis=0)
                index = (sigma_g != 0)
                correlation = ((output_all1 - mean_p) * (y_all1 - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
                correlation = (correlation[index]).mean()

                metrics[xxx,0] = torch.mean(torch.abs(output_all1 - y_all1))
                metrics[xxx,1] = torch.median(torch.abs(output_all1 - y_all1))
                metrics[xxx,2] = torch.sqrt(torch.mean((output_all1 - y_all1) ** 2))
                jj = torch.abs((output_all1[mape_mask] - y_all1[mape_mask]) / y_all1[mape_mask])
                jj = torch.where(torch.isnan(jj), torch.full_like(jj, 0), jj)
                metrics[xxx,3] = 100 * torch.mean(jj)
                metrics[xxx,4] = correlation

                print(
                    '====> test_loss = {:.4f}, mse = {:.4f}, medae = {:.4f}, rmse = {:.4f}, mape = {:.4f}%, correlation = {:.4f}, '.format(
                        test_loss, metrics[xxx, 0], metrics[xxx, 1], metrics[xxx, 2], metrics[xxx, 3], metrics[xxx, 4]))
                xxx = xxx + 1
                np.savez('output_all'+f_t[:6]+'.npz', output_all =output_all1, y_all=y_all1)
                np.savetxt('pre.csv', output_all1, delimiter = ',')
                np.savetxt('gt.csv', y_all1, delimiter = ',')
                np.savez('output_all'+f_t[:-4]+'.npz', output_all =output_all1, y_all=y_all1)
                metrics = torch.asarray(metrics)

            np.savetxt('metircs.csv', metrics , delimiter=',')
        return self.model

    def vali(self, data_x, data_stamp, is_test=False):
        self.model.eval()
        with torch.no_grad():
            output = self.model(data_x.float().to(self.device), data_stamp.float().to(self.device))
        return output
