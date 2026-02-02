import os, sys
import time
import torch.nn as nn
from evaluate import evaluate
from torch_geometric.data import Data, Batch
from embedder import embedder
from utils.process import GCN, update_S, drop_feature, Linearlayer
import numpy as np
from tqdm import tqdm
import random as random
import torch
from typing import Any, Optional, Tuple
import torch.nn.functional as F
from sklearn.decomposition import PCA

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)


def dense_to_ind_val(adj):
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)
    index = (torch.isnan(adj) == 0).nonzero(as_tuple=True)
    edge_attr = adj[index]
    return torch.stack(index, dim=0), edge_attr

def graph_data(fmri, corr):
    data_list = []
    for i in range(corr.size()[0]):
        edge_index, edge_attr = dense_to_ind_val(corr[i])
        data_list.append(Data(x=corr[i], edge_index=edge_index, edge_attr=edge_attr))
    graph_batch = Batch.from_data_list(data_list)
    return graph_batch

class GDA(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)   # 处理adj feature label没有涉及到构建模型,feature是一模一样的
        self.args = args
        self.criteria = nn.MSELoss(reduction='sum')
        self.sigm = nn.Sigmoid()
        self.log_sigmoid = nn.LogSigmoid()
        if not os.path.exists(self.args.save_root):
            os.makedirs(self.args.save_root)


    def testing(self):
        ae_model = GNNDAE(self.args).to(self.args.device)  # 开始构建模型，使用的是GNNDAE模型
        state_dict = torch.load('saved_model/best_hcp_Node.pkl')
        ae_model.load_state_dict(state_dict)
        ae_model.eval()
        itr = 1

        mea_func = []
        for i in range(self.args.batch_size):
            mea_func.append(Measure_F(self.args.c_dim, self.args.p_dim,
                                  [self.args.phi_hidden_size] * self.args.phi_num_layers,
                                  [self.args.phi_hidden_size] * self.args.phi_num_layers).to(self.args.device))
        optimizer = torch.optim.Adam([
            {'params': mea_func[0].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
            {'params': mea_func[1].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
            {'params': mea_func[2].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
            {'params': mea_func[3].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
            {'params': ae_model.parameters(), 'lr': self.args.lr_min}
        ], lr=self.args.lr_min)

        pri_all = []
        for iter1, (x_test, y_test, adj_test) in enumerate(self.test_data_loader):
            x_test, y_test, adj_test = x_test.cuda(), y_test.cuda(), adj_test.cuda()
            U = update_S(ae_model, x_test, adj_test, self.args.c_dim, self.args.batch_size)
            loss, match_err, recons, corr, contrastive, common, private = trainmultiplex(ae_model, mea_func, U,
                                                                                         x_test, adj_test, y_test,
                                                                                         self.test_idx_p_list,
                                                                                         self.args,
                                                                                         optimizer,
                                                                                         self.args.device, itr)
            cpu_arrays = [tensor.cpu().detach().numpy() for tensor in common]
            com = np.stack(cpu_arrays, axis=0)
            cpu_arrays = [tensor.cpu().detach().numpy() for tensor in private]
            pri = np.stack(cpu_arrays, axis=0)
            if iter1 == 0:
                com_all = com
                pri_all = pri
            else:
                com_all = np.vstack((com_all,com))
                pri_all = np.vstack((pri_all,pri))
        np.save('com_all.npy', com_all)
        np.save('pri_all.npy', pri_all)


    def training(self):
        seed = self.args.seed

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # # ===================================================#

        ae_model = GNNDAE(self.args).to(self.args.device)    # 开始构建模型，使用的是GNNDAE模型
        # graph independence regularization network
        mea_func = []
        for i in range(self.args.batch_size):
            mea_func.append(Measure_F(self.args.c_dim, self.args.p_dim,
                                  [self.args.phi_hidden_size] * self.args.phi_num_layers,
                                  [self.args.phi_hidden_size] * self.args.phi_num_layers).to(self.args.device))
        # Optimizer
        optimizer = torch.optim.Adam([
            {'params': mea_func[0].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
            {'params': mea_func[1].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
            {'params': mea_func[2].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
            {'params': mea_func[3].parameters(), 'lr': self.args.lr_max, 'weight_decay': self.args.weight_decay},
            {'params': ae_model.parameters(), 'lr': self.args.lr_min}
        ], lr=self.args.lr_min)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # 学习率每50个epoch衰减到原来的1/2

        # model.train()
        ae_model.train()
        best = 1e9
        for i in range(self.args.batch_size):
            mea_func[i].train()
        print("Started training...")
        jy = np.zeros((len(self.data_loader), self.args.c_dim))
        xzh = np.zeros((len(self.data_loader), self.args.p_dim))
        for itr in tqdm(range(1, self.args.num_iters + 1)):
            start_time = time.time()
            total_loss = 0

            for iter, (x, y, adj) in enumerate(self.data_loader):
                x, y, adj = x.cuda(), y.cuda(), adj.cuda()
                # Solve the S subproblem
                U = update_S(ae_model, x, adj, self.args.c_dim, self.args.batch_size)  # common部分的SVD分解的部分组成了U
                # Backprop to update # loss总损失，match_err匹配损失，recons_err + recons_nei重构损失，corr相关性损失，loss_con聚类损失，common公共部分，private私人部分
                loss, match_err, recons, corr, contrastive, common, private = trainmultiplex(ae_model, mea_func, U, x, adj, y,
                                                                                             self.idx_p_list, self.args,
                                                                                             optimizer, self.args.device, itr)
                # jy[iter,:] = np.mean(common[0].cpu().detach().numpy(), axis=0)
                # xzh[iter,:] = np.mean(private[0].cpu().detach().numpy(), axis=0)
                total_loss = loss.item() + total_loss
            # pca = PCA(n_components=2)  # 加载PCA算法，设置降维后主成分数目为2
            # reduced_jy = pca.fit_transform(jy)
            # reduced_xzh = pca.fit_transform(xzh)
            # np.savetxt('com.csv', reduced_jy,delimiter=',')
            # np.savetxt('pri.csv', reduced_xzh,delimiter=',')
            scheduler.step()
            print("第%d个epoch的学习率：%f" % (itr, optimizer.param_groups[0]['lr']))
            end_time = time.time()
            train_loss = total_loss / (iter + 1)
            print('====> Iteration: {} Total-Time= {:.2f} Loss = {:.4f}'.format(itr, end_time-start_time, train_loss))

            # val
            ae_model.eval()
            val_loss = 0
            for iter1, (x_val, y_val, adj_val) in enumerate(self.val_data_loader):
                x_val, y_val, adj_val = x_val.cuda(), y_val.cuda(), adj_val.cuda()
                U = update_S(ae_model, x_val, adj_val, self.args.c_dim, self.args.batch_size)
                loss, match_err, recons, corr, contrastive, common, private = trainmultiplex(ae_model, mea_func, U, x_val, adj_val, y_val,
                                                                                             self.val_idx_p_list, self.args,
                                                                                             optimizer, self.args.device, itr)

            val_loss = loss.item() + val_loss
            print('====> val_loss = {:.4f}'.format(val_loss))
            if val_loss < best:
                best = val_loss
                cnt_wait = 0
                torch.save(ae_model.state_dict(),
                           'saved_model/best_{}_{}.pkl'.format(self.args.dataset, self.args.custom_key))
                print('====> val_loss_best = {:.4f}'.format(best))
            else:
                cnt_wait += 1
                if cnt_wait == self.args.patience:
                    print("Early stopped!")
                    return

def compute_corr(x1, x2):
    # Subtract the mean
    x1_mean = torch.mean(x1, 0, True)
    x1 = x1 - x1_mean
    x2_mean = torch.mean(x2, 0, True)
    x2 = x2 - x2_mean

    # Compute the cross correlation
    sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
    sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
    corr = torch.abs(torch.mean(x1*x2))/(sigma1*sigma2)

    return corr

# The loss function for matching and reconstruction
# common, recons, features, labels, U, idx_p_list, args, epoch
def loss_matching_recons(s, x_hat, x, y, U_batch, idx_p_list, args, epoch):   # 共有，重构，特征，U，idx_p_list
    l = torch.nn.MSELoss(reduction='mean')

    # Matching loss
    match_err = l(torch.cat(s, 1), U_batch.repeat(1, args.batch_size))/x[0].shape[1]    # s共有表示，batch的公共变量U
    recons_err = 0
    # Feature reconstruction loss
    for i in range(args.batch_size):
        recons_err += l(x_hat[i], x[i])     # x_hat是重构特征，x是输入特征， recons_err是特征重构损失
    recons_err /= x[0].shape[1]    # 取平均

    # Topology reconstruction loss
    interval = int(args.neighbor_num/args.sample_neighbor)
    neighbor_embedding = []
    for i in range(args.batch_size):
        neighbor_embedding_0 = []
        for j in range(0, args.sample_neighbor+1):
            neighbor_embedding_0.append(x[i][idx_p_list[i][(epoch + interval * j) % args.neighbor_num]])
        neighbor_embedding.append(sum(neighbor_embedding_0) / args.sample_neighbor)
    recons_nei = 0
    for i in range(args.batch_size):
        recons_nei += l(x_hat[i], neighbor_embedding[i])   # neighbor_embedding是节点vi的随机采样的一阶邻居的特征均值，包含拓扑信息
    recons_nei /= x[0].shape[1]   # recons_nei是拓扑重构损失

    return match_err, recons_err, recons_nei    # 整体的重构损失包含recons_err和recons_nei


# The loss function for independence regularization
def loss_independence(phi_c_list, psi_p_list, batch_size):
    # Correlation
    corr2 = 0

    for i in range(len(phi_c_list)):
        if i == 0:
            phi_c = phi_c_list[i]
            psi_p = psi_p_list[i]
        else:
            phi_c = torch.cat((phi_c, phi_c_list[i]), 1)
            psi_p = torch.cat((psi_p, psi_p_list[i]), 1)

        corr2 += compute_corr(phi_c_list[i], psi_p_list[i])
    corr2 = corr2 / batch_size    # 越小越好
    corr1 = (torch.sum(torch.corrcoef(phi_c.T)) - batch_size) / batch_size  # 越大越好
    corr = corr1  - corr2
    return corr


# cContrastive loss
def loss_contrastive(U, private, adj_list, args):
    i = 0
    loss = 0
    for adj in adj_list:
        adj = adj_list[i]
        out_node = adj.to_sparse()._indices()[1]
        random = np.random.randint(out_node.shape[0], size=int((out_node.shape[0] / args.sample_num)))
        sample_edge = adj.to_sparse()._indices().T[random]
        dis = F.cosine_similarity(U[sample_edge.T[0]],U[sample_edge.T[1]])
        a, maxidx = torch.sort(dis, descending=True)
        idx1 = maxidx[:int(sample_edge.shape[0]*0.2)]
        b, minidx = torch.sort(dis, descending=False)
        idx2 = minidx[:int(sample_edge.shape[0]*0.1)]
        private_sample_0 = private[i][sample_edge[idx1].T[0]]
        private_sample_1 = private[i][sample_edge[idx1].T[1]]
        private_sample_2 = private[i][sample_edge[idx2].T[0]]
        private_sample_3 = private[i][sample_edge[idx2].T[1]]
        i += 1
        loss += semi_loss(private_sample_0, private_sample_1, private_sample_2, private_sample_3, args)
    return loss


def semi_loss(z1, z2, z3, z4, args):
    f = lambda x: torch.exp(x / args.tau)
    positive = f(F.cosine_similarity(z1, z2))
    negative = f(F.cosine_similarity(z3, z4))
    return -torch.log(
        positive.sum()
        / (positive.sum() + negative.sum() ))

def trainmultiplex(model, mea_func, U, features, adj_list, labels, idx_p_list, args,  optimizer, device, epoch):

    model.train()
    for i in range(args.batch_size):
        mea_func[i].train()
    common_mean, common, private, recons = model(features, adj_list)      # 所以8的是common，2的是private，recons重构的特征
    match_err, recons_err, recons_nei = loss_matching_recons(common, recons, features, labels, U, idx_p_list, args, epoch)   # 整体的重构损失包含recons_err和recons_nei，match_err是匹配损失，对公共表示common和公共变量U之间进行匹配损失，以捕获它们之间完整的公共信息。
    # Independence regularizer loss
    phi_c_list = []
    psi_p_list = []
    for i in range(args.batch_size):
        phi_c, psi_p = mea_func[i](common[i], private[i])
        phi_c_list.append(phi_c)
        psi_p_list.append(psi_p)
    corr = loss_independence(phi_c_list, psi_p_list, args.batch_size)   # 相关性
    loss_con = loss_contrastive(U, private, adj_list, args)   # 聚合相似节点，分离不相似的节点
    # Compute the overall loss, note that we use the gradient reversal trick
    # and that's why we have a 'minus' for the last term
    loss =  match_err + args.alpha * (recons_err + recons_nei) - args.beta * corr + args.lammbda * loss_con


    # Update all the parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # return loss, match_err, recons_err + recons_nei, corr, common, private
    return loss, match_err, recons_err + recons_nei, corr, loss_con, common, private




class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output
    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

def grad_reverse(x, coeff):
    return GradientReversalLayer.apply(x, coeff)


class GNNEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pipe = GCN(args.ft_size, args.hid_units, args.activation, args.dropout, args.isBias)
        # map to common
        self.S = nn.Linear(args.hid_units, args.c_dim)    # 256,8
        # map to private
        self.P = nn.Linear(args.hid_units, args.p_dim)    # 256,8

    def forward(self, x, adj):
        tmp = self.pipe(x, adj)
        common = self.S(tmp)
        private = self.P(tmp)
        return common, private

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.linear1 = Linearlayer(args.decolayer,args.c_dim + args.p_dim, args.hid_units, args.ft_size)
        self.linear2 = nn.Linear(args.ft_size, args.ft_size)

    def forward(self, s, p):
        recons = self.linear1(torch.cat((s, p), 1))
        recons = self.linear2(F.relu(recons))
        return recons

class GNNDAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_view = self.args.batch_size
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for _ in range(self.num_view):
            self.encoder.append(GNNEncoder(args))
            self.decoder.append(Decoder(args))

    def encode(self, x, adj_list):
        common = []
        private = []
        for i in range(self.num_view):
            if torch.isnan(adj_list[i,:,:]).any():
                adj = torch.nan_to_num(adj_list[i,:,:], nan=0.0, posinf=1e6, neginf=-1e6)
            else:
                adj = adj_list[i,:,:]
            tmp = self.encoder[i](x[i], adj)
            common.append(tmp[0])
            private.append(tmp[1])
        common_mean = sum(common)/ (self.num_view * len(tmp))
        return common_mean, common, private

    def decode(self, common, private):
        recons = []
        for i in range(self.num_view):
            tmp = self.decoder[i](common[i], private[i])
            # tmp = self.decoder[i](common[i], p[i])
            recons.append(tmp)

        return recons

    def forward(self, x, adj):
        common_mean, common, private = self.encode(x, adj)    # 所以8的是common，2的是private
        recons = self.decode(common, private)

        return common_mean, common, private, recons

    def embed(self, x, adj_list):
        common = []
        private = []
        for i in range(self.num_view):
            tmp = self.encoder[i](x[i], adj_list[i])
            common.append(tmp[0].detach())
            private.append(tmp[1].detach())
        return common, private

class MLP(nn.Module):
    def __init__(self, input_d, structure, output_d, dropprob=0.0):
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropprob)
        struc = [input_d] + structure + [output_d]

        for i in range(len(struc)-1):
            self.net.append(nn.Linear(struc[i], struc[i+1]))

    def forward(self, x):
        for i in range(len(self.net)-1):
            x = F.relu(self.net[i](x))
            x = self.dropout(x)

        # For the last layer
        y = self.net[-1](x)

        return y

# measurable functions \phi and \psi
class Measure_F(nn.Module):
    def __init__(self, view1_dim, view2_dim, phi_size, psi_size, latent_dim=1):
        super(Measure_F, self).__init__()
        self.phi = MLP(view1_dim, phi_size, latent_dim)   # [8,256], [256,256], [256,1]
        self.psi = MLP(view2_dim, psi_size, latent_dim)   # [2,256], [256,256], [256,1]
        # gradient reversal layer
        self.grl1 = GradientReversalLayer()    # 梯度反转
        self.grl2 = GradientReversalLayer()    # 梯度反转

    def forward(self, x1, x2):
        y1 = self.phi(grad_reverse(x1,1))
        y2 = self.psi(grad_reverse(x2,1))
        return y1, y2







