import os
import random
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Any, Optional, Tuple

from utils.embedder import embedder
from utils.process import GCN, Linearlayer, update_S

torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)


def build_idx_p_list(adj_list, args):
    """Build the same cyclic neighbor index table as the original embedder.

    The old code precomputed this table globally and then accidentally reused
    the first ``batch_size`` subjects for every batch. Building it per batch
    preserves the same sampling rule while matching each subject to its own
    adjacency matrix.
    """
    idx_p_list = []
    for i in range(adj_list.shape[0]):
        adj = torch.nan_to_num(adj_list[i], nan=0.0, posinf=0.0, neginf=0.0)
        sparse = adj.to_sparse()
        indices = sparse._indices()

        # A fully zero row has no sparse neighbor. Use the node itself so the
        # original cyclic sampling remains well-defined without changing shape.
        if indices.numel() == 0:
            idx_p_list.append([torch.arange(args.num_rois, device=adj.device) for _ in range(args.neighbor_num)])
            continue

        row = indices[0]
        col = indices[1]
        per_node_neighbors = []
        for node in range(args.num_rois):
            neighbors = col[row == node]
            if neighbors.numel() == 0:
                neighbors = torch.tensor([node], device=adj.device, dtype=torch.long)
            per_node_neighbors.append(neighbors)

        subject_idx = []
        for j in range(1, args.neighbor_num + 1):
            chosen = []
            for node, neighbors in enumerate(per_node_neighbors):
                chosen.append(neighbors[j % neighbors.numel()])
            subject_idx.append(torch.stack(chosen))
        idx_p_list.append(subject_idx)

    return idx_p_list


class GDA(embedder):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.criteria = nn.MSELoss(reduction='sum')
        self.sigm = nn.Sigmoid()
        self.log_sigmoid = nn.LogSigmoid()
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)

    def _build_model_and_measures(self):
        ae_model = GNNDAE(self.args).to(self.args.device)
        mea_func = [
            Measure_F(
                self.args.c_dim,
                self.args.p_dim,
                [self.args.phi_hidden_size] * self.args.phi_num_layers,
                [self.args.phi_hidden_size] * self.args.phi_num_layers,
            ).to(self.args.device)
            for _ in range(self.args.batch_size)
        ]
        return ae_model, mea_func

    def _build_optimizer(self, ae_model, mea_func):
        parameter_groups = [
            {
                'params': measure.parameters(),
                'lr': self.args.lr_max,
                'weight_decay': self.args.weight_decay,
            }
            for measure in mea_func
        ]
        parameter_groups.append({'params': ae_model.parameters(), 'lr': self.args.lr_min})
        return torch.optim.Adam(parameter_groups, lr=self.args.lr_min)

    @staticmethod
    def _set_training_mode(ae_model, mea_func, training):
        ae_model.train(training)
        for measure in mea_func:
            measure.train(training)

    def _reduce_learning_rate_once(self, optimizer):
        for group in optimizer.param_groups:
            group['lr'] *= self.args.lr_reduce_factor
        print(
            '====> Validation reconstruction has not improved for '
            f'{self.args.lr_reduce_patience} epochs; learning rates multiplied by '
            f'{self.args.lr_reduce_factor}.'
        )
        print('====> Current learning rates:', [group['lr'] for group in optimizer.param_groups])

    def _validate(self, ae_model, mea_func, epoch):
        self._set_training_mode(ae_model, mea_func, training=False)
        total_reconstruction = 0.0
        total_subjects = 0

        with torch.no_grad():
            for x_val, y_val, adj_val, _, real_count in self.val_data_loader:
                x_val = x_val.to(self.args.device)
                y_val = y_val.to(self.args.device)
                adj_val = adj_val.to(self.args.device)
                idx_p_list = build_idx_p_list(adj_val, self.args)
                _, _, _, recons = ae_model(x_val, adj_val)
                # Use a fixed neighbor-sampling phase for validation so the
                # checkpoint metric is comparable across epochs.
                batch_reconstruction = reconstruction_metric(
                    recons, x_val, idx_p_list, self.args, 0, real_count
                )
                total_reconstruction += batch_reconstruction.item() * real_count
                total_subjects += real_count

        if total_subjects == 0:
            raise RuntimeError('Validation split contains no subjects.')
        return total_reconstruction / total_subjects

    def training(self):
        seed = self.args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        ae_model, mea_func = self._build_model_and_measures()
        optimizer = self._build_optimizer(ae_model, mea_func)

        best_val_reconstruction = float('inf')
        no_improve_count = 0
        lr_reduced_for_current_streak = False

        print('Started Stage-I training (train + test subjects; val is selection only)...')
        for epoch in tqdm(range(1, self.args.num_iters + 1)):
            start_time = time.time()
            self._set_training_mode(ae_model, mea_func, training=True)

            total_loss = 0.0
            total_batches = 0
            for x, y, adj, _, _ in self.data_loader:
                x = x.to(self.args.device)
                y = y.to(self.args.device)
                adj = adj.to(self.args.device)
                idx_p_list = build_idx_p_list(adj, self.args)

                U = update_S(ae_model, x, adj, self.args.c_dim, self.args.batch_size)
                loss, match_err, recons, corr, contrastive, common, private = trainmultiplex(ae_model,mea_func,U,x,adj,y,idx_p_list,self.args,optimizer,self.args.device,epoch)
                total_loss += loss.item()
                total_batches += 1

            train_loss = total_loss / max(total_batches, 1)
            val_reconstruction = self._validate(ae_model, mea_func, epoch)
            elapsed = time.time() - start_time

            print(
                f'====> Epoch: {epoch} Time={elapsed:.2f}s '
                f'TrainLoss={train_loss:.6f} ValReconstruction={val_reconstruction:.6f}'
            )

            improved = val_reconstruction < (best_val_reconstruction - self.args.min_delta)
            if improved:
                best_val_reconstruction = val_reconstruction
                no_improve_count = 0
                lr_reduced_for_current_streak = False
                torch.save(ae_model.state_dict(), self.args.checkpoint_path)
                print(
                    f'====> Best validation reconstruction = {best_val_reconstruction:.6f}; '
                    f'saved {self.args.checkpoint_path}'
                )
            else:
                no_improve_count += 1
                print(
                    f'====> No validation improvement: {no_improve_count}/'
                    f'{self.args.early_stop_patience}'
                )
                if (
                    no_improve_count >= self.args.lr_reduce_patience
                    and not lr_reduced_for_current_streak
                ):
                    self._reduce_learning_rate_once(optimizer)
                    lr_reduced_for_current_streak = True

                if no_improve_count >= self.args.early_stop_patience:
                    print('Early stopped: validation reconstruction did not improve.')
                    break

        if not os.path.isfile(self.args.checkpoint_path):
            raise RuntimeError('Training finished without producing a best-model checkpoint.')
        return self.args.checkpoint_path

    @staticmethod
    def _atomic_write_latents(file_path, com, priv):
        file_path = os.path.abspath(file_path)
        with np.load(file_path, allow_pickle=True) as source:
            content = {key: source[key] for key in source.files}
        content['com'] = com.astype(np.float32, copy=False)
        content['priv'] = priv.astype(np.float32, copy=False)

        directory = os.path.dirname(file_path)
        with tempfile.NamedTemporaryFile(suffix='.npz', dir=directory, delete=False) as tmp:
            tmp_path = tmp.name
        try:
            np.savez(tmp_path, **content)
            os.replace(tmp_path, file_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def extract_latents(self):
        """Load the best checkpoint and append com/priv to every subject NPZ."""
        if not os.path.isfile(self.args.checkpoint_path):
            raise FileNotFoundError(
                f'Best Stage-I checkpoint not found: {self.args.checkpoint_path}'
            )

        ae_model = GNNDAE(self.args).to(self.args.device)
        state_dict = torch.load(self.args.checkpoint_path, map_location=self.args.device)
        ae_model.load_state_dict(state_dict)
        ae_model.eval()

        print(f'Loading best checkpoint for latent extraction: {self.args.checkpoint_path}')
        with torch.no_grad():
            for split, loader in self.latent_loaders.items():
                written = 0
                for x, _, adj, paths, real_count in loader:
                    x = x.to(self.args.device)
                    adj = adj.to(self.args.device)
                    _, common, private = ae_model.encode(x, adj)

                    for i in range(real_count):
                        com = common[i].detach().cpu().numpy()
                        priv = private[i].detach().cpu().numpy()
                        self._atomic_write_latents(paths[i], com, priv)
                        written += 1

                print(f'====> {split}: wrote com/priv to {written} subject NPZ files.')

        print('Stage-I latent extraction completed for train/val/test.')


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

def compute_multiplex_losses(model, mea_func, U, features, adj_list, labels, idx_p_list, args, epoch):
    """Compute the original Stage-I losses without changing model parameters."""
    common_mean, common, private, recons = model(features, adj_list)
    match_err, recons_err, recons_nei = loss_matching_recons(
        common, recons, features, labels, U, idx_p_list, args, epoch
    )

    phi_c_list = []
    psi_p_list = []
    for i in range(args.batch_size):
        phi_c, psi_p = mea_func[i](common[i], private[i])
        phi_c_list.append(phi_c)
        psi_p_list.append(psi_p)

    corr = loss_independence(phi_c_list, psi_p_list, args.batch_size)
    loss_con = loss_contrastive(U, private, adj_list, args)
    reconstruction = recons_err + recons_nei
    loss = match_err + args.alpha * reconstruction - args.beta * corr + args.lammbda * loss_con
    return loss, match_err, reconstruction, corr, loss_con, common, private, recons


def trainmultiplex(model, mea_func, U, features, adj_list, labels, idx_p_list, args, optimizer, device, epoch):
    """Original Stage-I optimization step, separated from validation."""
    model.train()
    for i in range(args.batch_size):
        mea_func[i].train()

    outputs = compute_multiplex_losses(
        model, mea_func, U, features, adj_list, labels, idx_p_list, args, epoch
    )
    loss = outputs[0]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return outputs[:7]


def reconstruction_metric(recons, features, idx_p_list, args, epoch, real_count):
    """Validation reconstruction score on real (non-padding) subjects only.

    This mirrors the original feature + topology reconstruction terms but does
    not include duplicated padding subjects in the checkpoint criterion.
    """
    mse = torch.nn.MSELoss(reduction='mean')
    recons_err = 0.0
    recons_nei = 0.0
    interval = max(1, int(args.neighbor_num / args.sample_neighbor))

    for i in range(real_count):
        recons_err = recons_err + mse(recons[i], features[i])
        neighbor_embedding = []
        for j in range(0, args.sample_neighbor + 1):
            neighbor_embedding.append(
                features[i][idx_p_list[i][(epoch + interval * j) % args.neighbor_num]]
            )
        neighbor_embedding = sum(neighbor_embedding) / args.sample_neighbor
        recons_nei = recons_nei + mse(recons[i], neighbor_embedding)

    # Keep the same ROI-feature normalization used in the original loss.
    denom = features[0].shape[1]
    return (recons_err + recons_nei) / (real_count * denom)




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







