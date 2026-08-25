import torch
from utils import process
import torch.utils.data as data
from torch_geometric.utils import degree,remove_self_loops
class NetDataSet(data.Dataset):
    def __init__(self, x, y, A, isTrainSet=False):
        self.isTrainSet = isTrainSet
        self.x = x.astype('float32')
        self.y = y.astype('float32')
        self.A = A.astype('float32')

    def __getitem__(self, index):
        if self.isTrainSet:
            return self.x[index], self.y[index], self.A[index]
            # return self.x[index].transpose(1, 0, 2), self.y[index].transpose(2, 1, 0), self.x_[index].transpose(1, 0, 2)
        else:
            return self.x[index], self.y[index], self.A[index]

    def __len__(self):
        return self.x.shape[0]

def DataLoader(x, y, A, batch_size, isTrainSet=False, shuffle=True):
    dataset = NetDataSet(x, y, A, isTrainSet)
    data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader
class embedder:
    def __init__(self, args):
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == -1:
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")
        adj_list,  features,  labels = process.load_h5('data/hcp-d/train.npz')
        data_loader = DataLoader(features, labels, adj_list, isTrainSet=True, batch_size=args.batch_size, shuffle=False)
        args.ft_size = features.shape[2]
        args.nb_nodes = adj_list.shape[1]
        idx_p_list = []
        sample_edge_list = []
        for i in range(adj_list.shape[0]):
            adj = torch.tensor(adj_list[i,:,:])
            feature = features[i,:,:]
            deg_list_0 = []
            idx_p_list_0 = []
            deg_list_0.append(0)
            A_degree = degree(adj.to_sparse()._indices()[0], feature.shape[0],
                              dtype=int).tolist()  # 计算给定一维索引张量的度。degree(index, num_nodes, dtype) index:图的一维索引，num_nodes图上节点数量，dtype数据类型
            out_node = adj.to_sparse()._indices()[1]
            for k in range(feature.shape[0]):  # features.shape[0] = nb_nodes
                deg_list_0.append(deg_list_0[-1] + A_degree[k])  # deg_list_0就是A_degree在最前面多了一个0
            for j in range(1, args.neighbor_num + 1):
                random_list = [deg_list_0[k] + j % A_degree[k] for k in range(feature.shape[0])]
                idx_p_0 = out_node[random_list]
                idx_p_list_0.append(idx_p_0)
            idx_p_list.append(idx_p_list_0)

        val_adj_list, val_features, val_labels = process.load_h5('data/hcp-d/val.npz')
        val_data_loader = DataLoader(val_features, val_labels, val_adj_list, isTrainSet=False, batch_size=args.batch_size, shuffle=False)
        args.ft_size = val_features.shape[2]
        args.nb_nodes = val_adj_list.shape[1]
        val_idx_p_list = []
        for i in range(val_adj_list.shape[0]):
            adj = torch.tensor(val_adj_list[i, :, :])
            feature = val_features[i, :, :]
            deg_list_0 = []
            idx_p_list_0 = []
            deg_list_0.append(0)
            A_degree = degree(adj.to_sparse()._indices()[0], feature.shape[0],
                              dtype=int).tolist()  # 计算给定一维索引张量的度。degree(index, num_nodes, dtype) index:图的一维索引，num_nodes图上节点数量，dtype数据类型
            out_node = adj.to_sparse()._indices()[1]
            for k in range(feature.shape[0]):  # features.shape[0] = nb_nodes
                deg_list_0.append(deg_list_0[-1] + A_degree[k])  # deg_list_0就是A_degree在最前面多了一个0
            for j in range(1, args.neighbor_num + 1):
                random_list = [deg_list_0[k] + j % A_degree[k] for k in range(feature.shape[0])]
                idx_p_0 = out_node[random_list]
                idx_p_list_0.append(idx_p_0)
            val_idx_p_list.append(idx_p_list_0)

        test_adj_list, test_features, test_labels = process.load_h5('data/hcp/val.npz')
        test_data_loader = DataLoader(test_features, test_labels, test_adj_list, isTrainSet=False, batch_size=args.batch_size, shuffle=False)
        args.ft_size = test_features.shape[2]
        args.nb_nodes = test_adj_list.shape[1]
        test_idx_p_list = []
        for i in range(test_adj_list.shape[0]):
            adj = torch.tensor(test_adj_list[i, :, :])
            feature = test_features[i, :, :]
            deg_list_0 = []
            idx_p_list_0 = []
            deg_list_0.append(0)
            A_degree = degree(adj.to_sparse()._indices()[0], feature.shape[0],
                              dtype=int).tolist()
            out_node = adj.to_sparse()._indices()[1]
            for k in range(feature.shape[0]):
                deg_list_0.append(deg_list_0[-1] + A_degree[k])  # deg_list_0就是A_degree在最前面多了一个0
            for j in range(1, args.neighbor_num + 1):
                random_list = [deg_list_0[k] + j % A_degree[k] for k in range(feature.shape[0])]
                idx_p_0 = out_node[random_list]
                idx_p_list_0.append(idx_p_0)
            test_idx_p_list.append(idx_p_list_0)

        self.adj_list = torch.FloatTensor(adj_list).to(args.device)
        self.val_adj_list = torch.FloatTensor(val_adj_list).to(args.device)
        self.test_adj_list = torch.FloatTensor(test_adj_list).to(args.device)

        self.features = torch.FloatTensor(features).to(args.device)
        self.val_features = torch.FloatTensor(val_features).to(args.device)
        self.test_features = torch.FloatTensor(test_features).to(args.device)

        self.labels = torch.FloatTensor(labels).to(args.device)
        self.val_labels = torch.FloatTensor(val_labels).to(args.device)
        self.test_labels = torch.FloatTensor(test_labels).to(args.device)

        self.data_loader = data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

        self.idx_p_list = idx_p_list
        self.val_idx_p_list = val_idx_p_list
        self.test_idx_p_list = test_idx_p_list

        self.sample_edge_list = sample_edge_list





