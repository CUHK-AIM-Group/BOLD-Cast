from data_processor.data_loader import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler, BatchSampler

data_dict = {'data': Dataset}


def data_provider(args, flag, f):
    Data = data_dict[args.data]
    seq_len = args.pre_len + args.his_len
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size_val
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size_val
    else:
        shuffle_flag = True
        drop_last = args.drop_last   # drop last batch in data loader
        batch_size = args.batch_size
    data_set = Data(
        root_path=args.root_path,
        data_path=f,
        flag=flag,
        size=[args.his_len + args.pre_len, args.his_len, args.pre_len],
        drop_short=args.drop_short,     # drop too short sequences in dataset
    )

    if args.use_multi_gpu:
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(data_set,
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last,
            )
    else:
        sequential_sampler = SequentialSampler(
            range(len(data_set.data_x) - args.pre_len - args.his_len -  seq_len))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=sequential_sampler,
            num_workers=args.num_workers,
            drop_last=drop_last)
    return data_set, data_loader