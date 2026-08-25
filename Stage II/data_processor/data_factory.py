from data_processor.data_loader import Dataset,Dataset_our
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler, BatchSampler

data_dict = {'boldcast': Dataset_our,
             'others': Dataset
             }


def data_provider(args, flag, f):
    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    elif flag == 'val':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(
        root_path=args.root_path,
        data_path=f,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        drop_short=drop_last,
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
            range(len(data_set.data_x) - args.pred_len - args.label_len -  args.seq_len))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

    return data_set, data_loader