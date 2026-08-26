from data_processor.data_loader import Dataset, Dataset_our
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def data_provider(args, flag, f):
    """Create a subject-level loader.

    BOLDCast uses Dataset_our, which additionally returns ``latent_embed``
    (concatenated Stage-I common/private latent). All other models keep the
    original four-item batch interface through Dataset.
    """
    is_boldcast = args.model == 'BOLDCast'
    Data = Dataset_our if is_boldcast else Dataset

    if flag == 'train':
        shuffle_flag = True
    else:
        shuffle_flag = False

    # Subject files can yield fewer windows than batch_size; never discard all of them.
    drop_last = False
    batch_size = args.batch_size

    dataset_kwargs = dict(
        root_path=args.root_path,
        data_path=f,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        drop_short=drop_last,
    )
    if is_boldcast:
        dataset_kwargs['sp_root'] = args.sp_path
    else:
        dataset_kwargs.update(
            features=args.features,
            freq=args.freq,
        )

    data_set = Data(**dataset_kwargs)

    if args.use_multi_gpu:
        sampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=True,
            drop_last=drop_last,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )

    return data_set, data_loader
