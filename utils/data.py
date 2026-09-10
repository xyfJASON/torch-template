from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_data_iterator(
    dataloader: DataLoader,
    datasampler: DistributedSampler,
    consumed_batches: int,
):
    epoch, first_offset = divmod(consumed_batches, len(dataloader))
    while True:
        datasampler.set_epoch(epoch)
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= first_offset:
                yield batch
        first_offset = 0
        epoch += 1
