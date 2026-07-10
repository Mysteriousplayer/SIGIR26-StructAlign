from torch.utils.data import DataLoader
from datasets.msrvtt_dataset_new import MSRVTTDataset
from datasets.actnet_dataset_new import ACTNETDataset

# Dataset wiring in this open-source copy follows the StructAlign pipeline.
# Legacy architectures are kept in the repository for reference, but their
# original data-layer behavior is not guaranteed without additional adaptation.

class DataFactory:

    @staticmethod
    def get_data_loader(config, dataset_name, data, model, split_type, img_transforms, replayed_pairs=None):
        if dataset_name == "MSRVTT":
            if split_type == 'train':
                dataset = MSRVTTDataset(config, data, model, split_type, img_transforms, replayed_pairs)

                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=True, num_workers=config.num_workers, pin_memory = True), dataset
            else:
                dataset = MSRVTTDataset(config, data, model, split_type, img_transforms)
                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=config.num_workers, pin_memory = True)
        
        elif dataset_name == 'ACTNET':
            if split_type == 'train':
                dataset = ACTNETDataset(config, data, model, split_type, img_transforms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers, pin_memory = True ,drop_last=True), dataset
            else:
                dataset = ACTNETDataset(config, data, model, split_type, img_transforms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers, pin_memory = True ,drop_last=True)

        else:
            raise NotImplementedError
