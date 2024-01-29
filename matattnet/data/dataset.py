import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        # Process the sample if needed
        # ...

        return sample
    
class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        super(MyDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )


if __name__=="__main__":
    # Create a dataset
    data = list(range(100))
    dataset = MyDataset(data)

    # Create a dataloader
    dataloader = MyDataLoader(dataset, batch_size=32, shuffle=True)

    # Iterate over the dataloader
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}: {batch}")
        # Do something with the batch
        # ...