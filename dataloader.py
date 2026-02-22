from torch.utils.data import Dataset, DataLoader
import kagglehub

from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path

path = kagglehub.dataset_download("msambare/fer2013")

data_dir = Path(path)

class EmotionDataset(Dataset):
    def __init__(self, set_name = "train", transform_pipeline = "default"):
        self.transform = self.get_transform(transform_pipeline)
        self.dataset = ImageFolder(data_dir / set_name, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_transform(self, transform_pipeline):
        xform = None

        if transform_pipeline == "default":
            xform = transforms.ToTensor()
        elif transform_pipeline == "augmented":
            xform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

        if xform is None:
            raise ValueError("Invalid transform pipeline")

        return xform

def load_data(set_name = "train", transform_pipeline = "default", num_workers = 2, batch_size = 8):

    dataset = EmotionDataset(set_name, transform_pipeline)

    shuffle = (set_name == "train")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

