from torch.utils.data import Dataset
from PIL import Image


class LoadCIFAR100(Dataset):
    def __init__(self, images, labels, transformations):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.transformations = transformations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transformations(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return image, label
