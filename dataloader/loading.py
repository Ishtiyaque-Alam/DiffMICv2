import os, torch, csv
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import dataloader.transforms as trans
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# The 7 disease classes in HAM10000 dataset
HAM_CLASSES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']


class HAM10000DataSet(Dataset):
    def __init__(self, csv_file, data_dir, train=True):
        """
        Args:
            csv_file: path to GroundTruth.csv with columns:
                      image, MEL, NV, BCC, AKIEC, BKL, DF, VASC
            data_dir: path to directory containing the .jpg images.
            train: if True, apply data augmentation.
        """
        self.data_dir = data_dir
        image_names = []
        labels = []

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header
            for row in reader:
                image_id = row[0].strip()
                # Build 7-class one-hot vector from CSV columns
                label = [int(float(row[i])) for i in range(1, 8)]

                image_names.append(os.path.join(data_dir, image_id + '.jpg'))
                labels.append(label)

        self.image_names = image_names
        self.labels = labels

        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        if train:
            self.transform_center = transforms.Compose([
                transforms.RandomResizedCrop((224, 224)),
                trans.RandomHorizontalFlip(),
                trans.RandomRotation(20),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform_center = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        image = self.transform_center(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)
