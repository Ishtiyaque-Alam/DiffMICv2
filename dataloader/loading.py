import os, torch, csv
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import dataloader.transforms as trans
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# The 14 disease classes in NIH Chest X-ray dataset
CHEST_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]


class ChestXrayDataSet(Dataset):
    def __init__(self, csv_file, data_dir, train=True):
        """
        Args:
            csv_file: path to CSV file with columns:
                      Image Index, Finding Labels, ...
            data_dir: path to directory containing the images.
            train: if True, apply data augmentation.
        """
        self.data_dir = data_dir
        image_names = []
        labels = []

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                image_name = row[0].strip()
                finding_labels = row[1].strip()

                # Build 14-class one-hot vector
                label = [0] * len(CHEST_CLASSES)
                if finding_labels != 'No Finding':
                    for disease in finding_labels.split('|'):
                        disease = disease.strip()
                        if disease in CHEST_CLASSES:
                            label[CHEST_CLASSES.index(disease)] = 1

                # 15th element: 1 if No Finding, 0 otherwise
                label.append(1 if (np.array(label) == 0).all() else 0)

                image_names.append(os.path.join(data_dir, image_name))
                labels.append(label)

        self.image_names = image_names
        self.labels = labels

        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        if train:
            self.transform_center = transforms.Compose([
                transforms.RandomResizedCrop(224),
                trans.RandomHorizontalFlip(),
                trans.RandomRotation(20),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform_center = transforms.Compose([
                transforms.Resize(224),
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

