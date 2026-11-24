from collections import defaultdict
from pathlib import Path
import random
from PIL import Image
from numpy import negative

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision import tv_tensors


class SiameseDataset(Dataset):
    def __init__(self, 
                 path, 
                 std, 
                 mean, 
                 img_size, 
                 number_of_sample,
                 pair_type, 
                 enable_transform = False,
                 train_type = "train") -> None:
        self.path = path
        self.std = std
        self.mean = mean
        self.img_size = img_size
        self.number_of_sample = number_of_sample
        self.pair_type = pair_type
        self.enable_transform = enable_transform
        self.train_type = train_type
        self.transform = self.train_transform() if self.train_type == "train" else self.val_transform()


        self.identities = self.create_identities()
        self.pairs = self.create_pairs()
    #Dùng để tạo ra một defaultdict chứa các identity
    def create_identities(self):
        identities = defaultdict(list)
        accepted_extensions = [".png", ".jpg", ".jpeg"]
        files = [i for i in Path(self.path).rglob("*") if i.is_file() and i.suffix.lower() in accepted_extensions]

        for i in files:
            identities[i.parent.name].append(i)

        identities = {
            k: v for k, v in identities.items()
            if len(v) > 1
        }
        return identities

    #Dùng để tạo ra các pair, có thể tùy chỉnh (chưa phải là pair chính thức dùng để train)
    def create_pairs(self):
        match self.pair_type:
            case 1: #Contrastive Loss
                positive_sample = self.number_of_sample // 2
                negative_sample = self.number_of_sample - positive_sample
                positive_pairs, negative_pairs = [], []

                keys = list(self.identities.keys())
                # Generate positive pairs
                for _ in range(positive_sample):
                    key = random.choice(keys)
                    image1, image2 = random.sample(self.identities[key], 2)

                    positive_pairs.append((0, image1, image2)) #0 là positive

                # Generate negative pairs
                for _ in range(negative_sample):
                    key1, key2 = random.sample(keys, 2)

                    image1 = random.choice(self.identities[key1])
                    image2 = random.choice(self.identities[key2])

                    negative_pairs.append((1, image1, image2)) #1 là negative

                pairs = positive_pairs + negative_pairs
                random.shuffle(pairs)
                return pairs
            case 2: #Triplet Loss
                triplets = set()
                keys = list(self.identities.keys())

                while len(triplets) < self.number_of_sample:

                    anchor_key = random.choice(keys)

                    anchor, positive = random.sample(
                        self.identities[anchor_key], 2
                    )

                    negative_key = random.choice(
                        [k for k in keys if k != anchor_key]
                    )

                    negative = random.choice(
                        self.identities[negative_key]
                    )

                    triplets.add((anchor, positive, negative))
                return list(triplets)
            case _:
                raise ValueError(
                    f"Invalid pair_type: {self.pair_type}. Expected 1 or 2."
                )


    def train_transform(self):
        if self.enable_transform == False:
            return v2.Compose([
                v2.Resize(self.img_size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=self.mean,
                    std=self.std
                )
            ])
        return v2.Compose([
            v2.Resize(self.img_size),
            v2.RandomChoice(
                [
                    v2.RandomHorizontalFlip(),
                    v2.RandomRotation(degrees=(-15,15)),
                    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                    v2.Lambda(lambda x: x),
                ]
            ),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=self.mean,
                std=self.std
            )
        ])
    def val_transform(self):
        return v2.Compose([
            v2.Resize(self.img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=self.mean,
                std=self.std
            )
        ])

    def regenerate_pairs(self):
        self.pairs = self.create_pairs()
    
    def __len__(self):
        return self.number_of_sample
    
    def __getitem__(self, idx):
        if self.pair_type == 1: #Contrastive Loss
            label, image1, image2 = self.pairs[idx]
            img1 = Image.open(image1).convert("RGB")
            img2 = Image.open(image2).convert("RGB")
            img1, img2 = self.transform(img1), self.transform(img2)
            return tv_tensors.Image(img1), tv_tensors.Image(img2), torch.tensor(label, dtype=torch.float32)
        else:
            anchor_img, positive_img, negative_img = self.pairs[idx]
            anchor_img = Image.open(anchor_img).convert("RGB")
            positive_img = Image.open(positive_img).convert("RGB")
            negative_img = Image.open(negative_img).convert("RGB")
            anchor_img, positive_img, negative_img = self.transform(anchor_img), self.transform(positive_img), self.transform(negative_img)
            return tv_tensors.Image(anchor_img), tv_tensors.Image(positive_img), tv_tensors.Image(negative_img)

class DatasetLoader():
    def __init__(self, 
                 path, 
                 std, 
                 mean, 
                 img_size, 
                 batch_size, 
                 number_of_sample,
                 pair_type=1,  
                 enable_transform = False) -> None:
        """
        Load dataset for SNN model, this can be use for both Contrastive Loss and Triplet Loss

        Args:
            std (list): Array containing the std of the dataset. \n
            mean (list): Array containing the mean of the dataset. \n
            img_size (list): Array containing the desire dimension for the image. \n
            batch_size (int): The number of batch size. \n
            number_of_sample (int): The number of sample that will be generate from the dataset. \n 
            pair_type (int): The type of pair to be generate, 1 is use Contrastive loss and 2 if use Triplet Loss. \n
            enable_transform (boolean): Set to true if you want the training set to be augmentate. \n
        """
        
        self.path = path
        self.std = std
        self.mean = mean
        self.img_size = img_size
        self.batch_size = batch_size
        self.number_of_sample = number_of_sample
        self.pair_type = pair_type
        self.enable_transform = enable_transform


    def dataset_loader(self, type):
        if type == "train":
            self.training_dataset = SiameseDataset(
                path=self.path,
                pair_type=self.pair_type,
                enable_transform=self.enable_transform,
                img_size=self.img_size,
                mean=self.mean,
                std=self.std,
                number_of_sample=self.number_of_sample,
                train_type="train"
            )
            print("Total train image: {0}".format(len(self.training_dataset)))
            loader = DataLoader(
                self.training_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,          # START HERE
                pin_memory=True,
                persistent_workers=False, #Chỉnh cái này thành False để tránh hết Ram
                prefetch_factor=2
            )
        else:
            self.val_dataset = SiameseDataset(
                path=self.path,
                pair_type=self.pair_type,
                enable_transform=False,
                img_size=self.img_size,
                mean=self.mean,
                std=self.std,
                number_of_sample=self.number_of_sample,
                train_type="val"
            )
            print("Total test image: {0}".format(len(self.val_dataset)))
            loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                persistent_workers=False #Chỉnh cái này thành False để tránh hết Ram
            )
        print()
        return loader
    def regenerate_pairs(self):
        self.training_dataset.regenerate_pairs()