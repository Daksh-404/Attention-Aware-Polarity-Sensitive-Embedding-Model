import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from PIL import ImageOps as IOP


class TripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, filepath, mode,transform):
        self.filepath=filepath
        self.train = mode
        self.transform=transform
        if self.train=="train":
            train_img_names=[]
            train_img_labels=[]
            with open(filepath,'r') as file:
                for line in file.readlines():
                    lst=line.split()
                    train_img_names.append(lst[0])
                    train_img_labels.append(lst[1])

            self.train_labels = train_img_labels
            self.train_data = np.array(train_img_names)
            self.labels_set = set(np.array(self.train_labels))
            self.label_to_indices = {label: np.where(np.array(self.train_labels) == label)[0]
                                     for label in self.labels_set}
            print("1")

        elif self.train=="valid":
            valid_img_names=[]
            valid_img_labels=[]
            with open(filepath,'r') as file:
                for line in file.readlines():
                    lst=line.split()
                    valid_img_names.append(lst[0])
                    valid_img_labels.append(lst[1])

            self.valid_labels = valid_img_labels
            self.valid_data = np.array(valid_img_names)
            self.labels_set = set(np.array(self.valid_labels))
            self.label_to_indices = {label: np.where(np.array(self.valid_labels) == label)[0]
                                     for label in self.labels_set}
        else:
            test_img_names=[]
            test_img_labels=[]
            with open(filepath,'r') as file:
                for line in file.readlines():
                    lst=line.split()
                    test_img_names.append(lst[0])
                    test_img_labels.append(lst[1])


            self.test_labels = test_img_labels
            self.test_data = np.array(test_img_names)
            # generate fixed triplets for testing
            self.labels_set = set(np.array(self.test_labels))
            self.label_to_indices = {label: np.where(np.array(self.test_labels) == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets
    def __getitem__(self, index):
        if self.train=="train":
            img1, label1 = self.train_data[index], self.train_labels[index]#.item()
            root ='D:/dataScience/ASEP/Fi/fi/'
            img1= Image.open(root + img1).convert('RGB')
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
            img2 = Image.open(root + img2).convert('RGB')
            img3 = Image.open(root + img3).convert('RGB')

        elif self.train=="valid":
            img1, label1 = self.valid_data[index], self.valid_labels[index]#.item()
            root ='D:/dataScience/ASEP/Fi/fi/'
            img1= Image.open(root + img1).convert('RGB')
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])

            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
            img2 = Image.open(root + img2).convert('RGB')
            img3 = Image.open(root + img3).convert('RGB')

        else:
            root = 'D:/dataScience/ASEP/Fi/fi/'
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]
            img1 = Image.open(root + img1).convert('RGB')
            img2 = Image.open(root + img2).convert('RGB')
            img3 = Image.open(root + img3).convert('RGB')

        img1=img1.resize((256,256))
        img2=img2.resize((256,256))
        img3=img3.resize((256,256))

        # Crop the center of the image
        """border=((16,16,16,16))
        img1 = IOP.crop(img1,border)
        img1 = IOP.crop(img2,border)
        img3 = IOP.crop(img3,border)"""

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return (img1, img2, img3), []

    def __len__(self):
        if self.train=="train":
            return len(self.train_data)
        elif self.train=="valid":
            return len(self.valid_data)
        else:
            return len(self.test_data)
