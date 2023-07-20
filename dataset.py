from lib import *

class StandfordDogDataset(data.Dataset):
    def __init__(self, file_lst, labels_lst, transform=None, phase='train'):
        self.file_lst = file_lst
        self.labels_lst = labels_lst
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_lst)
    
    def __getitem__(self, idx):
        img_path = self.file_lst[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.labels_lst[idx]

        if self.transform is not None:
            img_transformed = self.transform(img, self.phase)

        return img_transformed, label