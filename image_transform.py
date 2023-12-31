from lib import *

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(resize, scale=(0.5,1.0)),
                # transforms.RandomHorizontalFlip(),
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),

            'val': transforms.Compose([
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        }
    
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)
        