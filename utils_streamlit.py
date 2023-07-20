import os
from dataset import StandfordDogDataset
from image_transform import ImageTransform
from utils import make_datapath_lst, get_resize, get_model, load_pretrained_model
from config import mean, std, random_state
from callback import TorchEarlyStop, TorchModelCheckpoint
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

MODEL_DICT = {"resnet50" : "ResNet",
              "inception_v3": "Inception3",
              "efficientnet_b0": "EfficientNet",
              "mobilenet_v2": "MobileNetV2",
              "mobilenet_v3_small": "MobileNetV3",
              }

def get_best_model_path(model_name, model_number = 0):
    model_path = "./model/best_{}_{}.pth".format(MODEL_DICT[model_name], model_number)
    print(model_path)
    if not os.path.exists(model_path):
        raise("Model teacher doesn't exist {}".format(model_path))
    
    return model_path

def check_dataset():
    if not os.path.exists("./stanford-dogs-dataset"):
        return True
    else:
        return False
        
def get_model_numbers(model_name):
    number_lst = ["IMAGENET"]
    model_number = 0
    while os.path.exists("./model/best_{}_{}.pth".format(MODEL_DICT[model_name], model_number)):
        number_lst.append("Model " + str(model_number))
        model_number += 1
    else:
        return number_lst

def get_dataloader(model_name, batch_size = 16):
    # data path & label
    X_train, X_test, y_train, y_test = make_datapath_lst()

    # dataset
    resize = get_resize(model_name)
    train_dataset = StandfordDogDataset(X_train, y_train, transform=ImageTransform(resize, mean, std), phase="train")
    val_dataset = StandfordDogDataset(X_test, y_test, transform=ImageTransform(resize, mean, std), phase="val")

    # dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)
    dataloader_dict = {"train":train_dataloader, "val":val_dataloader}
    
    return dataloader_dict

def get_model_number(model_name):
    if not os.path.exists("./model/best_{}_0.pth".format(model_name)):
        return 0
    else:
        model_number = 1
        while os.path.exists("./model/best_{}_{}.pth".format(model_name, model_number)):
            model_number += 1
        else:
          return model_number
      
      