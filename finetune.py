from lib import *
from config import *
from image_transform import ImageTransform
from utils import make_datapath_lst, train_model, train_model_kd, get_resize, get_model
from dataset import StandfordDogDataset

def main(model_name, pretrained_param, batch_size, num_epochs):

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

    # network
    model = get_model(model_name, pretrained_param)
    
    # loss
    criterior = nn.CrossEntropyLoss()
    # criterior = nn.NLLLoss()
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # training
    train_model(model, dataloader_dict, criterior, optimizer, num_epochs)

if __name__ == "__main__":
    
    pretrained_param = True
    model_name = "resnet50"
    batch_size = 16
    num_epochs = 10
    
    main(model_name, pretrained_param, batch_size, num_epochs)