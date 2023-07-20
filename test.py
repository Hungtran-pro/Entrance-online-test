from config import *
from lib import *
from utils import make_datapath_lst, get_model, get_model_number
from dataset import StandfordDogDataset
from image_transform import ImageTransform
from torchvision.models import resnet50, inception_v3, efficientnet_b0, mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large

model = resnet50(pretrained = False)
print(model.__class__.__name__)
print(get_model_number(model.__class__.__name__))