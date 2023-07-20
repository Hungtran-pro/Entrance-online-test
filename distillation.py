from lib import *
from config import *
from image_transform import ImageTransform
from utils import make_datapath_lst, train_model, train_model_kd, get_resize, get_model, load_pretrained_model, get_best_model_path
from dataset import StandfordDogDataset

class DistillationLoss:
  def __init__(self):
    self.student_loss = nn.CrossEntropyLoss()
    self.distillation_loss = nn.KLDivLoss()
    self.temperature = 1
    self.alpha = 0.25

  def __call__(self, student_logits, student_target_loss, teacher_logits):
    distillation_loss = self.distillation_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                                F.softmax(teacher_logits / self.temperature, dim=1))

    loss = (1 - self.alpha) * student_target_loss + self.alpha * distillation_loss
    return loss

def main(model_student_name, model_teacher_name, pretrained_param, batch_size, num_epochs, model_number = 0):

    # data path & label
    X_train, X_test, y_train, y_test = make_datapath_lst()

    # dataset
    resize = get_resize(model_student_name)
    train_dataset = StandfordDogDataset(X_train, y_train, transform=ImageTransform(resize, mean, std), phase="train")
    val_dataset = StandfordDogDataset(X_test, y_test, transform=ImageTransform(resize, mean, std), phase="val")

    # dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)
    dataloader_dict = {"train":train_dataloader, "val":val_dataloader}

    # network
    model_teacher = load_pretrained_model(get_model(model_name=model_teacher_name, pretrained_param=False), get_best_model_path(model_number = 0))
    model_student = get_model(model_student_name, pretrained_param)
    
    # loss
    criterior = nn.CrossEntropyLoss()
    criterior_distil = DistillationLoss()
    # criterior = nn.NLLLoss()
    
    # optimizer
    optimizer = optim.Adam(model_student.parameters(), lr=0.0005)

    # training
    train_model_kd(model_student, model_teacher, dataloader_dict, optimizer, criterior, criterior_distil, num_epochs)

if __name__ == "__main__":
    
    pretrained_param = True
    model_student_name = "efficientnet_b0"
    model_teacher_name = "resnet50"
    batch_size = 16
    num_epochs = 20
    model_number = 0
    
    main(model_student_name, model_teacher_name, pretrained_param, batch_size, num_epochs, model_number)