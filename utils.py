from lib import *
from config import *
from callback import TorchEarlyStop, TorchModelCheckpoint

def get_model_number(model_name):
    if not os.path.exists("./model/best_{}_0.pth".format(model_name)):
        return 0
    else:
        model_number = 1
        while os.path.exists("./model/best_{}_{}.pth".format(model_name, model_number)):
            model_number += 1
        else:
          return model_number

def make_datapath_lst():
    '''
        Return a list of datapath for training and test dataset.
    '''
    dog_classes = os.listdir('./stanford-dogs-dataset/images/Images/')
    breeds = [breed.split('-',1)[1] for breed in dog_classes]
    full_paths = ['./stanford-dogs-dataset/images/Images/{}'.format(dog_class) for dog_class in dog_classes]

    full_datapaths, full_labels = [], []

    for idx, full_path in enumerate(full_paths):
        for img_name in os.listdir(full_path):
            full_datapaths.append(full_path + '/' + img_name)
            full_labels.append(breeds[idx])

    encoder = LabelEncoder()
    full_num_labels = encoder.fit_transform(full_labels)

    X_train, X_test, y_train, y_test = train_test_split(full_datapaths, full_num_labels, test_size=0.2, random_state=random_state)

    return X_train, X_test, y_train, y_test

def train_model(model, dataloader_dict, criterior, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # set model number
    model_number = get_model_number(model.__class__.__name__)

    # callback
    early_stop = TorchEarlyStop(100)
    save_checkpoint = TorchModelCheckpoint("./model/best_{}_{}.pth".format(model.__class__.__name__, model_number))
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.8, patience=10, threshold=1e-5, cooldown=50, min_lr=1e-5, verbose=True)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))

        # move network to device(GPU/CPU)
        model.to(device)
        epoch_loss = 0.0
        early_stop_flag = False
        torch.backends.cudnn.benchmark = True

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
                
            epoch_loss = 0.0
            epoch_corrects = 0
            
            if (epoch == 0) and (phase == "train"):
                continue
            for inputs, labels in tqdm(dataloader_dict[phase]):
                
                # move inputs, labels to CPU/GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # set gradient of optimizer to be zero
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = criterior(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item()*inputs.size(0)
                    epoch_corrects += torch.sum(preds==labels.data)
                    
            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))

            if phase == "val":
                # callbacks
                save_checkpoint.on_epoch_end(model, epoch_loss)
                if early_stop.on_epoch_end(model, epoch_loss):
                    early_stop_flag = True
                    
        lr_scheduler.step(epoch_loss)
                
        if early_stop_flag:
            print("No improvement! ---- Stop training!")
            break

    torch.save(model.state_dict(), "./model/pre_{}_{}.pth".format(model.__class__.__name__, model_number))

def train_model_kd(model_student, model_teacher, dataloader_dict, optimizer, criterior, criterior_distil, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    model_student.to(device)
    model_teacher.to(device)
    model_teacher.eval()
    
    # set model number
    model_student_number = get_model_number(model_student.__class__.__name__)

    # callback
    early_stop_flag = False
    early_stop = TorchEarlyStop(100)
    save_checkpoint = TorchModelCheckpoint("./model/best_distil_{}_{}.pth".format(model_student.__class__.__name__, model_student_number))
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.8, patience=10, threshold=1e-5, cooldown=50, min_lr=1e-5, verbose=True)
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        
        epoch_loss = 0.0
        early_stop_flag = False
        torch.backends.cudnn.benchmark = True

        for phase in ['train', 'val']:
            if phase == 'train':
                model_student.train()
            else:
                model_student.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloader_dict[phase]):
                
                # move inputs, labels to CPU/GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # set gradient of optimizer to be zero
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs_student = model_student(inputs)
                    loss_student = criterior(outputs_student, labels)
                
                    if phase == 'train':
                        outputs_teacher = model_teacher(inputs)
                        distil_loss = criterior_distil(outputs_student, loss_student.item() * inputs.size(0), outputs_teacher)
                        distil_loss.backward()
                        optimizer.step()
                
                    _, preds_student = torch.max(outputs_student, 1)

                    running_loss += loss_student.item() * inputs.size(0)
                    running_corrects += torch.sum(preds_student == labels.data)

            epoch_loss = running_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloader_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    
            if phase == "val":
                # callbacks
                save_checkpoint.on_epoch_end(model_student, epoch_loss)
                if early_stop.on_epoch_end(model_student, epoch_loss):
                    early_stop_flag = True
                        
        lr_scheduler.step(epoch_loss)
                    
        if early_stop_flag:
            print("No improvement! ---- Stop training!")
            break
    
    torch.save(model_student.state_dict(), "./model/pre_distil_{}_{}.pth".format(model_student.__class__.__name__, model_student_number))

def get_resize(model_name):
    if model_name in ["resnet50", "efficientnet_b0", "mobilenet_v2", "mobilenet_v3_small"]:
        return 224
    elif model_name in ["inception_v3"]:
        return 299

def get_num_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def get_best_model_path(model_number = 0):
    model_path = "./model/best_ResNet_{}.pth".format(model_number)
    
    if not os.path.exists(model_path):
        raise("Model teacher doesn't exist {}".format(model_path))
    
    return model_path

def load_pretrained_model(model, model_path="./model/best_ResNet_0.pth"):
    
    if not os.path.exists(model_path):
        raise("File doesn't exist {}".format(model_path))
    
    if torch.cuda.is_available():
        load_weights = torch.load(model_path)
    else:
        load_weights = torch.load(model_path,  map_location={"cuda:0": "cpu"})
        
    model.load_state_dict(load_weights)
    return model

def get_model(model_name="resnet50", pretrained_param=True):
    
    n_classes = 120
    
    if model_name == "resnet50":
        # 25557032
        model = models.resnet50(pretrained=pretrained_param)
        
        # Freeze early layers
        if pretrained_param:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
            
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, n_classes),
            # nn.LogSoftmax(dim=1)
            )

    elif model_name =="inception_v3":
        # 27161264
        model = models.inception_v3(pretrained=pretrained_param)
        
        # Freeze early layers
        if pretrained_param:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
            
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, n_classes),
            # nn.LogSoftmax(dim=1)
            )

    elif model_name == "efficientnet_b0":
        # 5288548
        model = models.efficientnet_b0(pretrained=pretrained_param)
        
        # Freeze early layers
        if pretrained_param:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
            
        n_inputs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(n_inputs, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, n_classes),
            # nn.LogSoftmax(dim=1)
            )

    elif model_name == "mobilenet_v2":
        # 3504872
        model = models.mobilenet_v2(pretrained=pretrained_param)
        
        # Freeze early layers
        if pretrained_param:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
            
        n_inputs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(n_inputs, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, n_classes),
            # nn.LogSoftmax(dim=1)
            )

    elif model_name == "mobilenet_v3_small":
        # 2542856
        model = models.mobilenet_v3_small(pretrained=pretrained_param)
        
        # Freeze early layers
        if pretrained_param:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
            
        n_inputs = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(n_inputs, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, n_classes),
            # nn.LogSoftmax(dim=1)
            )
    return model