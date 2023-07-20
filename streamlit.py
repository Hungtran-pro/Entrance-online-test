import streamlit as st
from utils_streamlit import *


# Set up the app layout
st.set_page_config(page_title="Demo")
st.title("Training model on Stanford dataset")
st.info("Please check whether the dataset is available or not before training the model")

if st.button("Check dataset"):
    if check_dataset():
        st.warning("stanford-dogs-dataset not found! Please follow the instruction to get the dataset folder!", icon="⚠️")
    else:
        st.text("Dataset available")

# Set up the options selection section
option_selected = st.selectbox("Select a model to train", ["resnet50", "inception_v3", "efficientnet_b0", "mobilenet_v2", "mobilenet_v3_small"])

# Set up pretrained-model selection
if option_selected is not None:
    pretrained_model_number_lst = get_model_numbers(option_selected)
    if len(pretrained_model_number_lst) > 0:
        option_model = st.selectbox("Select a pretrained model", pretrained_model_number_lst)

# Set up the Predict button
if st.button("Train the model"):
    if check_dataset():
        st.text("Please check dataset!")

    batch_size = 16
    num_epochs = 2
    dataloader_dict = get_dataloader(model_name=option_selected, batch_size=batch_size)
    
    # network
    if option_model.strip() == "IMAGENET":
        model = get_model(model_name=option_selected, pretrained_param=True)
    else:
        model_number = int(option_model.split()[-1])
        model = load_pretrained_model(get_model(model_name=option_selected, pretrained_param=False), get_best_model_path(model_name = option_selected, model_number = model_number))
    
    # loss
    criterior = nn.CrossEntropyLoss()
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_name = "GPU" if device == "cuda:0" else "CPU"
    st.write("Device used [CPU/GPU]:", device_name)
    # set model number
    model_number = get_model_number(model.__class__.__name__)

    # callback
    early_stop = TorchEarlyStop(100)
    save_checkpoint = TorchModelCheckpoint("./model/best_{}_{}.pth".format(model.__class__.__name__, model_number))
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.8, patience=10, threshold=1e-5, cooldown=50, min_lr=1e-5, verbose=True)

    for epoch in range(num_epochs):
        st.write("Epoch {}/{}".format(epoch, num_epochs - 1))

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

            st.write("Phase: {} --- Loss: {:.4f} --- Accuracy: {:.4f}".format(phase, epoch_loss, epoch_accuracy))

            if phase == "val":
                # callbacks
                save_checkpoint.on_epoch_end(model, epoch_loss)
                if early_stop.on_epoch_end(model, epoch_loss):
                    early_stop_flag = True
                    
        lr_scheduler.step(epoch_loss)
                
        if early_stop_flag:
            st.write("No improvement! ---- Stop training!")
            break

    torch.save(model.state_dict(), "./model/pre_{}_{}.pth".format(model.__class__.__name__, model_number))