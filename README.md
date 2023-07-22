## REQUIRED LIBRARY

```
Pytorch
Sklearn
Numpy
Streamlit
```

## DATASET REQUIREMENT

Please download **[stanford-dogs-dataset.zip](https://drive.google.com/file/d/1sAXc9_XvRo2HtzPbfRngfrViMRXNXJfl/view?usp=sharing)** or via *Kaggle stanford-dogs-dataset dataset* then unzip to get "stanford-dogs-dataset" folder

## ENJOY YOURSELF

After following the aforementioned steps, your folder should be looked like this:

### Folder Structure

- model
  - *[list files of model]*
- stanford-dogs-dataset
  - annotations
  - images
- app.py
- callback.py
- ...
- utils.py

2 folders, 16 files (including 1 zip file - stanford-dogs-dataset.zip)

## Solution:

**I. Build a dog breed classification**

**2.a:**

Train model (from pre-trained IMAGENET dataset/ from scratch)

Please kindly change the below variables in finetune.py as your purpose.

```
pretrained_param = True
model_name = "resnet50"
batch_size = 16
num_epochs = 10
```

Then run
```sh
python finetune.py
```

**2.b:**

I use 2 TorchCallback:
- TorchModelCheckpoint to make sure that model can keep training from its best previous version even when being interrupted.
- TorchEarlyStop to make sure the model stops running when there is no improvement.

**2.c:**

As using ResNet50 pretrained from the IMAGENET dataset as a baseline.

Result achievement:

| Model  | Loss | Accuracy |
| ------------- | ------------- | ------------- |
| ResNet50  | NaN  | 0.7075  |

**3:**

```sh
python distillation.py
```

To train a model which is 20% parameters less than the base model (Resnet50).

I decide to choose well-known models such as (efficientnet_b0, mobilenet_v2, and mobilenet_v3_small) & apply *Response-based knowledge* in Knowledge Distillation.

| Model  | Loss | Accuracy |
| ------------- | ------------- | ------------- |
| EfficientNet  |  11.9173 | 0.3785  |

**II. Build a UI for the model using Streamlit**

```sh
streamlit run streamlit.py
```

-----
Due to the lack of resources, there are some limitations in my work.

I was only able to train EfficientNet for 10 epochs and did not achieve a very high accuracy

Hopefully, you accept my approach.

Thank you!
