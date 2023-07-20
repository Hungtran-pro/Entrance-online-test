## REQUIRED LIBRARY

```
Pytorch
Sklearn
Numpy
Streamlit
```

## DATASET REQUIREMENT

Please download **[stanford-dogs-dataset.zip](https://drive.google.com/file/d/1sAXc9_XvRo2HtzPbfRngfrViMRXNXJfl/view?usp=sharing)** then unzip to get "stanford-dogs-dataset" folder

## PRETRAINED MODEL

Due to Github's policy, I couldn't upload weight of pretrained models.
So please follow **[folder here]()** to access all pretrained weights

## ENJOY YOURSELF

After following the aforementioned steps, your folder should be looked like:

.
├── model
│   ├── 
│   └── 
├── stanford-dogs-dataset
│   ├── annotations
│   └── images
│   
├── app.py
├── callback.py
├── ...
│── ...
│── ...
└── utils.py

2 fodlers, 16 files (including 1 zip-file)

### Running instructions

**0. Train model without distillation technique **

```sh
python finetune.py
```

**1. Train model with distillation technique **

```sh
python distillation.py
```

**2. Run streamlit **

```sh
streamlit run streamlit.py
```

## Solution:

**I. Build a dog breed classification**

**2.a:**
Train model (from pretrained-IMAGENET/ from scratch)

Please kindly change below variables in finetune.py as your purpose.

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
- TorchModelCheckpoint to make sure that model can keep training from its best previous version even when being interupted.
- TorchEarlyStop to make sure model stop running when there is no improvement.

**2.c:**
As using ResNet50 pretrained from IMAGENET dataset as a baseline.

Result achievement:

| Loss  | Accuracy | Accuracy |
| ------------- | ------------- | ------------- |
| NaN  | 0.7075  | 0.7075  |

**3:**

To train a model which is 20% parameters less than the base model (Resnet50).
I decide to choose well-known models such as (efficientnet_b0, mobilenet_v2, mobilenet_v3_small) & apply *Response-based knowledge* in Knowledge Distillation.

To desmonstrate my approach, pretrained parameters from IMAGENET is not used.

| Model  | Loss | Accuracy |
| ------------- | ------------- | ------------- |
| ResNet50  | NaN  | 0.7075  |
| EfficientNet  |  NaN | NaN  |
| MobileNetV2  |  NaN |  NaN |

-----
Due to the lack of resource, there are some limitation in my work.

Hopefully you accept my approach.
Thank you!