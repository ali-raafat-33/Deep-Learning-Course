# CNN (Convolutional Neural Network) — Student Guide

## Purpose

This README explains Convolutional Neural Networks (CNNs) step by step, in the same order as the companion visual guide. Each section below links to its matching section in the visual guide — click a heading link to see the same concept illustrated with a worked example (grids, numbers, colors) instead of just text.

> **Companion visual guide:** [open the interactive CNN guide](https://b43c7146-95ee-4afd-85b9-24b22de6da0e.frame.claudeusercontent.com/_f/1790548838-ac6b/#top)

## Prerequisites

- **Python** basics (variables, loops, functions, classes).
- **NumPy** basics — arrays, shapes, element-wise operations.
- **General neural network concepts**: neuron, weight, activation function, loss function, backpropagation (a general understanding is enough).
- **PyTorch** (used in the code example) — basic familiarity with tensors and `nn.Module`.
- Python 3.9+. A GPU speeds up training but isn't required.

---

## 01. Before CNNs

Before CNNs, computers couldn't learn what to look for in an image on their own. Human experts had to hand-design rules and formulas — called "features" — to describe things like edges, corners, or color patterns, then feed those features into a classic machine learning model.

```
[Input] → [Human-designed feature extraction] → [ML model] → [Output]
```

This didn't scale: every new pattern needed a new hand-built rule.

## 02. Why CNNs

Instead of humans hand-crafting features, a CNN learns its own filters directly from data during training. This lets it automatically discover the best patterns to look for — something no hand-designed method could do at scale.

Images also have **spatial structure**: nearby pixels are related to each other. CNNs preserve these spatial relationships, while a traditional network would flatten the image into a long list of numbers and lose that structure.

## 03. What is a CNN

A Convolutional Neural Network (CNN) is a type of neural network built specifically to process grid-like data, such as images. It automatically learns to recognize patterns like edges, shapes, and objects — without being told what to look for.

| Component | Function |
|---|---|
| Convolution layer | Extracts local features using learnable filters |
| Activation (ReLU) | Introduces non-linearity |
| Pooling | Shrinks the feature map, keeps the strongest signals |
| Flatten | Converts 3D feature maps into a 1D vector |
| Fully connected | Combines features into the final decision |
| Loss function | Measures the error and drives training |

## 04. CNN vs. Traditional Networks

Traditional (fully connected) networks treat every pixel as a separate, independent input, which needs huge numbers of parameters and ignores spatial patterns. CNNs share the same small filter across the whole image, making them far more efficient and better at spotting a pattern no matter where it appears.

## 05. Pixels

A pixel is the smallest unit of a digital image — a single point holding a color or brightness value. An image is simply a grid made up of thousands of these pixels.

## 06. Image Dimensions

Every image has a Height and a Width, measured in pixels (for example, 224 × 224). These dimensions define the size of the grid the CNN will process.

## 07. Channels

A grayscale image has 1 channel, storing only brightness. A color image has 3 channels — Red, Green, and Blue — which combine to form every color you see.

## 08. Image Tensors

A CNN doesn't see a "picture" — it sees a 3D array of numbers shaped `Height × Width × Channels`. This numerical grid, called a **tensor**, is the actual input fed into the network.

## 09. Convolution

Convolution is the core operation of a CNN. A small filter slides across the image, multiplying its values with the pixels underneath and summing the result to detect a specific pattern at that location.

```
Output[0][0] = (9×0)+(4×2)+(1×1)+(1×1)+(1×0)+(1×1)+(1×2)+(2×0)+(1×1) = 16
```

## 10. Filters / Kernels

A filter (or kernel) is a small matrix of numbers — often 3×3 — that the network learns during training. Different filters learn to detect different features, such as edges, corners, or textures.

## 11. Feature Maps

As a filter slides across the whole image, it produces a **feature map**: a new grid showing where in the image that particular pattern was found, and how strongly.

## 12. Stride

Stride is how many pixels the filter moves at each step. A stride of 1 moves one pixel at a time (more detail, bigger output); a larger stride skips more pixels (less detail, smaller output).

## 13. Padding

Padding adds a border of zeros around the image before convolution. This lets the filter properly process edge pixels and gives control over the size of the output feature map.

## 14. Pooling

Pooling shrinks the size of feature maps while keeping the most important information. This reduces computation, helps prevent overfitting, and makes the model more robust to small shifts or distortions in the image.

## 15. Max Pooling

Max pooling looks at a small region of the feature map (e.g., 2×2) and keeps only the highest value — the strongest signal that a feature was detected there.

```
[2 2 7 3]        [9 7]
[9 4 6 1]   →     [8 6]
[8 5 2 4]
[3 1 2 6]
```

## 16. Average Pooling

Average pooling takes the average of the values in each region instead of the maximum, producing a smoother, more gradual downsampling of the feature map.

---
## Pipeline Walkthrough

1. **Data Loading** — read images from labeled folders.
2. **Preprocessing** — resize, convert to tensor, normalize.
3. **Augmentation (optional)** — flips, rotations, brightness jitter.
4. **Train/Test Split** — separate data for training vs. honest evaluation.
5. **Model Definition** — stack Conv + Pooling + Fully Connected layers.
6. **Training Loop** — forward pass → loss → backward pass → optimizer step, repeated per epoch.
7. **Evaluation** — measure accuracy on unseen test data.
8. **Inference** — predict on a brand-new image.

---

## 7. Code Explanation — PyTorch

The following example is a complete, real binary classification implementation (cat/dog) using a simple CNN in PyTorch, broken down and explained step by step.

### 7.1 Importing Libraries

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
```
**Here I imported** the core PyTorch library (`torch`), the neural network building package (`nn`), the optimization algorithms package such as Adam (`optim`), plus `torchvision` utilities for handling image data, and `DataLoader` for feeding data to the model in batches during training.

### 7.2 Fixing the Random Seed (Reproducibility)

```python
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```
**Here I fixed** the random seed for Python, NumPy, and PyTorch, so that every run of the code produces exactly the same results (the same data split, the same initial weights) — this is important for comparing experiments fairly.

### 7.3 Setting Up Transforms and Loading the Data

```python
IMAGE_SIZE = 64
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()  # Convert the image into a Tensor shaped [3, 64, 64]
])

dataset = datasets.ImageFolder(
    root="path/to/PetImages",
    transform=transform
)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))
```
**Here I defined** a processing pipeline that resizes every image to 64×64 pixels, then converts it into a numeric Tensor shaped `[Channels, Height, Width]`. I then used `ImageFolder`, which automatically reads images from subfolders (a `Cat` folder and a `Dog` folder) and assigns the label based on the folder name itself.

### 7.4 Splitting the Data and Building the DataLoaders

```python
train_size = int(0.90 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
```
**Here I split** the data into 90% training and 10% testing, then wrapped each part in a `DataLoader` that feeds the network batches of 32 images at a time. Note that `shuffle=True` is only enabled for the training data (to shuffle the image order every epoch), while the test data keeps a fixed order.

### 7.5 Selecting the Compute Device (CPU/GPU)

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)
```
**Here I checked** automatically whether a CUDA-compatible GPU is available, using it if found to significantly speed up training, or falling back to the regular CPU otherwise.

### 7.6 Defining the Model Architecture

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # The convolutional part: extracts features from the image
        self.cnn = nn.Sequential(
            # Input: 3 channels (RGB), Output: 16 filters
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Input: 16 feature maps, Output: 32 filters
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Input: 32 feature maps, Output: 64 filters
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # The classifier part: makes the final decision from the extracted features
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # [channels, height, width] => single vector
            nn.Linear(64 * 8 * 8, 128),    # Hidden layer
            nn.ReLU(),
            nn.Linear(128, 1)              # Output: cat or dog
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.classifier(x)
        return x
```
**Here I built** the network in two parts: a convolutional part (`self.cnn`) made of 3 identical blocks (Conv2d → ReLU → MaxPool2d), where the number of filters grows progressively (16 → 32 → 64) while the spatial dimensions shrink each time (because of Max Pooling with stride 2). Then a classifier part (`self.classifier`) that flattens the convolutional output and passes it through two Fully Connected layers to produce a single number representing the confidence that the image is a "dog."

> **Dimension note:** the image starts at 64×64. After each MaxPool2d(stride=2), each dimension halves: 64→32→16→8. That's why the final feature map is 8×8 with 64 channels, and hence `64 * 8 * 8` in the first `Linear` layer.

### 7.7 Setting Up the Loss Function and Optimizer

```python
model = SimpleCNN().to(device)

loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
**Here I moved** the model to the appropriate device (GPU/CPU), then chose `BCEWithLogitsLoss` as the loss function (suitable for binary classification where the output is a single raw value before Sigmoid), and used `Adam` as the optimizer with a learning rate of 0.001 to gradually update the network's weights.

### 7.8 Training Loop

```python
EPOCHS = 4

for epoch in range(EPOCHS):
    model.train()  # training mode (activates layers like Dropout, if present)

    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()        # clear gradients accumulated from the previous batch

        outputs = model(images)      # forward pass: compute the model's prediction
        loss = loss_function(outputs, labels)  # compute the loss

        loss.backward()              # backward pass: compute gradients
        optimizer.step()             # update the weights based on the gradients

        predictions = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}/{EPOCHS} | Training Accuracy: {accuracy:.2f}%")
```
**Here I repeated** the training process for 4 epochs. For each batch of images: I clear the old gradients, pass the images through the model to get predictions, compute how far these predictions are from the true labels (loss), then use `loss.backward()` to compute each weight's contribution to the error (via backpropagation), and finally `optimizer.step()` updates the weights to reduce that error next time. I also track training accuracy to monitor progress after each epoch.

### 7.9 Evaluating on the Test Set

```python
model.eval()  # evaluation mode (disables Dropout, freezes BatchNorm, if present)

correct = 0
total = 0

with torch.no_grad():  # no need to compute gradients during evaluation
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        predictions = (torch.sigmoid(outputs) >= 0.5).float()

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
```
**Here I switched** the model to evaluation mode (`model.eval()`) and disabled gradient computation (`torch.no_grad()`) since we don't need to update weights here — we're only measuring the model's performance on data it never saw during training, which is the real measure of its ability to generalize.

### 7.10 Using the Model to Predict a New Image

```python
from PIL import Image

image = Image.open("my_photo.jpg").convert("RGB")

# Same preprocessing used during training
image_tensor = transform(image).unsqueeze(0).to(device)  # add a batch dimension

model.eval()
with torch.no_grad():
    output = model(image_tensor)
    dog_probability = torch.sigmoid(output).item()

prediction = "Dog" if dog_probability >= 0.5 else "Cat"
confidence = dog_probability if dog_probability >= 0.5 else 1 - dog_probability

print(f"Prediction: {prediction} ({confidence:.2%})")
```
**Here I applied** the exact same preprocessing steps (`transform`) used during training to a brand-new image, then added an extra dimension (`unsqueeze(0)`) since the model expects a batch of images rather than a single one. I then converted the model's raw output into a probability via `sigmoid`, and made the final classification decision based on the 0.5 threshold.

---

## 8. Practical Tips

- **Increase epochs gradually:** start with a small number (e.g., 3–4) to confirm the code runs correctly, then increase it and watch the accuracy/loss curve.
- **Watch the gap between training and test accuracy:** if training accuracy is very high while test accuracy is low, that's a sign of **overfitting** (the network memorizing the data instead of learning general patterns). Common fixes: adding Dropout, reducing model size, or using data augmentation.
- **Choosing the learning rate:** too high makes training unstable (loss oscillates or diverges), too low makes training extremely slow. `0.001` with Adam is a common, solid starting point.
- **Make sure the Flatten dimensions match:** the most common beginner mistake is miscalculating the input size of the first `Linear` layer after `Flatten`. Always compute it based on the original image size and how many times pooling was applied.
- **Class balance:** if the number of cat images differs drastically from the number of dog images, the model may become biased toward the more frequent class. Check data balance before training.
- **Use a GPU when available:** the training speed difference between CPU and GPU can be tens of times faster, especially with larger images or more data.
- **Always fix the random seed** when experimenting with changes to the model, so comparisons between experiments stay fair.

---

## 9. Running Instructions

### Dependencies

```bash
pip install torch torchvision numpy pillow matplotlib
```

### Steps to Run

1. Prepare a data folder in the following structure (required automatically by `ImageFolder`):
```
PetImages/
├── Cat/
│   ├── image1.jpg
│   └── ...
└── Dog/
    ├── image1.jpg
    └── ...
```
2. Update the `root` value in the code to point to your local `PetImages` folder path.
3. Run the code in the order shown in Section 7 (imports → data loading → model definition → training → evaluation).
4. Training accuracy will print at the end of each epoch, and the final test accuracy will print after training completes.

### Expected Output

```
Using: cuda
Classes: ['Cat', 'Dog']
Total images: ...
Epoch 1/4 | Training Accuracy: ~XX%
Epoch 2/4 | Training Accuracy: ~XX%
Epoch 3/4 | Training Accuracy: ~XX%
Epoch 4/4 | Training Accuracy: ~XX%
Test Accuracy: ~XX%
```
(Actual values depend on the size and quality of your dataset.)

---

## 10. References and Further Reading

- Official PyTorch tutorials: [pytorch.org/tutorials](https://pytorch.org/tutorials/)
- Official `torchvision.datasets.ImageFolder` documentation on the PyTorch website.
- The original LeNet-5 paper — Yann LeCun et al. (1998), the foundation of modern CNN architectures.
- The AlexNet paper — Krizhevsky, Sutskever, Hinton (2012), the turning point of the ImageNet competition.
- The VGGNet paper — Simonyan & Zisserman (2014), the principle of small, repeated filters.
- Stanford's CS231n course — "Convolutional Neural Networks for Visual Recognition," one of the best free, in-depth resources on CNNs.
