# Understanding Convolutional Neural Networks (CNNs) — A Practical Student Guide

## 1. Title and Purpose

**Goal of this guide:** to give university students a clear, structured understanding of Convolutional Neural Networks (CNNs) — starting from *why* we need CNNs at all, moving through every core component (Convolution, Pooling, Stride, Padding) both mathematically and visually, and ending with a fully documented, line-by-line practical example that trains a real image classifier (cats vs. dogs) in PyTorch.

By the end of this guide you will be able to:
- Explain the difference between traditional (Fully Connected) networks and CNNs, and why CNNs are better suited to images.
- Understand how a Kernel/Filter works and what a Feature Map represents.
- Understand how Stride and Padding affect the output size.
- Understand the types of Pooling (Max / Average) and why we use them.
- Read and understand a complete, real PyTorch training pipeline for building, training, and evaluating a CNN.

---

## 2. Prerequisites

Before starting, you should ideally have:

- **Python programming** basics (variables, loops, functions, classes).
- **NumPy** basics — arrays, shapes, and element-wise operations.
- **Basic neural network concepts**: what a neuron, weights, activation function, loss function, and backpropagation are (a general understanding is enough — mastery is not required).
- **PyTorch** (used in the practical example below) or TensorFlow — basic awareness that these libraries build and train neural networks with automatic differentiation.
- A Python 3.9+ environment. A GPU is optional but speeds up training significantly.

---

## 3. Conceptual Overview

### Before CNNs: Manual Feature Engineering

In the past, human experts had to hand-design rules and formulas ("features") to describe edges, corners, and color patterns in an image, which were then fed into a traditional machine learning model. This approach is slow, limited, and doesn't scale well with large, complex datasets.

### With CNNs: Learning Features Automatically

Instead of humans hand-crafting filters, a CNN **learns its own filters directly from the data during training**. This lets it automatically discover the best patterns to detect — something that's extremely hard to achieve manually at scale.

### Why CNNs Specifically for Images?

Images have **spatial structure**: nearby pixels are related to each other (an edge, a color region, a texture...). A traditional Fully Connected network flattens the image into a long list of numbers and loses that structure, while a CNN preserves spatial relationships by sliding a small filter over local regions of the image, instead of connecting every single pixel to every neuron.

**Another key benefit:** the same filter is shared (reused) across the entire image, which drastically reduces the number of parameters compared to a Fully Connected network, and lets the network detect a pattern regardless of where it appears in the image.

### Core Components of a CNN

| Component | Function |
|---|---|
| **Convolution Layer** | Extracts local features (edges, corners, textures) using learnable filters |
| **Activation Function (usually ReLU)** | Introduces non-linearity so the network can learn complex patterns |
| **Pooling Layer** | Shrinks the feature map while keeping the most important information, reducing computation |
| **Flatten Layer** | Converts the 3D output of the convolutional layers into a 1D vector for the Fully Connected layer |
| **Fully Connected Layer** | Combines the extracted features and produces the final decision (classification) |
| **Loss Function** | Measures how far the network's prediction is from the true label, and drives training |

**How an image is represented internally:** an image is represented as a 3D **Tensor** shaped `Height × Width × Channels`. A grayscale image has 1 channel (brightness), while a color RGB image has 3 channels (Red, Green, Blue) that combine to form every color we see.

---

## 4. Architecture Blocks

Below is a text-based structural description of the most well-known classic CNN architectures, and the role each part plays:

### LeNet-5 (the simplest classic architecture)

```
[Input Image] → [Conv + Activation] → [Pooling] → [Conv + Activation] → [Pooling]
             → [Flatten] → [Fully Connected] → [Fully Connected] → [Classification Output]
```
- Only two convolutional layers, each followed by a pooling layer to gradually reduce spatial dimensions.
- Ends with small Fully Connected layers that produce the final decision.
- Originally designed for recognizing handwritten digits.

### AlexNet (the 2012 turning point)

```
[Input Image] → [Large Conv + ReLU] → [Pooling] → [Conv + ReLU] → [Pooling]
             → [Conv + ReLU] × 3 → [Pooling] → [Flatten]
             → [FC + Dropout] × 2 → [Classification Output]
```
- Much deeper than LeNet (5 convolutional layers).
- Used ReLU instead of slower activation functions, and Dropout to reduce overfitting.
- The first architecture to prove that deep networks could be trained practically on massive datasets (ImageNet) using GPUs.

### VGG (simplicity and consistent depth)

```
[Input Image] → [Conv 3×3 + ReLU] ×2 → [Pooling]
             → [Conv 3×3 + ReLU] ×2 → [Pooling]
             → [Conv 3×3 + ReLU] ×3 → [Pooling]  (this block pattern repeats)
             → [Flatten] → [FC] ×3 → [Classification Output]
```
- Relies entirely on small, repeated 3×3 filters, instead of varying larger filter sizes.
- Its large depth (16 or 19 layers) is what gives it strong representational power, at the cost of higher computation.

> **Note:** The practical example in this guide (Section 7) uses a simpler architecture than these three (just 3 blocks of Conv+ReLU+Pooling), which is entirely sufficient for a simple binary classification problem like "cat or dog," and illustrates the exact same principles without the computational complexity of the larger architectures.

---

## 5. Illustrated Explanations

> The figures below are described in text (no actual drawing) so you can sketch them yourself or picture them clearly while reading.

**Figure 1 — The Filter (Kernel) and Convolution**
*Caption: "A 3×3 filter slides across the input image, multiplies its values with the pixels underneath, then sums the results into a single cell in the output map."*
Picture a 5×5 input matrix, with a 3×3 window (the filter) moving across it starting from the top-left. At each position, the filter's values are multiplied element-wise with the corresponding image values, then the nine products are summed into one number written into the output feature map.

**Figure 2 — The Feature Map**
*Caption: "Each different filter produces a different feature map: one filter detects horizontal edges, another detects vertical edges, and so on."*
When a single filter slides across the whole image, it produces a new grid (map) showing where in the image the pattern it's looking for was found, and how strongly (a higher value = a stronger match).

**Figure 3 — Stride**
*Caption: "Stride=1 moves the filter one pixel at a time (more detail, larger output); Stride=2 skips two pixels at a time (less detail, roughly half the output size)."*
Picture the same 5×5 image with a 3×3 filter: with Stride=1 we get a 3×3 output map, while with Stride=2 we get a smaller map (roughly 2×2) because the filter jumps more positions each step.

**Figure 4 — Padding**
*Caption: "Adding a border of zeros around the original image before convolution, so that edge pixels are processed the same number of times as center pixels."*
Without padding, edge pixels participate in fewer convolution operations than central pixels, losing some information. Adding a border of zeros (zero-padding) solves this, and also gives precise control over the output feature map size (e.g., keeping it the same size as the input — "Same Padding").

**Figure 5 — Max Pooling**
*Caption: "A 2×2 window slides over the feature map and keeps only the highest value in each region, shrinking the map by half while preserving the strongest signals."*
Example: a 2×2 region containing the values (2, 2, 9, 4) → Max Pooling keeps only the value 9 and discards the rest. This reduces size and reduces sensitivity to small shifts in the image.

---

## 6. Pipeline Walkthrough (Data → Prediction)

The full practical pipeline for any image classification project with a CNN goes through these stages:

1. **Data Loading:** reading images from labeled folders (e.g., a `Cat` folder and a `Dog` folder), automatically linking each image to its label based on the folder name.
2. **Preprocessing:** standardizing the dimensions of every image (e.g., 64×64), converting them from an image format into a numeric Tensor the network understands, usually normalizing pixel values between 0 and 1.
3. **Augmentation (optional):** generating modified copies of the original images (horizontal flip, slight rotation, brightness change) to increase training data diversity and reduce overfitting. Not used in the practical example below to keep it simple, but it's a common and useful addition.
4. **Train/Test Split:** splitting the data into a training set (to learn the weights) and a test set (to evaluate performance on data never seen during training).
5. **Model Definition:** building the network architecture (Conv + Pooling + Fully Connected layers), as detailed in the next section.
6. **Training Loop:** for each batch of images — a forward pass computes the prediction, the loss is calculated, a backward pass computes the gradients, and the optimizer updates the weights. These steps repeat for several "epochs."
7. **Evaluation:** running the model on the test set (without updating weights) to measure real performance on unseen data.
8. **Inference:** using the trained model to predict a single new image that wasn't part of the original dataset at all.

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
