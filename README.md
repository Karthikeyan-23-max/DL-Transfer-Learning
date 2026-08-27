# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Develop an image classification model using transfer learning with the pre-trained VGG19 model. 


## Neural Network Model
<img width="987" height="792" alt="image" src="https://github.com/user-attachments/assets/1e412a42-ced2-4b84-87ca-f2fd46388fed" />

## DESIGN STEPS
### STEP 1:
Load and preprocess the `defect` and `notdefect` image datasets by resizing them to `224 × 224`, converting them into tensors, and normalizing them.

### STEP 2:
Load the pre-trained VGG19 model with ImageNet weights and freeze the pre-trained layers.

### STEP 3:
Replace the final classifier layer with a layer having 2 output classes and define Cross Entropy Loss and Adam optimizer.

### STEP 4:
Train the model for 10 epochs using the training dataset and calculate training and validation loss.

### STEP 5:
Evaluate the model using test accuracy, confusion matrix, and classification report.

### STEP 6:
Predict individual test images using Softmax and display the actual class, predicted class, and confidence.


## PROGRAM

### Name:Karthikeyan C

### Register Number:212224040152

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
```
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for pre-trained model input
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
```
```python
dataset_path = "C:\\Users\\admin\\Downloads\\deep learning\\chip_data\\dataset"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)
```
```python
def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0) 
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.show()
```
```python
show_sample_images(train_dataset)
```
```python
print(f"Total number of training samples: {len(train_dataset)}")
first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")
```
```python
print(f"Total number of testing samples: {len(test_dataset)}")
first_image, label = test_dataset[0]
print(f"Shape of the first image: {first_image.shape}")
```
```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```
```python
num_classes=len(train_dataset.classes)
print("Number of classes:",num_classes)
model=models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad=False
model.classifier[6]=nn.Linear(model.classifier[6].in_features,num_classes)
criterion=nn.CrossEntropyLoss()

```
```python
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)
```

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```
```python
from torchsummary import summary
summary(model, input_size=(3, 224, 224))
```
```python
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    training_losses = []
    validation_losses = []
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            running_loss += loss.item()

        training_losses.append(running_loss / len(train_loader))
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {training_losses[-1]:.4f}")

    print("Training complete.")
```
```python
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    
    # Plot training and validation loss
   
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
```
```python
## Step 4: Test the Model and Compute Confusion Matrix & Classification Report
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
```

```python
train_model(model, train_loader,test_loader)
test_model(model, test_loader)
```
```python
def predict_image(model, image_index, dataset):
    model.eval()

    image, label = dataset[image_index]

    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)

        # Two classes: defect and notdefect
        probabilities = torch.softmax(output, dim=1)
        predicted = torch.argmax(probabilities, dim=1).item()

    class_names = dataset.classes

    # Display image
    image_to_display = transforms.ToPILImage()(image)

    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_display)
    plt.title(
        f"Actual: {class_names[label]}\n"
        f"Predicted: {class_names[predicted]}"
    )
    plt.axis("off")
    plt.show()

    print("Actual:", class_names[label])
    print("Predicted:", class_names[predicted])
    print("Confidence:", f"{probabilities[0][predicted].item():.2%}")

```
```python
# Example Prediction
predict_image(model, image_index=55, dataset=test_dataset)

#Example Prediction
predict_image(model, image_index=25, dataset=test_dataset)
```


### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="741" height="796" alt="image" src="https://github.com/user-attachments/assets/2a26973c-d66a-49d4-9ecc-f3c874b4f339" />


## Confusion Matrix

<img width="692" height="621" alt="image" src="https://github.com/user-attachments/assets/953dc952-38ae-410c-918f-3057b3d8f843" />


## Classification Report

<img width="687" height="216" alt="image" src="https://github.com/user-attachments/assets/1eb682f3-051d-4c04-ad30-3d63f9cfaacc" />


### New Sample Data Prediction

<img width="1053" height="997" alt="image" src="https://github.com/user-attachments/assets/ef668e9f-e81b-4fdd-a1ca-b0ba396631c4" />


## RESULT
The VGG19 transfer learning model was successfully developed to classify chip images into defect and notdefect classes. The model was evaluated using accuracy, confusion matrix, and classification report, and successfully predicted sample images.
