import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cnn import SketchyCNN


import PIL.Image
from PIL.Image import Image
import numpy as np
import matplotlib as plt


class SketchyRecognizer:

    cnn: SketchyCNN = SketchyCNN()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    object_list: list[str] = ["house plant", "guitar", "basketball", "sword", "door", "key", "lantern", "chair", "pencil", "axe"]

    batch_size: int = 32

    train_accuracies = []
    train_losses = []

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.0), (0.5))
        ])
    

    def __init__(self, auto_load_model: bool = True):
        print("Using:\t", SketchyRecognizer.device)

        if auto_load_model:
            self.load_model()
        
        SketchyRecognizer.cnn.to(SketchyRecognizer.device)


    def load_model(self, model_name: str = "model.mdl") -> None: 
        try: 
            SketchyRecognizer.cnn.load_state_dict(torch.load(model_name))
        except: 
            print("Model was not loaded")


    def save_model(self, model_name: str = "model.mdl") -> None: 
        try:
            torch.save(SketchyRecognizer.cnn, model_name)
        except: 
            print("Model could not be saved")


    def _predict(self, tensor: torch.Tensor) -> dict[str, int|str]:
        device_tensor = tensor.to(SketchyRecognizer.device)
        prediction: torch.Tensor = SketchyRecognizer.cnn.forward(device_tensor)
        host_prediction = prediction.to("cpu")
        predicted_class: int = host_prediction.argmax(dim=0).item()
        return {"class_id": predicted_class, "class_name": SketchyRecognizer.object_list[predicted_class]} 


    def predict_from_image(self, image: Image) -> dict[str, int|str]:
        input_tensor: torch.Tensor = SketchyRecognizer.transform(image)
        return self._predict(input_tensor)


    def predict_from_array(self, array: np.ndarray) -> dict[str, int|str]:
        input_tensor = transforms.ToTensor()(np.array(array))
        input_tensor = transforms.Resize((64, 64))(input_tensor)
        input_tensor = transforms.Normalize((0.0), (0.5))(input_tensor)
        return self._predict(input_tensor)


    def _train_one_epoch(self, dataloader, criterion, optimizer: optim.Optimizer):
        SketchyRecognizer.cnn.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = SketchyRecognizer.cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total * 100
        return total_loss, accuracy


    def train(self, epochs: int = 69, learning_rate: float = 0.001) -> None:
        train_dir = "assets/train"
        valid_dir = "assets/valid"

        train_dataset = datasets.ImageFolder(train_dir, transform=SketchyRecognizer.transform)
        valid_dataset = datasets.ImageFolder(valid_dir, transform=SketchyRecognizer.transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

        print(f"Classes: {train_dataset.classes}")
        print(f"Training images: {len(train_dataset)}, Validation images: {len(valid_dataset)}")
    
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(SketchyRecognizer.cnn.parameters(), lr=learning_rate)

        for _ in epochs:
            loss, accuracy = self._train_one_epoch(dataloader, criterion, optimizer)
            self.train_accuracies.append(accuracy)
            self.train_losses.append(loss)


    def _validate(self, model, dataloader, criterion):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total * 100
        return total_loss, accuracy


    def evaluate(self) -> None: 
        # best_accuracy = 0.0

        test_dir = "assets/valid"
        test_dataset = datasets.ImageFolder(test_dir, transform=SketchyRecognizer.transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()

        loss, accuracy = self._validate(SketchyRecognizer.cnn, test_loader, criterion)

        self.train_losses.append(loss)
        self.train_accuracies.append(accuracy)
        epochs = list(range(1, len(self.train_losses) + 1)) + ['Eval']

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(epochs, self.train_losses, color='red')
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        plt.title('Loss per Training Epoch Plot')

        plt.subplot(1, 2, 2)
        plt.bar(epochs, self.train_accuracies, color='green')
        plt.xlabel('Epoch Number')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Training Epoch Plot')

        plt.tight_layout()
        plt.show()

        # if accuracy > best_accuracy:
        #     best_accuracy = accuracy
        #     torch.save(SketchyRecognizer.cnn.state_dict(), "sketchy_model_best.pth")
        #     print("Saved better one")

        print("DONE")

