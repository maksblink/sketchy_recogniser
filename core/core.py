import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from core.cnn import SketchyCNN

# Traning params
BATCH_SIZE = 32
EPOCHS = 77
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SketchyRecognizer:

    cnn: SketchyCNN = SketchyCNN()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    object_list: list[str] = ["house plant", "guitar", "basketball", "sword", "door", "key", "lantern", "chair", "pencil", "axe"]

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
        
        # SketchyRecognizer.cnn.to(SketchyRecognizer.device)

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

    def predict(self, image) -> dict[str, int|str]:
        prediction: torch.Tensor = SketchyRecognizer.cnn.forward(image)
        predicted_class: int = prediction.argmax(dim=0)
        return {"class_id": predicted_class, "class_name": SketchyRecognizer.object_list[predicted_class]} 
    
    def train() -> None: # tu przepisać z core, wszystkie dane podzielić na train in test i zapisać w polach klasowych? 
        train_dir = "assets/train"
        valid_dir = "assets/valid"

        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print(f"Classes: {train_dataset.classes}")
        print(f"Training images: {len(train_dataset)}, Validation images: {len(valid_dataset)}")
    
        # model = SketchyCNN().to(DEVICE)


        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


        # def train_one_epoch(model, dataloader, criterion, optimizer):
        #     model.train()
        #     total_loss, correct, total = 0.0, 0, 0

        #     for images, labels in dataloader:
        #         images, labels = images.to(DEVICE), labels.to(DEVICE)

        #         optimizer.zero_grad()
        #         outputs = model(images)
        #         loss = criterion(outputs, labels)
        #         loss.backward()
        #         optimizer.step()

        #         total_loss += loss.item()
        #         _, predicted = outputs.max(1)
        #         correct += (predicted == labels).sum().item()
        #         total += labels.size(0)

        #     accuracy = correct / total * 100
        #     return total_loss, accuracy

        # def validate(model, dataloader, criterion):
        #     model.eval()
        #     total_loss, correct, total = 0.0, 0, 0

        #     with torch.no_grad():
        #         for images, labels in dataloader:
        #             images, labels = images.to(DEVICE), labels.to(DEVICE)
        #             outputs = model(images)
        #             loss = criterion(outputs, labels)

        #             total_loss += loss.item()
        #             _, predicted = outputs.max(1)
        #             correct += (predicted == labels).sum().item()
        #             total += labels.size(0)

        #     accuracy = correct / total * 100
        #     return total_loss, accuracy

    def evaluate() -> None: # tu zrobić przejście przez dane testowe, może też tu je załadować. Ma pokazać wykresy z wynikami
        # best_val_acc = 0.0

        # for epoch in range(EPOCHS):
        #     train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        #     val_loss, val_acc = validate(model, valid_loader, criterion)

        #     print(f"\nEpoch {epoch+1}/{EPOCHS}")
        #     print(f"Train - Loss: {train_loss:.5f}, Accuracy: {train_acc:.5f}%")
        #     print(f"Valid - Loss: {val_loss:.5f}, Accuracy: {val_acc:.5f}%")


        #     if val_acc > best_val_acc:
        #         best_val_acc = val_acc
        #         torch.save(model.state_dict(), "sketchy_model_best.pth")
        #         print("Saved better one")

        # print("DONE")


















# Merge this to class method

# import os
# import shutil
# import random
# from pathlib import Path


# BASE_DIR = Path(__file__).resolve().parent.parent
# SOURCE_DIR = BASE_DIR / "assets" / "classes"
# TRAIN_DIR = BASE_DIR / "assets" / "train"
# VALID_DIR = BASE_DIR / "assets" / "valid"

# SPLIT_RATIO = 0.8
# SEED = 42

# random.seed(SEED)

# for target_dir in [TRAIN_DIR, VALID_DIR]:
#     if target_dir.exists():
#         shutil.rmtree(target_dir)
#     target_dir.mkdir(parents=True)


# for class_folder in SOURCE_DIR.iterdir():
#     if class_folder.is_dir():
#         images = list(class_folder.glob("*.*"))
#         random.shuffle(images)

#         split_idx = int(len(images) * SPLIT_RATIO)
#         train_images = images[:split_idx]
#         valid_images = images[split_idx:]

#         train_class_dir = TRAIN_DIR / class_folder.name
#         valid_class_dir = VALID_DIR / class_folder.name
#         train_class_dir.mkdir(parents=True)
#         valid_class_dir.mkdir(parents=True)

#         for img_path in train_images:
#             shutil.copy(img_path, train_class_dir / img_path.name)
#         for img_path in valid_images:
#             shutil.copy(img_path, valid_class_dir / img_path.name)

#         print(f"Class '{class_folder.name}': {len(train_images)} train, {len(valid_images)} valid")

# print("DONE")
