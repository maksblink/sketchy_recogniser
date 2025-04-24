import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import PIL.Image

class SketchyCNN(nn.Module): 

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)   # 64x64 → 60x60
        self.pool1 = nn.MaxPool2d(kernel_size=2)                                # 60x60 → 30x30

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)  # 30x30 → 28x28
        self.pool2 = nn.MaxPool2d(kernel_size=2)                                # 28x28 → 14x14

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3) # 14x14 → 12x12

        self.fc1 = nn.Linear(3*3*256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

        self.dropout1 = nn.Dropout(p=0.5)
        # self.dropout2 = nn.Dropout(p=0.2) # Optional

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout2(x) # Optional
        x = self.fc3(x)
        return x


class SketchyRecognizer:

    cnn: SketchyCNN = SketchyCNN()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    object_list: list[str] = ["house plant", "guitar", "basketball", "sword", "door", "key", "lantern", "chair", "pencil", "axe"]

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
        pass

    def evaluate() -> None: # tu zrobić przejście przez dane testowe, może też tu je załadować. Ma pokazać wykresy z wynikami
        pass


sketchy_recognizer = SketchyRecognizer()
test_im_path: str = "/home/nsjg/Desktop/Sketchy_prj/sketchy_recogniser/assets/classes/basketball/basketball_9000.png"

image = PIL.Image.open(test_im_path)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.Resize((28, 28)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.0), (0.5))
])

transformed_image = transform(image)


# print(transformed_image)
print(sketchy_recognizer.predict(transformed_image))