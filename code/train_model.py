import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import os

# ================= DATASET =================
class LaneDataset(Dataset):
    def __init__(self, image_folder, label_folder):
        self.images = sorted(os.listdir(image_folder))
        self.image_folder = image_folder
        self.label_folder = label_folder

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.images[idx])
        label_path = os.path.join(self.label_folder, self.images[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = cv2.imread(label_path, 0)  # grayscale

        img = self.transform(img)
        label = cv2.resize(label, (224, 224))
        label = torch.tensor(label, dtype=torch.long)

        return img, label


# ================= LOAD DATA =================
dataset = LaneDataset("../dataset/frames/", "../dataset/annotations/")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Loaded {len(dataset)} images")

# ================= MODEL =================
model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
model.train()

# ================= LOSS & OPTIMIZER =================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ================= TRAINING LOOP =================
epochs = 1  # keep small for demo

for epoch in range(epochs):
    for images, labels in dataloader:

        outputs = model(images)['out']

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# ================= SAVE MODEL =================
torch.save(model.state_dict(), "custom_lane_model.pth")

print("Model training complete and saved!")