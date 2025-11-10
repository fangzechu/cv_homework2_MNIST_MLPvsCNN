
# -*- coding: utf-8 -*-
import argparse, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class MLP_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            Flatten(),
            nn.Linear(28*28, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )
    def forward(self, x): return self.net(x)

class MLP_B(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            Flatten(),
            nn.Linear(28*28, 8), nn.ReLU(inplace=True),
            nn.Linear(8, 10),
        )
    def forward(self, x): return self.net(x)

class CNN_A(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(4, 8, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(Flatten(), nn.Linear(8*4*4, 10))
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CNN_B(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(Flatten(), nn.Linear(16*4*4, 10))
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def train_one(model, train_loader, test_loader, device, epochs=5, lr=1e-3, weight_decay=0.0):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = accuracy(model, test_loader, device)
        print(f"Epoch {ep:02d} | Test Acc: {acc*100:.2f}% | time {time.time()-t0:.1f}s")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="MLP_A",
                   choices=["MLP_A", "MLP_B", "CNN_A", "CNN_B"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--aug", action="store_true", help="是否加轻度数据增强")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tfms = [transforms.ToTensor()]
    if args.aug:
        tfms = [transforms.RandomAffine(degrees=10, translate=(0.05,0.05), scale=(0.95,1.05))] + tfms
    transform = transforms.Compose(tfms)

    train_set = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_set  = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    model = {"MLP_A": MLP_A, "MLP_B": MLP_B, "CNN_A": CNN_A, "CNN_B": CNN_B}[args.arch]()
    print(model)
    print("Total params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_one(model, train_loader, test_loader, device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)

if __name__ == "__main__":
    main()
