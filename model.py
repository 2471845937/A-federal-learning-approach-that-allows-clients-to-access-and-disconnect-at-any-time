import torch
from torch import nn
from torchvision.models import mobilenet_v2


def create_model():
    model = mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, 10)  # CIFAR10 has 10 classes
    return model


def test(model, test_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    avg_loss = total_loss / len(test_loader)
    return avg_loss, accuracy