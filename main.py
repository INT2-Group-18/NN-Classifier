import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv_layers = nn.Sequential(

            # Conv Layer 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.05),

            # Conv Layer 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.05),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.05),

            # Conv Layer 4
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer 5
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.05),

            # Conv Layer 6
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layers(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x


# a function to calculate the accuracy of our model, given the predictions and the labels
def accuracy(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    return torch.mean((classes == labels).float())


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # transform datasets and augument training data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.ColorJitter(),
         transforms.RandomCrop(32, padding=4),
         transforms.RandomRotation(15),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
         transforms.RandomHorizontalFlip()])

    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
         transforms.RandomHorizontalFlip()])

    batch_size = 64

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=3, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=3, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    PATH = './cifar_net.pth'
    net = Net()
    net.to(device, non_blocking=True)

    # initialize optimizer, loss function, learning rate scheduler and automatic mixed precision
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.008, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.008, max_lr=0.012)
    scaler = amp.GradScaler()

    # main training loop
    for epoch in range(50):  # loop over the dataset multiple times

        net.train()
        test_accuracy = 0.0
        running_loss = 0.0
        running_accuracy = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with amp.autocast():
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            # forward + backward + optimize using mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            # scheduler.step()
            scaler.update()

            # calculate statistics
            running_loss += loss.item()
            running_accuracy += accuracy(outputs, labels)

        correct = 0
        net.eval()
        with torch.no_grad():
            for testdata in testloader:
                images, test_labels = testdata
                images, test_labels = images.to(device), test_labels.to(device)
                test_outputs = net(images)
                correct += accuracy(test_outputs, test_labels)

            test_accuracy = correct / len(testloader)

        print('%d, loss: %.3f, accuracy: %.3f, test accuracy: ' %
              (epoch + 1, running_loss / len(trainloader), running_accuracy / len(trainloader), test_accuracy))

    print('Finished Training')

    # after training, save the model and calculate the validation accuracy
    torch.save(net.state_dict(), PATH)
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    net = Net()
    net.load_state_dict(torch.load(PATH))
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


if __name__ == "__main__":
    main()
