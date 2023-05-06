import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torchvision
import torchvision.transforms as transforms

# 定义模型
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 2)  # 二分类

    def forward(self, x):
        x = self.resnet18(x)
        x = self.fc(x)
        return x

# 加载数据集

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.6821, 0.6261, 0.6226), (0.3039, 0.3045, 0.2991))
])

trainset = torchvision.datasets.ImageFolder(root='my_crawler', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, pin_memory=True)

# 定义损失函数、优化器和学习率调度器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# 训练模型
print(torch.cuda.is_available())

best_acc = 0
for epoch in range(20):  # 训练50个epoch
    net.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # 前向传播、计算损失、反向传播、更新权重
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 统计训练损失、准确率
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # 输出训练结果
    print('Epoch %d, Training Loss: %.3f, Training Accuracy: %.3f%%' % (epoch + 1, train_loss / (i + 1), 100.0 * correct / total))
    if 100.0 * correct / total > best_acc:
        best_acc = 100.0 * correct / total
        checkpoint = {
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }

        torch.save(checkpoint, 'checkpoint_r18_best.pth')

    # 更新学习率
    scheduler.step()
    
print("hello world")
import pdb
pdb.set_trace()

