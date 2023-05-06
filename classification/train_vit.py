import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
# from timm.models.vision_transformer import vit_base_patch16_224
# from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from transformers import ViTImageProcessor, ViTForImageClassification
# 设置随机种子
torch.manual_seed(42)

# 定义数据预处理

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.6821, 0.6261, 0.6226), (0.3039, 0.3045, 0.2991))
])

# 加载数据集
train_dataset = ImageFolder('my_crawler', transform=transform)
val_dataset = ImageFolder('my_crawler', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)

# 加载模型
# model = vit_base_patch16_224(pretrained=True)
# model = AutoModelForImageClassification.from_pretrained("lysandre/tiny-vit-random")
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model = AutoModelForImageClassification.from_pretrained("WinKawaks/vit-small-patch16-224")
model.classifier = nn.Linear(model.classifier.in_features, 2)
# import pdb
# pdb.set_trace()

del model.vit.encoder.layer[6:]
print(model)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 训练模型
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
best_acc = 0
for epoch in range(num_epochs):
    # 训练
    model.train()
    train_loss = 0.0
    train_corrects = 0
    from tqdm import tqdm
    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # import pdb
        # pdb.set_trace()
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs.logits, 1)
        train_loss += loss.item() * inputs.size(0)
        train_corrects += torch.sum(preds == labels.data)
    train_loss = train_loss / len(train_dataset)
    train_acc = train_corrects.double() / len(train_dataset)

    # 验证
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            _, preds = torch.max(outputs.logits, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
    val_loss = val_loss / len(val_dataset)
    val_acc = val_corrects.double() / len(val_dataset)

    # 输出训练情况
    print('Epoch [{}/{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))
    if val_acc > best_acc:
        best_acc = val_acc
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }

        torch.save(checkpoint, 'checkpoint_small_best.pth')

print("hello world")
import pdb
pdb.set_trace()