import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from model import *
from torch.utils.tensorboard import SummaryWriter
import  time

###cuda加速只能针对 model、损失函数、数据

#准备数据集
train_data = torchvision.datasets.CIFAR10(
    root='..\data_set', train=True, download=True, transform=torchvision.transforms.ToTensor()
)
test_data = torchvision.datasets.CIFAR10(
    root='..\data_set', train=False, download=True, transform=torchvision.transforms.ToTensor()
)

#length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度:{}".format(train_data_size))
print("测试数据集长度:{}".format(test_data_size))

#利用Dataloader数据集加载数据
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=64)

#创建网络模型
sh = Net()
if torch.cuda.is_available():
    sh =sh.cuda()
    print("cuda.is available!")
#损失函数
loss_fun = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fun = loss_fun.cuda()
#优化器
learning_rate = 1e-2
optimizer = torch.optim.Adam(sh.parameters(), lr=learning_rate)

##记录训练的网络参数
train_step = 0
#test_step = 0
epochs = 10
##添加tesnsorboard
writer = SummaryWriter('./logs_train')

start_time = time.time()
for epoch in range(epochs):
    print("-----------第{}轮训练开始----------".format(epoch+1))

    #训练
    sh.train()
    for data in train_dataloader:
        img,targets = data
        if torch.cuda.is_available():
            img,targets = img.cuda(),targets.cuda()
        output = sh(img)
        loss = loss_fun(output, targets)

        #优化器
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_step += 1
        if train_step % 100 == 0:
            end_time = time.time()
            print("花费时间:{}".format(end_time - start_time))
            print("训练次数：{},损失{}".format(train_step,loss.item()))
            writer.add_scalar('loss', loss.item(), train_step)

    #测试
    sh.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_dataloader:
            img, targets = data
            if torch.cuda.is_available():
                img, targets = img.cuda(),targets.cuda()
            output = sh(img)
            loss = loss_fun(output, targets)
            total_loss += loss.item()
            #pred = output.argmax(dim=1)
            correct +=(output.argmax(dim=1)==targets).sum().item()

    print("整体测试集上的loss:{}".format(total_loss))
    print("整体测试集上的正确率:{}".format(correct/test_data_size))
    writer.add_scalar('test_loss', total_loss, epoch)
    writer.add_scalar('test_accuracy', correct/test_data_size, epoch)

    ##保存模型
    torch.save(sh.state_dict(), 'sh_{}.pth'.format(epoch))
    print("saved")
writer.close()





