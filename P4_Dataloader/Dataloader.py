import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#准备测试集
test_data = torchvision.datasets.CIFAR10("../data_set",train=False, transform=torchvision.transforms.ToTensor())
#实例化DataLoader
test_loader = DataLoader(\
    test_data, batch_size=64, shuffle=False, num_workers=0,drop_last=False\
      )

img,target = test_data[0]
print(img.shape)
print(target)

# import os
# log_dir = os.path.abspath('dataloader')
# writer = SummaryWriter(log_dir)
# print(log_dir)
writer = SummaryWriter("../dataloader")
##对样本读取2//模拟训练
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs,targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images(f"Epoch:{epoch}",imgs,step)
        step += 1

writer.close()