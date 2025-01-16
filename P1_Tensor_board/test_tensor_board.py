from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('../logs')

#writer.add_image()
#y=x
for  i in range(0,100):
    writer.add_scalar('y=2x', 3*i, i)
writer.add_scalar('loss', 1, global_step=5000)

writer.close()
