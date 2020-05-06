import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from Cifar10Dataset import Cifar10Dataset
from VGGNet import VGGNet

    

def train(model, train_dataset, test_dataset, log_name, n_epochs, optimizer, batch_size, gpu_available, transform=False):
    if gpu_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 定义损失函数和优化器
    writer = SummaryWriter(f'./log/' + log_name + '/')
    lossfunc = torch.nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # 开始训练
    for epoch in range(n_epochs):
        if (epoch + 1) % 50 == 0:   # 每50个epoch乘0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            output = model(data)  # 得到预测值
            # print(target.size())
            # print(output.size())
            loss = lossfunc(output, target)  # 计算两者的误差
            loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
            train_loss += loss.item() * data.size(0)
        train_loss = train_loss / len(train_loader.dataset)
        print('\tEpoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
        # 每遍历一遍数据集，测试一下准确率
        train_accuracy = test(model, train_dataset, batch_size, gpu_available) # 训练集准确率
        print('\ttrain Accuracy: %.4f %%' % (100 * train_accuracy))
        test_accuracy = test(model, test_dataset, batch_size, gpu_available)  # 测试集准确率
        print('\ttest Accuracy: %.4f %%' % (100 * test_accuracy))

        writer.add_scalar('train/Loss', train_loss, epoch + 1)
        writer.add_scalar('train/Accuracy', train_accuracy, epoch + 1)
        writer.add_scalar('test/Accuracy', test_accuracy, epoch + 1)
        writer.flush()
    writer.close()


# 在数据集上测试神经网络
def test(model, test_dataset, batch_size, gpu_available):
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    if gpu_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    correct = 0
    total = 0
    with torch.no_grad():  # 测试集中不需要反向传播
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)   # 概率最大的就是输出的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 有几个相同的

    return correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", dest="batch_size", default=128, type=int)
    parser.add_argument("--epoch", dest="epoch", default=100, type=int)
    parser.add_argument("--learning_rate", dest="lr", default=0.01,type=float)
    parser.add_argument("--gpu", dest="gpu", default=True, type=bool)
    args = parser.parse_args()

    train_dataset_NoTransform = Cifar10Dataset('./cifar-10-batches-py/', train=True)
    test_dataset = Cifar10Dataset('./cifar-10-batches-py/', train=False)

    
    # lrs = [0.1, 0.01, 1e-3, 1e-4, 1e-5]
    # for lr in lrs:
    #     model = VGGNet(se_block=False)
    #     sgd = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)
    #     print('VGG_SGD_NoSE_NoTransform_lr'+str(lr)+':')
    #     try:
    #         train(model, train_dataset_NoTransform, test_dataset, log_name='VGG_SGD_NoSE_NoTransform_lr'+str(lr), n_epochs=args.epoch, optimizer=sgd, batch_size=args.batch_size, gpu_available=args.gpu)
    #     except KeyboardInterrupt:
    #         pass
        
    # for lr in lrs:
    #     model = VGGNet(se_block=False)
    #     adam = torch.optim.Adam(params=model.parameters(), lr=lr)
    #     print('VGG_Adam_NoSE_NoTransform_lr'+str(lr)+':')
    #     try:
    #         train(model, train_dataset_NoTransform, test_dataset, log_name='VGG_Adam_NoSE_NoTransform_lr'+str(lr), n_epochs=args.epoch, optimizer=adam, batch_size=args.batch_size, gpu_available=args.gpu)
    #     except KeyboardInterrupt:
    #         pass
    
    lrs = [0.1, 0.01, 1e-3, 1e-4, 1e-5]
    for lr in lrs:
        model = VGGNet(se_block=True)
        sgd = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)
        print('VGG_SGD_SE_NoTransform_lr'+str(lr)+':')
        try:
            train(model, train_dataset_NoTransform, test_dataset, log_name='VGG_SGD_SE_NoTransform_lr'+str(lr), n_epochs=args.epoch, optimizer=sgd, batch_size=args.batch_size, gpu_available=args.gpu)
        except KeyboardInterrupt:
            pass    
            
    for lr in lrs:
        model = VGGNet(se_block=True)
        adam = torch.optim.Adam(params=model.parameters(), lr=lr)
        print('VGG_Adam_SE_NoTransform_lr'+str(lr)+':')
        try:
            train(model, train_dataset_NoTransform, test_dataset, log_name='VGG_Adam_SE_NoTransform_lr'+str(lr), n_epochs=args.epoch, optimizer=adam, batch_size=args.batch_size, gpu_available=args.gpu)
        except KeyboardInterrupt:
            pass