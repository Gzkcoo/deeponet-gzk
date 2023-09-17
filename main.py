from models import DeepONet, CustomizedDataset
from utils import generate_data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as opt
from tqdm import tqdm
import numpy as np


def train(model, data, optimizer, epoch, batch_size):
    customized_train_data = CustomizedDataset(data[0])
    customized_test_data = CustomizedDataset(data[1])
    train_data_loader = DataLoader(dataset=customized_train_data, batch_size=batch_size, shuffle=True,
                                   num_workers=0, drop_last=True)
    test_data_loader = DataLoader(dataset=customized_test_data, batch_size=batch_size, shuffle=False,
                                   num_workers=0, drop_last=False)
    num_test_data = data[1][1].shape[0]

    loss_func_train = nn.MSELoss(reduction="mean")
    loss_func_test = nn.MSELoss(reduction="none")
    for i in tqdm(range(1, epoch+1)):
        model.train()
        for _, (x, y) in enumerate(train_data_loader):
            # x = torch.from_numpy(x)
            # y = torch.from_numpy(y)
            x = x.float()
            y = y.float()
            optimizer.zero_grad()
            loss = loss_func_train(model(x), y)
            loss.backward()
            optimizer.step()
            # test数据上的损失值
        # if i % 10 == 0:
        #     model.eval()
        #     test_loss = 0.0
        #     for _, (x, y) in enumerate(test_data_loader):
        #         # x = torch.from_numpy(x)
        #         # y = torch.from_numpy(y)
        #         x = x.float()
        #         y = y.float()
        #         test_loss += torch.sum(loss_func_test(model(x),y)).item()
        #     test_loss /= num_test_data
        #     print(test_loss)






if __name__ == '__main__':
    data = [generate_data(num_data=10000), generate_data(num_data=5000)]
    model = DeepONet()
    lr = 1e-3
    epoch = 1000
    batch_size = 128
    optimizer = opt.Adam(model.parameters(), lr)
    train(model, data, optimizer, epoch, batch_size)

    # 保存模型参数
    torch.save(model.state_dict(),'deeponet.params')

    # 从文件回复模型
    copy_model = DeepONet()
    copy_model.load_state_dict(torch.load('deeponet.params'))

    # 测试

    x_sin = np.linspace(0, 1, num=100)
    u_sin = torch.sin(torch.from_numpy(x_sin))
    u_sin = torch.unsqueeze(u_sin, 0)
    test = torch.from_numpy(np.hstack((u_sin, [[0.3]])))
    # test = torch.from_numpy(np.hstack((u_sin, [[0.5]])))
    test = torch.from_numpy(np.vstack((test, np.hstack((u_sin, [[0.5]])))))
    test = test.float()
    print(f'model: {model(test)}')
    print(f'copy_model: {copy_model(test)}')




