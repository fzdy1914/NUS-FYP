import random
import sys

from torch.autograd import Variable
from torch import optim

import numpy as np
from torch.backends import cudnn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from ResidualLoss.dataset import cifar10_data_loader_test, cifar10_data_loader_train, cifar10_dataset_train
from ResidualLoss.model import CIFAR_11_6


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        log_loc = "./log/%s.txt" % sys.argv[0].split("/")[-1].split(".")[0]
        self.log = open(log_loc, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


sys.stdout = Logger()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True


setup_seed(1914)
num_epochs = 2000
batch_size = 100
evaluation_batch_size = 100
learning_rate = 0.0001

model = CIFAR_11_6().cuda()
state_dict = torch.load('./CNN-Train/CIFAR_11_6/epoch-5000.pt')
model.eval()

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

train_dataset = cifar10_dataset_train()
train_data_length = len(train_dataset)
sampler = WeightedRandomSampler([1] * train_data_length, num_samples=train_data_length, replacement=True)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

evaluation_data_loader = cifar10_data_loader_train(batch_size=evaluation_batch_size, shuffle=False)
test_data_loader = cifar10_data_loader_test(batch_size)


def residual_train():
    prob = torch.zeros(len(train_dataset), dtype=torch.float64)
    start_index = 0
    for data, target in evaluation_data_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.argmax(dim=1)
        correct_list = pred.eq(target)

        for i in range(correct_list.size(0)):
            if not correct_list[i].item():
                prob[start_index + i] = 1

        start_index += evaluation_batch_size

    sampler.weights = prob
    print(prob.sum())

    total_correct_sum = 0
    total_classification_loss = 0
    model.load_state_dict(CIFAR_11_6().state_dict())

    for epoch in range(num_epochs):
        total_correct = 0
        model.eval()

        with torch.no_grad():
            for data, target in evaluation_data_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                total_correct += pred.eq(target.view_as(pred)).sum().item()

        model.train()
        total_train_loss = 0
        for data, target in train_data_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            output, features = model.features(data)
            loss = F.nll_loss(output, target)

            loss.backward()
            optimizer.step()

            total_train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

        total_train_loss /= train_data_length
        total_correct_sum += total_correct
        total_classification_loss += total_train_loss
        if (epoch + 1) % 50 == 0:
            print('epoch [{}/{}], loss:{:.4f} Accuracy: {}/{}'.format(epoch + 1, num_epochs, total_train_loss, total_correct, train_data_length))
            torch.save(model.state_dict(), "./CNN-Train/CIFAR_11_6/false_epoch-%s.pt" % (epoch + 1))

    prob = torch.ones(len(train_dataset), dtype=torch.float64)
    sampler.weights = prob
    print(prob.sum())

    for epoch in range(int(num_epochs / 2)):
        total_correct = 0
        model.eval()

        with torch.no_grad():
            for data, target in evaluation_data_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                total_correct += pred.eq(target.view_as(pred)).sum().item()

        model.train()
        total_train_loss = 0
        for data, target in train_data_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            output = model(data)
            loss = F.nll_loss(output, target)

            loss.backward()
            optimizer.step()

            total_train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

        total_train_loss /= train_data_length
        total_correct_sum += total_correct
        total_classification_loss += total_train_loss
        if (epoch + 1) % 50 == 0:
            print('epoch [{}/{}], loss:{:.4f} Accuracy: {}/{}'.format(epoch + 1, num_epochs, total_train_loss, total_correct, train_data_length))
            torch.save(model.state_dict(), "./CNN-Train/CIFAR_11_6/back_epoch-%s.pt" % (epoch + 1))

    print("average correct:", total_correct_sum / num_epochs)
    print("average loss:", total_classification_loss / num_epochs)


# 1000, 500, 200, 100, 75, 50, 25, 10, 5, 1, 0.5,


if __name__ == '__main__':
    model.load_state_dict(state_dict)
    residual_train()