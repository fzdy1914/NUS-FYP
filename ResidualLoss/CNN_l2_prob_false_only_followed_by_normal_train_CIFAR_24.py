import random
import sys

from torch.autograd import Variable
from torch import optim

import numpy as np
from torch.backends import cudnn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import cifar10_data_loader_test, cifar10_data_loader_train, cifar10_dataset_train
from model import CIFAR_24


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

model = CIFAR_24().to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

train_dataset = cifar10_dataset_train()
train_data_length = len(train_dataset)
sampler = WeightedRandomSampler([1] * train_data_length, num_samples=train_data_length, replacement=True)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

evaluation_data_loader = cifar10_data_loader_train(batch_size=evaluation_batch_size, shuffle=False)
test_data_loader = cifar10_data_loader_test(batch_size)


def residual_train():
    prob = torch.zeros(len(train_dataset), dtype=torch.float64)
    lst = torch.load("./analysis-and-draw/data/CNN-30-CIFAR_24-lower_10.pt")
    for idx in lst:
        prob[idx] = 1

    sampler.weights = prob
    print(prob.sum())

    total_correct_sum = 0
    total_classification_loss = 0

    for epoch in range(num_epochs):
        total_correct = 0
        model.eval()

        with torch.no_grad():
            for data, target in evaluation_data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                total_correct += pred.eq(target.view_as(pred)).sum().item()

        model.train()
        total_train_loss = 0
        for data, target in train_data_loader:
            data, target = data.to(device), target.to(device)
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
            torch.save(model.state_dict(), "./CNN-Train/CIFAR_24/false_epoch-%s.pt" % (epoch + 1))

    prob = torch.ones(len(train_dataset), dtype=torch.float64)
    sampler.weights = prob
    print(prob.sum())

    for epoch in range(200):
        total_correct = 0
        model.eval()

        with torch.no_grad():
            for data, target in evaluation_data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                total_correct += pred.eq(target.view_as(pred)).sum().item()

        model.train()
        total_train_loss = 0
        for data, target in train_data_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)
            loss = F.nll_loss(output, target)

            loss.backward()
            optimizer.step()

            total_train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

        total_train_loss /= train_data_length
        total_correct_sum += total_correct
        total_classification_loss += total_train_loss
        print('epoch [{}/{}], loss:{:.4f} Accuracy: {}/{}'.format(epoch + 1, num_epochs, total_train_loss, total_correct, train_data_length))
        torch.save(model.state_dict(), "./CNN-Train/CIFAR_24/back_epoch-%s.pt" % (epoch + 1))

    print("average correct:", total_correct_sum / num_epochs)
    print("average loss:", total_classification_loss / num_epochs)


# 1000, 500, 200, 100, 75, 50, 25, 10, 5, 1, 0.5,


if __name__ == '__main__':
    residual_train()
