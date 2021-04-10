import torch
from torch import nn

import torch.nn.functional as F


class PaperModel(nn.Module):
    def __init__(self):
        super(PaperModel, self).__init__()
        self.hidden1 = nn.Linear(784, 784)
        self.hidden2 = nn.Linear(784, 784)
        self.output = nn.Linear(784, 10)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x


class MINST_1(nn.Module):
    def __init__(self):
        super(MINST_1, self).__init__()
        self.output = nn.Linear(784, 10)

    def forward(self, x):
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x


class MINST_2(nn.Module):
    def __init__(self):
        super(MINST_2, self).__init__()
        self.hidden1 = nn.Linear(784, 200)
        self.hidden2 = nn.Linear(200, 100)
        self.hidden3 = nn.Linear(100, 60)
        self.hidden4 = nn.Linear(60, 30)
        self.hidden5 = nn.Linear(30, 10)

    def forward(self, x):
        x = self.hidden1(x)
        x = torch.sigmoid(x)
        x = self.hidden2(x)
        x = torch.sigmoid(x)
        x = self.hidden3(x)
        x = torch.sigmoid(x)
        x = self.hidden4(x)
        x = torch.sigmoid(x)
        x = self.hidden5(x)
        x = F.log_softmax(x, dim=1)
        return x


class MINST_3(nn.Module):
    def __init__(self):
        super(MINST_3, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(inplace=True)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(inplace=True)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(100, 60),
            nn.ReLU(inplace=True)
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(60, 30),
            nn.ReLU(inplace=True)
        )
        self.hidden5 = nn.Sequential(
            nn.Linear(30, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = F.log_softmax(x, dim=1)
        return x


class MINST_4(nn.Module):
    def __init__(self):
        super(MINST_4, self).__init__()
        self.hidden1 = nn.Linear(784, 200)
        self.hidden2 = nn.Linear(200, 100)
        self.hidden3 = nn.Linear(100, 60)
        self.hidden4 = nn.Linear(60, 30)
        self.hidden5 = nn.Linear(30, 10)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.25)
        x = self.hidden2(x)
        x = F.relu(x)
        x = F.dropout(x, 0.25)
        x = self.hidden3(x)
        x = F.relu(x)
        x = F.dropout(x, 0.25)
        x = self.hidden4(x)
        x = F.relu(x)
        x = F.dropout(x, 0.25)
        x = self.hidden5(x)
        x = F.log_softmax(x, dim=1)
        return x


class MINST_7(nn.Module):
    def __init__(self):
        super(MINST_7, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(784, 200),
            nn.BatchNorm1d(200),
            nn.Sigmoid()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.Sigmoid()
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(100, 60),
            nn.BatchNorm1d(60),
            nn.Sigmoid()
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(60, 30),
            nn.BatchNorm1d(30),
            nn.Sigmoid()
        )
        self.hidden5 = nn.Sequential(
            nn.Linear(30, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = F.log_softmax(x, dim=1)
        return x


class MINST_8(nn.Module):
    def __init__(self):
        super(MINST_8, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(784, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(100, 60),
            nn.BatchNorm1d(60),
            nn.ReLU(inplace=True)
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(60, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(inplace=True)
        )
        self.hidden5 = nn.Sequential(
            nn.Linear(30, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = F.log_softmax(x, dim=1)
        return x


class MINST_9(nn.Module):
    def __init__(self):
        super(MINST_9, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, 28, 28)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_1(nn.Module):
    def __init__(self):
        super(CIFAR_1, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(1024 * 3, 2048),
            nn.ReLU(inplace=True)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )
        self.hidden5 = nn.Sequential(
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_2(nn.Module):
    def __init__(self):
        super(CIFAR_2, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(1024 * 3, 2048),
            nn.ReLU(inplace=True)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(inplace=True)
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_3(nn.Module):
    def __init__(self):
        super(CIFAR_3, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(1024 * 3, 2048),
            nn.ReLU(inplace=True)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_4(nn.Module):
    def __init__(self):
        super(CIFAR_4, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(1024 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_5(nn.Module):
    def __init__(self):
        super(CIFAR_5, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(1024 * 3, 32),
            nn.ReLU(inplace=True)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_6(nn.Module):
    def __init__(self):
        super(CIFAR_6, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(1024 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.hidden5 = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_7(nn.Module):
    def __init__(self):
        super(CIFAR_7, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(1024 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden5 = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_8(nn.Module):
    def __init__(self):
        super(CIFAR_8, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(1024 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = F.log_softmax(x, dim=1)
        return x

    def features(self, x):
        o1 = self.hidden1(x)
        o2 = self.hidden2(o1)
        o3 = self.hidden3(o2)
        o4 = self.hidden4(o3)
        o5 = F.log_softmax(o4, dim=1)
        return o5, (o1, o2, o3, o4)


class CIFAR_9(nn.Module):
    def __init__(self):
        super(CIFAR_9, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(1024 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_10(nn.Module):
    def __init__(self):
        super(CIFAR_10, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(1024 * 3, 32),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(32, 10),
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_11(nn.Module):
    def __init__(self):
        super(CIFAR_11, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_12(nn.Module):
    def __init__(self):
        super(CIFAR_12, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense = nn.Sequential(
            nn.Linear(8 * 4 * 4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_13(nn.Module):
    def __init__(self):
        super(CIFAR_13, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(1024 * 3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(16, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_14(nn.Module):
    def __init__(self):
        super(CIFAR_14, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(1024 * 3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_15(nn.Module):
    def __init__(self):
        super(CIFAR_15, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(1024 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_16(nn.Module):
    def __init__(self):
        super(CIFAR_16, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(1024 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.hidden4 = nn.Sequential(
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = F.log_softmax(x, dim=1)
        return x

    def features(self, x):
        o1 = self.hidden1(x)
        o2 = self.hidden2(o1)
        o3 = self.hidden3(o2)
        o4 = self.hidden4(o3)
        o5 = F.log_softmax(o4, dim=1)
        return o5, (o1, o2, o3, o4)


class CIFAR_17(nn.Module):
    def __init__(self):
        super(CIFAR_17, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(8 * 4 * 4, 32),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def features(self, x, retain_grad=True):
        x = x.reshape(x.shape[0], 3, 32, 32)
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        o3 = self.conv3(o2)
        if retain_grad:
            o3.retain_grad()
        o4 = o3.view(o3.shape[0], -1)
        o5 = self.dense1(o4)
        o6 = self.dense2(o5)
        o7 = F.log_softmax(o6, dim=1)
        return o7, (o1, o2, o3, o4, o5, o6)

    def cnn_encoding(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        o3 = self.conv3(o2)
        return o3, (o1, o2)

    def dense(self, x):
        x = x.view(x.shape[0], -1)
        o5 = self.dense1(x)
        o6 = self.dense2(o5)
        o7 = F.log_softmax(o6, dim=1)
        return o7, (o5, o6)


class CIFAR_18(nn.Module):
    def __init__(self):
        super(CIFAR_18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dense1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_19(nn.Module):
    def __init__(self):
        super(CIFAR_19, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dense1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.max_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_20(nn.Module):
    def __init__(self):
        super(CIFAR_20, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_20_1(nn.Module):
    def __init__(self):
        super(CIFAR_20_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 17, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(17, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_20_2(nn.Module):
    def __init__(self):
        super(CIFAR_20_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 17, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(17, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_20_3(nn.Module):
    def __init__(self):
        super(CIFAR_20_3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 9, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(144, 16),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_21(nn.Module):
    def __init__(self):
        super(CIFAR_21, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_22(nn.Module):
    def __init__(self):
        super(CIFAR_22, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 48, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(48, 6, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(6, 6, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(96, 16),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_23(nn.Module):
    def __init__(self):
        super(CIFAR_23, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 12, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 12, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(128, 18),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(18, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_11_1(nn.Module):
    def __init__(self):
        super(CIFAR_11_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense = nn.Sequential(
            nn.Linear(16 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_11_2(nn.Module):
    def __init__(self):
        super(CIFAR_11_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 48, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(48, 24, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(24, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense = nn.Sequential(
            nn.Linear(16 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_11_3(nn.Module):
    def __init__(self):
        super(CIFAR_11_3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense = nn.Sequential(
            nn.Linear(16 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_11_4(nn.Module):
    def __init__(self):
        super(CIFAR_11_4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense = nn.Sequential(
            nn.Linear(16 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_11_5(nn.Module):
    def __init__(self):
        super(CIFAR_11_5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 12, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 12, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 12, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense = nn.Sequential(
            nn.Linear(12 * 4 * 4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_11_6(nn.Module):
    def __init__(self):
        super(CIFAR_11_6, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense = nn.Sequential(
            nn.Linear(16 * 4 * 4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_11_6_1(nn.Module):
    def __init__(self):
        super(CIFAR_11_6_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 17, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(17, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense = nn.Sequential(
            nn.Linear(16 * 4 * 4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_11_6_2(nn.Module):
    def __init__(self):
        super(CIFAR_11_6_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 17, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(17, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense = nn.Sequential(
            nn.Linear(16 * 4 * 4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_11_6_3(nn.Module):
    def __init__(self):
        super(CIFAR_11_6_3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 17, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense = nn.Sequential(
            nn.Linear(17 * 4 * 4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_17_1(nn.Module):
    def __init__(self):
        super(CIFAR_17_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 9, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(9, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(8 * 4 * 4, 32),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_17_2(nn.Module):
    def __init__(self):
        super(CIFAR_17_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 9, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(9, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(8 * 4 * 4, 32),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_17_3(nn.Module):
    def __init__(self):
        super(CIFAR_17_3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 9, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(9 * 4 * 4, 32),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_17_X(nn.Module):
    def __init__(self, l1=8, l2=8, l3=8):
        super(CIFAR_17_X, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, l1, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(l1, l2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(l2, l3, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(l3 * 4 * 4, 32),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.log_softmax(x, dim=1)
        return x


class CIFAR_24(nn.Module):
    def __init__(self):
        super(CIFAR_24, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 9, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(9, 9, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(9, 9, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(9 * 4 * 4, 32),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def features(self, x, retain_grad=True):
        x = x.reshape(x.shape[0], 3, 32, 32)
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        o3 = self.conv3(o2)
        if retain_grad:
            o3.retain_grad()
        o4 = o3.view(o3.shape[0], -1)
        o5 = self.dense1(o4)
        o6 = self.dense2(o5)
        o7 = F.log_softmax(o6, dim=1)
        return o7, (o1, o2, o3, o4, o5, o6)

    def cnn_encoding(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        o3 = self.conv3(o2)
        return o3, (o1, o2)

    def dense(self, x):
        x = x.view(x.shape[0], -1)
        o5 = self.dense1(x)
        o6 = self.dense2(o5)
        o7 = F.log_softmax(o6, dim=1)
        return o7, (o5, o6)


class CIFAR_24_1(nn.Module):
    def __init__(self):
        super(CIFAR_24_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 10, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 9, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(9, 9, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(9 * 4 * 4, 32),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def features(self, x, retain_grad=True):
        x = x.reshape(x.shape[0], 3, 32, 32)
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        o3 = self.conv3(o2)
        if retain_grad:
            o3.retain_grad()
        o4 = o3.view(o3.shape[0], -1)
        o5 = self.dense1(o4)
        o6 = self.dense2(o5)
        o7 = F.log_softmax(o6, dim=1)
        return o7, (o1, o2, o3, o4, o5, o6)

    def cnn_encoding(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        o3 = self.conv3(o2)
        return o3, (o1, o2)

    def dense(self, x):
        x = x.view(x.shape[0], -1)
        o5 = self.dense1(x)
        o6 = self.dense2(o5)
        o7 = F.log_softmax(o6, dim=1)
        return o7, (o5, o6)


class CIFAR_24_2(nn.Module):
    def __init__(self):
        super(CIFAR_24_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 9, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(9, 10, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 9, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(9 * 4 * 4, 32),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def features(self, x, retain_grad=True):
        x = x.reshape(x.shape[0], 3, 32, 32)
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        o3 = self.conv3(o2)
        if retain_grad:
            o3.retain_grad()
        o4 = o3.view(o3.shape[0], -1)
        o5 = self.dense1(o4)
        o6 = self.dense2(o5)
        o7 = F.log_softmax(o6, dim=1)
        return o7, (o1, o2, o3, o4, o5, o6)

    def cnn_encoding(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        o3 = self.conv3(o2)
        return o3, (o1, o2)

    def dense(self, x):
        x = x.view(x.shape[0], -1)
        o5 = self.dense1(x)
        o6 = self.dense2(o5)
        o7 = F.log_softmax(o6, dim=1)
        return o7, (o5, o6)


class CIFAR_24_3(nn.Module):
    def __init__(self):
        super(CIFAR_24_3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 9, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(9, 9, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(9, 10, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(inplace=True)
        )
        self.dense2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def features(self, x, retain_grad=True):
        x = x.reshape(x.shape[0], 3, 32, 32)
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        o3 = self.conv3(o2)
        if retain_grad:
            o3.retain_grad()
        o4 = o3.view(o3.shape[0], -1)
        o5 = self.dense1(o4)
        o6 = self.dense2(o5)
        o7 = F.log_softmax(o6, dim=1)
        return o7, (o1, o2, o3, o4, o5, o6)

    def cnn_encoding(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        o3 = self.conv3(o2)
        return o3, (o1, o2)

    def dense(self, x):
        x = x.view(x.shape[0], -1)
        o5 = self.dense1(x)
        o6 = self.dense2(o5)
        o7 = F.log_softmax(o6, dim=1)
        return o7, (o5, o6)