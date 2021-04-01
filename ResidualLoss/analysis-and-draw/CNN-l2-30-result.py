import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ResidualLoss.dataset import cifar10_data_loader_train
from ResidualLoss.model import CIFAR_24

model = CIFAR_24().cuda()
model.eval()

evaluation_batch_size = 25000
evaluation_data_loader = cifar10_data_loader_train(batch_size=evaluation_batch_size, shuffle=False, loc="../../data")

result_list = list()
for i in range(1, 31):
    correct_list = list()
    with torch.no_grad():
        state_dict = torch.load('../CNN-30/CIFAR_24/iter-%s.pt' % i)
        model.load_state_dict(state_dict)

        start_index = 0
        for data, target in evaluation_data_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)

            pred = output.argmax(dim=1)
            correct = pred.eq(target)
            correct_list.append(correct)
    result_list.append(torch.hstack(correct_list))
    print(i)

torch.save(torch.vstack(result_list), "./data/CNN-30-CIFAR_24-result.pt")

result = torch.load("./data/CNN-30-CIFAR_24-result.pt")
correct_num = result.sum(dim=1)
occur = result.int().sum(dim=0)

lst_2 = list()
for i in range(50000):
    if occur[i] <= 10:
        lst_2.append(i)
#
torch.save(lst_2, "./data/CNN-30-CIFAR_24-lower_10.pt")
print(len(lst_2))