import torchvision
from torch.utils import data
from torchvision import transforms
from datetime import datetime as dt


def read(size_batch):
    trans = transforms.ToTensor()
    data_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans)
    data_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans)
    iter_train = data.DataLoader(data_train, size_batch, shuffle=True, num_workers=2)
    iter_test = data.DataLoader(data_test, size_batch, shuffle=False, num_workers=2)
    return iter_train, iter_test


def main():
    iter_train, iter_test = read(256)
    t0 = dt.now()
    for X, y in iter_train:
        continue

    print(dt.now() - t0)
    t0 = dt.now()
    for X, y in iter_test:
        continue

    print(dt.now() - t0)


if __name__ == '__main__':
    main()
