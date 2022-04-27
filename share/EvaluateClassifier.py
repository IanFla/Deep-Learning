import torch


def accuracy(y_hat, y):
    y_hat = y_hat.argmax(axis=1)
    return float((y_hat.type(y.dtype) == y).sum())


def evaluate_accuracy(md, iter_data):
    if isinstance(md, torch.nn.Module):
        md.eval()

    metric = [0.0, 0.0]
    for X, y in iter_data:
        metric[0] += accuracy(md(X), y)
        metric[1] += y.numel()

    return metric[0] / metric[1]


def main():
    pass


if __name__ == '__main__':
    main()
