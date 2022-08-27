import matplotlib.pyplot as plt
import torch

def loss_plot(filename):
    y = []
    enc = torch.load(filename)
    tempy = list (enc)
    y += tempy
    z = []
    for i in range(10):
        temp = 0
        for j in range(235):
            temp += y[j+i*235]
        z.append(temp)

    x = range(len(z))

    plt.plot(x,z,".-")
    plt_title = 'BATCH_SIZE = 256'
    plt.title(plt_title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(filename)

if __name__ == "__main__":
    loss_plot("epoch_loss")
