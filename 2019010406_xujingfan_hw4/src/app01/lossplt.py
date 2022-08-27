import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import torch

def loss_plot(epochs,filename,batch_size):
    y = []
    enc = torch.load("static/result/"+filename)
    tempy = list (enc)
    y += tempy
    z = []
    for i in range(epochs):
        temp = 0
        for j in range(batch_size):
            temp += y[j+i*batch_size]
        z.append(temp)

    x = range(len(z))

    plt.plot(x,z,".-")
    batch_size=str(batch_size)
    plt_title = 'BATCH_SIZE = '+ batch_size
    plt.title(plt_title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("static/pic/{}".format(filename))
    plt.close()

