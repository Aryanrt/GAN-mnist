from gettext import lngettext
import numpy as np
import matplotlib.pyplot as plt

with open('cgan/output.log') as f:
    lines = f.readlines()
    accuracy = [0]*len(lines)
    for i in range(0, len(lines)):
        x = lines[i]
        acc=""
        for j in range(x.index('[')+9, x.index(',')):
            acc = acc+x[j]

        accuracy[i]=float(acc)
        

x=range(0,10000)
plt.title("CGAN")
plt.ylabel("Discriminator Accuracy")
plt.xlabel("Epoch")
plt.scatter(x, accuracy,s=0.1)
plt.savefig('cgan-accuracy.png')