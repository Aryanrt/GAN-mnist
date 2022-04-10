from gettext import lngettext
from grpc import protos_and_services
import numpy as np
import matplotlib.pyplot as plt

with open('cgan/output.log') as f:
    lines = f.readlines()
    totalLoss = [0]*len(lines)
    accuracy = [0]*len(lines)
    for i in range(0, len(lines)):
        x = lines[i]
        loss=""
        for j in range(x.index('[')+9, x.index(',')):
            loss = loss+x[j]
        totalLoss[i]=float(loss)

        acc=""
        for j in range(x.index(']')-6, x.index(']')-1):
            acc = acc+x[j]
        
        accuracy[i]=float(acc)
        

x=range(0,10000)
plt.title("CGAN Loss")
plt.ylabel("Discriminator Loss")
plt.xlabel("Epoch")
plt.scatter(x, totalLoss,s=0.1)
plt.savefig('cgan-D-Loss.png')

x=range(2000,5000)
means=1000*[0]
x=[0]*1000

for i in range(0,1000):
    means[i]= np.mean(accuracy[i*10:i*10+9])
    x[i]=i*10

plt.clf()
plt.title("CGAN Accuracy")
plt.ylabel("Discriminator Accuracy")
plt.xlabel("Epoch")
plt.scatter(x, means,s=0.1)
plt.savefig('cgan-accuracy.png')