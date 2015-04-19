from Net import *
import random


def trainNet(net, inputValues, expectedOutputValues):
    net.feedForward(inputValues)
    net.backProp(expectedOutputValues)


def trainII(net):
    ins = list()
    outs = list()
    ins.append(1)
    ins.append(1)
    outs.append(0)
    trainNet(net, ins, outs)


def trainOO(net):
    ins = list()
    outs = list()
    ins.append(0)
    ins.append(0)
    outs.append(0)
    trainNet(net, ins, outs)


def trainOI(net):
    ins = list()
    outs = list()
    ins.append(0)
    ins.append(1)
    outs.append(1)
    trainNet(net, ins, outs)


def trainIO(net):
    ins = list()
    outs = list()
    ins.append(1)
    ins.append(0)
    outs.append(1)
    trainNet(net, ins, outs)


def printResults(net):
    ins = list()
    ins.append(1)
    ins.append(0)
    net.feedForward(ins)
    print("1 ^ 0 == 1 ==", net.getResults().getOutputs()[0])

    ins = list()
    ins.append(0)
    ins.append(1)
    net.feedForward(ins)
    print("0 ^ 1 == 1 ==", net.getResults().getOutputs()[0])

    ins = list()
    ins.append(1)
    ins.append(1)
    net.feedForward(ins)
    print("1 ^ 1 == 0 ==", net.getResults().getOutputs()[0])

    ins = list()
    ins.append(0)
    ins.append(0)
    net.feedForward(ins)
    print("0 ^ 0 == 0 ==", net.getResults().getOutputs()[0])


topo = list()
topo.append(2)
topo.append(3)
topo.append(4)
topo.append(1)
net = Net(topo)
print(net)

for i in range(0, 500):
    r = random.randint(0, 3)
    if r == 0:
        trainII(net)
    elif r == 1:
        trainIO(net)
    elif r == 2:
        trainOI(net)
    else:
        trainOO(net)

printResults(net)
