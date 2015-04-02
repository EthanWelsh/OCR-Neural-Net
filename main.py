from Net import *


def trainNet(net, inputValues, expectedOutputValues):
    net.feedForward(inputValues)
    results = net.getResults()
    net.backProp(expectedOutputValues)

topo = list()
topo.append(2)
topo.append(3)
topo.append(3)
topo.append(1)
net = Net(topo)


for i in range(0, 5000):
    ins = list()
    outs = list()
    ins.append(0)
    ins.append(0)
    outs.append(0)
    trainNet(net, ins, outs)

    ins = list()
    outs = list()
    ins.append(0)
    ins.append(1)
    outs.append(1)
    trainNet(net, ins, outs)

    ins = list()
    outs = list()
    ins.append(1)
    ins.append(0)
    outs.append(1)
    trainNet(net, ins, outs)

    ins = list()
    outs = list()
    ins.append(1)
    ins.append(1)
    outs.append(0)
    trainNet(net, ins, outs)

    ins = list()
    outs = list()
    ins.append(0)
    ins.append(0)
    outs.append(0)
    trainNet(net, ins, outs)

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

