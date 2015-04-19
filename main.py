from Net import *
import random


def trainNet(net, inputValues, expectedOutputValues):
    net.feedForward(inputValues)
    net.backProp(expectedOutputValues)


def trainII(net):
    trainNet(net, [1, 1], [0])


def trainOO(net):
    trainNet(net, [0, 0], [0])


def trainOI(net):
    trainNet(net, [0, 1], [1])


def trainIO(net):
    trainNet(net, [1, 0], [1])


def printResults(net):

    net.feedForward([1, 0])
    print "1 ^ 0 == 1 ==", "{0:.2f}".format(net.getResults()[0])

    net.feedForward([0, 1])
    print "0 ^ 1 == 1 ==", "{0:.2f}".format(net.getResults()[0])

    net.feedForward([1, 1])
    print "1 ^ 1 == 0 ==", "{0:.2f}".format(net.getResults()[0])

    net.feedForward([0, 0])
    print "0 ^ 0 == 0 ==", "{0:.2f}".format(net.getResults()[0])

net = Net([2, 3, 3, 1])

print(net)

for i in range(0, 100000):
    r = random.randint(0, 3)

    if r == 0:      trainII(net)
    elif r == 1:    trainIO(net)
    elif r == 2:    trainOI(net)
    else:           trainOO(net)

    if i % 10000 == 0:
        printResults(net)
        print

print(net)