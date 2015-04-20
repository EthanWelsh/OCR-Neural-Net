from Net import *
import random

o = [1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 1]

x = [1, 0, 0, 0, 1,
     0, 1, 0, 1, 0,
     0, 0, 1, 0, 0,
     0, 1, 0, 1, 0,
     1, 0, 0, 0, 1]

def trainNet(net, inputValues, expectedOutputValues):
    net.feedForward(inputValues)
    net.backProp(expectedOutputValues)

def printResults(net):

    net.feedForward(o)
    print "O:", "{0:.10f}".format(net.getResults()[0])

    net.feedForward(x)
    print "X:", "{0:.10f}".format(net.getResults()[0])

net = Net([25, 5, 1])

print(net)

for i in range(0, 100000):
    r = random.randint(0, 1)

    if r == 0:
        trainNet(net, o, [0])
    elif r == 1:
        trainNet(net, x, [1])


    if i % 10000 == 0:
        printResults(net)
        print

print(net)