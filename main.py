from Net import *
import random

def printList(list):
    ret = ""
    for i in list:
        ret += "{0:.1f}".format(i) + " "

    return ret

a = [0, 0, 1, 0, 0,
     0, 1, 0, 1, 0,
     1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1]

b = [1, 1, 1, 0, 0,
     1, 0, 0, 1, 0,
     1, 0, 1, 0, 0,
     1, 0, 0, 1, 0,
     1, 1, 1, 0, 0]

c = [1, 1, 1, 1, 1,
     1, 0, 0, 0, 0,
     1, 0, 0, 0, 0,
     1, 0, 0, 0, 0,
     1, 1, 1, 1, 1]

d = [1, 0, 0, 0, 0,
     1, 1, 0, 0, 0,
     1, 0, 1, 0, 0,
     1, 0, 1, 0, 0,
     1, 1, 0, 0, 0]

e = [1, 1, 1, 1, 0,
     1, 0, 0, 0, 0,
     1, 1, 1, 0, 0,
     1, 0, 0, 0, 0,
     1, 1, 1, 1, 0]

def trainNet(net, inputValues, expectedOutputValues):
    net.feedForward(inputValues)
    net.backProp(expectedOutputValues)

def printResults(net):

    net.feedForward(a)
    print "A: ", printList(net.getResults())

    net.feedForward(b)
    print "B: ", printList(net.getResults())

    net.feedForward(c)
    print "C: ", printList(net.getResults())

    net.feedForward(d)
    print "D: ", printList(net.getResults())

    net.feedForward(e)
    print "E: ", printList(net.getResults())

net = Net([25, 5, 5])

print(net)

for i in range(0, 1000000):
    r = random.randint(0, 4)

    if r == 0:
        trainNet(net, a, [0, 0, 0, 0, 1])
    elif r == 1:
        trainNet(net, b, [0, 0, 0, 1, 0])
    elif r == 2:
        trainNet(net, c, [0, 0, 1, 0, 0])
    elif r == 3:
        trainNet(net, d, [0, 1, 0, 0, 0])
    elif r == 4:
        trainNet(net, e, [1, 0, 0, 0, 0])

    if i % 10000 == 0:
        printResults(net)
        print

print(net)