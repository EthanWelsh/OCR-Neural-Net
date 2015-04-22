import Font
import Net
from string import ascii_uppercase
import random
import sys


def printList(list):
    ret = ""
    for i in list:
        ret += "{0:.1f}".format(i) + " "

    return ret

def trainNet(net, inputValues, expectedOutputValues):
    net.feedForward(inputValues)
    net.backProp(expectedOutputValues)


def getTrainingData():
    trainValues = {}
    index = 0

    for char in ascii_uppercase:
        bm = font.bitmap[char]
        outputVal = []

        for i in range(26):
            if i == index:
                outputVal.append(1)
            else:
                outputVal.append(-1)

        index += 1
        trainValues[char] = outputVal

    return trainValues


def whatChar(netOutput):

    positionOfBit = 0

    s = ""
    for bit in netOutput:
        if bit > .7:
            s += "(" + ascii_uppercase[positionOfBit] + ":" + "{0:.2f}".format(bit) + ")"

        positionOfBit += 1

    return s


def randomLetter():
    return random.choice(ascii_uppercase)


font = Font.Font("fontLibrary/col.png", 2, 13, 15)
outputValues = getTrainingData()

net = Net.Net([225, 26, 26])

i = 0

print ((("-"*9) + "|") * 10)

samples = 100000



for i in range(samples):

    if i % (samples * .01) == 0 and i != 0:
        sys.stdout.write('=')

    randLetter = randomLetter()
    trainNet(net, font.bitmap[randLetter], outputValues[randLetter])

sys.stdout.write('=')

print

for char in ascii_uppercase:
    net.feedForward(font.bitmap[char])
    print char + ": " + whatChar(net.getResults())