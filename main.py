import Font
import Net
from PIL import Image
from string import ascii_uppercase

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
                outputVal.append(0)

        index += 1
        trainValues[char] = outputVal

    return trainValues

def whatChar(netOutput):

    positionOfBit = 0

    s = ""



    for bit in netOutput:
        if bit > .7:
            s += "(" + ascii_uppercase[positionOfBit] + ":" + str(bit) + ")"

        positionOfBit += 1

    return s + "\n"

font = Font.Font("fontLibrary/col.png", 2, 13, 15)

net = Net.Net([255, 26, 26])
