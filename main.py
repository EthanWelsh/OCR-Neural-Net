import Font
from PIL import Image

def printList(list):
    ret = ""
    for i in list:
        ret += "{0:.1f}".format(i) + " "

    return ret

def trainNet(net, inputValues, expectedOutputValues):
    net.feedForward(inputValues)
    net.backProp(expectedOutputValues)

font = Font.Font("fontLibrary/col.png", 2, 13, 15)
