from PIL import Image
from string import ascii_uppercase

class Font:

    def __init__(self, fname, charsPerRow, charsPerCol, size):

        self.charsPerRow = charsPerRow
        self.charsPerCol = charsPerCol

        self.size = size

        self.im = Image.open(fname)
        fontArr = []

        for c in range(self.charsPerCol):
            for r in range(self.charsPerRow):
                fontArr.append(self._getLetter(r, c).resize((size, size)))

        self.font = {}

        i = 0

        for char in ascii_uppercase:
            self.font[char] = fontArr[i]
            i+=1

        self.bitmap = {}

        for char in ascii_uppercase:
            letterImg = self.getLetter(char)
            self.bitmap[char] = self._getBitmap(letterImg)

    def _getBitmap(self, img):

        bitmap = []

        pix = img.load()
        for r in range(self.size):
            for c in range(self.size):

                sum = pix[c, r][0] + pix[c, r][1] + pix[c, r][2]
                if sum != 255*3:
                    bitmap.append(1)
                else:
                    bitmap.append(0)

        return bitmap

    def getLetter(self, letter):
        return self.font[letter]

    def _getLetter(self, row, col):

        w, h = self.im.size

        if row >= self.charsPerRow or col >= self.charsPerCol:
            print "ERROR"
            return -1

        widthPerChar = w/self.charsPerRow
        heightPerChar = h/self.charsPerCol

        startX = widthPerChar * row
        startY = heightPerChar * col

        a = (startX, startY, startX + widthPerChar, startY + heightPerChar)
        return self.im.crop(a)