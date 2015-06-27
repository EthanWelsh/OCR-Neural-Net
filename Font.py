from string import ascii_uppercase

from PIL import Image


class Font:
    def __init__(self, fname, chars_per_row, chars_per_col, size):

        self.chars_per_row = chars_per_row
        self.chars_per_col = chars_per_col

        self.size = size

        self.im = Image.open(fname)
        fonts = []

        for c in range(self.chars_per_col):
            for r in range(self.chars_per_row):
                fonts.append(self._get_letter(r, c).resize((size, size)))

        self.font = {}

        i = 0

        for char in ascii_uppercase:
            self.font[char] = fonts[i]
            i += 1

        self.bitmap = {}

        for char in ascii_uppercase:
            letterImg = self.get_letter(char)
            self.bitmap[char] = self._get_bitmap(letterImg)

    def _get_bitmap(self, img):

        bitmap = []

        pix = img.load()
        for r in range(self.size):
            for c in range(self.size):
                gray_scale_sum = pix[c, r][0] + pix[c, r][1] + pix[c, r][2]
                if gray_scale_sum != 255 * 3:
                    bitmap.append(1)
                else:
                    bitmap.append(0)
        return bitmap

    def get_letter(self, letter):
        return self.font[letter]

    def _get_letter(self, row, col):

        w, h = self.im.size

        if row >= self.chars_per_row or col >= self.chars_per_col:
            print "ERROR"
            return -1

        width_per_char = w / self.chars_per_row
        height_per_char = h / self.chars_per_col

        start_x = width_per_char * row
        start_y = height_per_char * col

        a = (start_x, start_y, start_x + width_per_char, start_y + height_per_char)
        return self.im.crop(a)
