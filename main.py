import Font
import Net
from string import ascii_uppercase
import random
import sys


def print_list(list_to_print):
    ret = ""
    for i in list_to_print:
        ret += "{0:.1f}".format(i) + " "

    return ret


def train_net(net, input_vals, expected_output_values):
    net.feedForward(input_vals)
    net.backProp(expected_output_values)


def get_training_data():
    train_values = {}
    index = 0

    for char in ascii_uppercase:
        output_val = []

        for i in range(26):
            if i == index:
                output_val.append(1)
            else:
                output_val.append(-1)

        index += 1
        train_values[char] = output_val

    return train_values


def what_char(net_Output):

    bit_position = 0

    s = ""
    for bit in net_Output:
        if bit > 0:
            s += "(" + ascii_uppercase[bit_position] + ":" + "{0:.2f}".format(bit) + ")"

        bit_position += 1

    return s


def random_letter():
    return random.choice(ascii_uppercase)


def main():
    font = Font.Font("fontLibrary/col.png", 2, 13, 15)

    outputValues = get_training_data()
    net = Net.Net([225, 150, 26])
    print ((("-"*9) + "|") * 10)

    samples = 1000
    for i in range(samples):
        if i % (samples * .01) == 0 and i != 0:
            sys.stdout.write('=')

        randLetter = random_letter()
        train_net(net, font.bitmap[randLetter], outputValues[randLetter])

    sys.stdout.write('=')

    for char in ascii_uppercase:
        net.feedForward(font.bitmap[char])
        print char + ": " + what_char(net.getResults())


if __name__ == '__main__':
    main()