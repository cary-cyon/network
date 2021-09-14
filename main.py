import NeuralNetwork.Network
import numpy as np
import matplotlib.pyplot as plt


# ункция приводит данне к виду от 0.01 до 1 (изначально от 0 до 255)
def to_normal(array):
    im_array = array[1:]
    im_array /= 255
    im_array *= 0.99
    im_array += 0.01
    return im_array


# считывает павильный результат
def create_right_res(num):
    res = np.zeros(10)
    res += 0.01
    res[int(num)] = 0.99
    return res


def convert_to_np(array):
    value = np.array(array.split(","), dtype="float64")
    return value


def main():
    data = open("data/train.csv", "r")
    data_list = data.readlines()
    data.close()
    data_list = data_list[1:]
    lol = NeuralNetwork.Network.NeuralNetwork(784, 10, 21, 0.3)
    plot = np.zeros(len(data_list))
    j = 0
    for i in data_list:
        value = convert_to_np(i)
        res = create_right_res(value[0])
        im_array = to_normal(value)
        plot[j] = lol.train(im_array, res)
        j += 1

    fig = plt.figure()
    plt.plot(plot)
    plt.show()


if __name__ == "__main__":
    main()
