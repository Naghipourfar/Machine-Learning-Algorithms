import sys

import matplotlib.pyplot as plt

sys.setrecursionlimit(10000000)

"""
    Created by Mohsen Naghipourfar on 2/14/18.
    Email : mn7697np@gmail.com
"""


def scatter(x, y, linewidths=3, color='black'):
    plt.scatter(x, y, linewidths=linewidths, color=color)


def plot(x, y, linewidths=3, color='blue'):
    plt.plot(x, y, linewidth=linewidths, color=color)


def axis_ticks():
    plt.xticks()
    plt.yticks()


def show():
    axis_ticks()
    plt.show()


if __name__ == '__main__':
    ax = plt.gca()
    # ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()