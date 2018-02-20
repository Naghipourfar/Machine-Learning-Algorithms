import sys

import matplotlib.pyplot as plt

sys.setrecursionlimit(10000000)

"""
    Created by Mohsen Naghipourfar on 2/14/18.
    Email : mn7697np@gmail.com
"""


def scatter(x, y, linewidths=3, color='black', label=None):
    plt.scatter(x, y, linewidths=linewidths, color=color, label=label)


def plot(x, y=None, linewidths=3, color='blue', label=None, *args):
    if y is None:
        plt.plot(x, linewidths=linewidths, color=color, label=label, *args)
    else:
        plt.plot(x, y, linewidth=linewidths, color=color, label=label, *args)


def axis_ticks():
    plt.xticks()
    plt.yticks()


def show():
    axis_ticks()
    plt.legend()
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
