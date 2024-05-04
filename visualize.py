import numpy as np
from numpy._typing import NDArray
from matplotlib import pyplot as plt
import argparse


def read_matrix(filename: str = 'result.bin') -> NDArray:
    with open(filename, 'rb') as file:
        rows = np.fromfile(file, dtype=np.int32, count=1)[0]
        cols = np.fromfile(file, dtype=np.int32, count=1)[0]
        matrix = np.fromfile(file, dtype=np.uint32).reshape((rows, cols))

    return matrix


def main():
    parser = argparse.ArgumentParser(description='Read and plot a matrix from a binary file.')
    parser.add_argument('-f', '--file', type=str, default='result.bin', help='Path to the matrix file')
    parser.add_argument('-o', '--output', type=str, default='figure.png', help='Path to save the plot')
    parser.add_argument('-t', '--title', type=str, default='Matrix Visualization', help='Title for the plot')

    args = parser.parse_args()

    matrix = read_matrix(args.file)

    plt.imshow(matrix)
    plt.colorbar()
    plt.title(args.title)
    plt.savefig(args.output)
    plt.show()


if __name__ == '__main__':
    main()

