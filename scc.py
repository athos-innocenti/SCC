import sys
import random
import numpy as np
import matplotlib.pyplot as plt

sys.setrecursionlimit(1000)
time = 0


class Node:
    def __init__(self, value):
        self.identifier = value
        self.d = 0
        self.f = 0
        self.pi = None
        self.color = 'white'


def dfs_transpose(matrix, nodes, nodes_ordered):
    for u in range(len(nodes)):
        nodes[u].color = 'white'
        nodes[u].pi = None
    global time
    time = 0
    count_scc = 0
    for u in range(len(nodes_ordered)):
        if nodes[nodes_ordered[u].identifier].color == 'white':
            dfs_visit(matrix, nodes, nodes_ordered[u].identifier)
            count_scc += 1
    return count_scc


def dfs(matrix, nodes):
    for u in range(len(nodes)):
        nodes[u].color = 'white'
        nodes[u].pi = None
    global time
    time = 0
    count_radix_df = 0
    for u in range(len(nodes)):
        if nodes[u].color == 'white':
            count_radix_df += 1
            dfs_visit(matrix, nodes, u)
    return count_radix_df


def dfs_visit(matrix, nodes, u):
    global time
    time += 1
    nodes[u].d = time
    nodes[u].color = 'gray'
    for v in range(len(nodes)):
        if matrix[u][v] != 0 and nodes[v].color == 'white':
            nodes[v].pi = nodes[u]
            dfs_visit(matrix, nodes, v)
    nodes[u].color = 'black'
    time += 1
    nodes[u].f = time


def scc(matrix, nodes):
    dfs(matrix, nodes)
    matrix_t = matrix.transpose()
    nodes_ordered = nodes[:]
    nodes_ordered.sort(key=lambda x: x.f)
    nodes_ordered.reverse()
    return dfs_transpose(matrix_t, nodes, nodes_ordered)


def dfs_test(matrix, nodes):
    count_radix = dfs(matrix, nodes)
    print "Numero radici:", count_radix
    return count_radix


def scc_test(matrix, nodes):
    count_scc = scc(matrix, nodes)
    print "Numero SCC:", count_scc
    return count_scc


def adiacent_matrix_creation(size, prob):
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i != j and random.randint(1, 100) <= prob:  # non ci possono essere cappi
                matrix[i][j] = 1
    return matrix


def average(values):
    values_sum = 0
    for i in range(len(values)):
        values_sum += values[i]
    avg = values_sum / len(values)
    return avg


def main():
    tries = 20
    max_prob = 101

    for size in (25, 100, 500):
        radix_dfs = []
        count_scc = []
        nodes = []
        prob_vect = np.arange(0, max_prob)
        for i in range(size):
            nodes.append(Node(i))
        for prob in range(0, max_prob):
            print "Probabilita:", prob
            tries_dfs = []
            tries_scc = []
            for j in range(0, tries):
                matrix = adiacent_matrix_creation(size, prob)
                matrix_scc = matrix[:]
                tries_dfs.append(dfs_test(matrix, nodes))
                tries_scc.append(scc_test(matrix_scc, nodes))
            radix_dfs.append(average(tries_dfs))
            count_scc.append(average(tries_scc))
        plt.plot(prob_vect, radix_dfs)
        plt.xlabel('Probabilita di avere un arco')
        plt.ylabel('Numero di radici nel primo attraversamento')
        plt.show()
        plt.plot(prob_vect, count_scc)
        plt.xlabel('Probabilita di avere un arco')
        plt.ylabel('Numero SCC')
        plt.show()
        # grafo da 25 centrato in [0,100] - grafo da 100 in [0,50] - grafo da 500 in [0,25]
        max_prob = max_prob/2 + 1


if __name__ == '__main__':
    main()
