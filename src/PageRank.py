import sys

import numpy as np

def load_data(fileName):
    conversions = dict()
    idx = 0
    outData = []
    with open(fileName, 'r') as f:
        for ln in f.readlines():
            arr = ln.strip().replace('"', '').split(',')
            print(arr)
            if arr[0] in conversions.keys():
                arr[0] = conversions[arr[0]]
            else:
                conversions[arr[0]] = idx
                idx = idx + 1
                arr[0] = conversions[arr[0]]

            if arr[2] in conversions.keys():
                arr[2] = conversions[arr[2]]
            else:
                conversions[arr[2]] = idx
                idx = idx + 1
                arr[2] = conversions[arr[2]]
            outData.append(arr[:4])
    print(outData)
    return np.asarray(outData, dtype=int), conversions

def PageRank2(V):
    return np.ones(V.shape[0]) / V.shape[0]

from numpy.linalg import norm
def pageRank(M, d, eps):
    n = M.shape[0]
    ranks = np.ones(n) / n
    M_weighted = (M * d)
    M_weighted[M_weighted > 0] = M_weighted[M_weighted > 0] + ((1 - d) / n)
    r = 0
    while True:
        r = r + 1
        print(r)
        ranksP = np.dot(M_weighted ,ranks)
        if norm(ranksP - ranks) < eps:
            return ranksP
        else:
            ranks = ranksP
        if r > len(ranksP) * 3:
            print("maxItr: " + str(r*3))
            return norm( ranks - ranksP)

if __name__ == '__main__':
    dataSource = sys.argv[1]
    d = float(sys.argv[2])
    eps = float(sys.argv[3])
    print(f"Running PageRank with {dataSource}")
    V, conversions = load_data(dataSource)
    # V = np.asarray([
    #     [1, 0, 2, 0],
    #     [1, 0, 3, 0],
    #     [1, 0, 4, 0],
    #     [2, 0, 1, 0],
    #     [2, 0, 4, 0],
    #     [3, 0, 4, 0],
    #     [4, 0, 3, 0],
    #     [4, 0, 2, 0]
    # ])

    # d = .85
    Vcardinality = V.shape[0]
    inverseVcardinality = 1 / Vcardinality
    allNodes, outBoundEdges = np.unique(np.concatenate((V[..., 0], V[..., 2])), return_counts=1)

    pageRanki = PageRank2(allNodes)

    inverseOutBound = 1 / outBoundEdges

    inboundEdges = [0] * allNodes.shape[0]
    # inboundEdgesC = [0] * allNodes.shape[0]
    # for i in range(allNodes.shape[0]):
    #     inboundEdges[i] = (V[V[..., 0] == allNodes[i]][..., 2])
    #     inboundEdgesC[i] = inboundEdges[i].shape[0]
    # inboundEdgesC = np.asarray(inboundEdgesC)
    # inboundEdges = np.asarray(inboundEdges)
    #

    #
    # previousPageRanki = pageRanki[:]
    #
    # while(sum(pageRanki) - )

    matrix = np.zeros((allNodes.shape[0], allNodes.shape[0]))
    for i, _, j, _ in V:
        matrix[j - 1, i - 1] = 1
    matrix  = matrix * (np.divide(1, np.count_nonzero(matrix, axis=0),
                                  out=np.zeros_like(matrix),
                                  where = (np.count_nonzero(matrix, axis=0) > 0)))
    constant1 = (1 - d) * inverseVcardinality
    epsilon = .05
    r = 1
    pageRanki = pageRank(matrix, d, eps)
    # while True:
    #     print(r)
    #     r = r + 1
    #     previousPageRanki = pageRanki[:]
    #     # invert the
    #     # matrix[matrix != 0] = 1 / matrix[matrix != 0]
    #     pageRanki = constant1 + (d * (outBoundEdges * pageRanki))
    #     diff = np.sum((pageRanki - previousPageRanki)) - epsilon
    #     print(np.sum(pageRanki))
    #     if diff < 0:
    #         break

    sort_idx = np.argsort(pageRanki)[::-1]
    inv_map = {v: k for k, v in conversions.items()}
    ranks_sorted = pageRanki[sort_idx]
    names = [inv_map[x] for x in allNodes[sort_idx]]
    for i, name in enumerate(names[:10]):
        print(f"{ranks_sorted[i]:<.02f} : {name}")
    # print(pageRanki)