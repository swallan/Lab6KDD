import sys

import numpy as np

def load_data(fileName):
    conversions = dict()
    idx = 1
    outData = []
    with open(fileName, 'r') as f:
        for ln in f.readlines():
            arr = ln.strip().replace('"', '').split(',')
            print(arr)
            if arr[0] in conversions.keys():
                arr[0] = conversions[arr[0]]
            else:
                conversions[arr[0]] = idx
                idx += 1
                arr[0] = conversions[arr[0]]

            if arr[2] in conversions.keys():
                arr[2] = conversions[arr[2]]
            else:
                conversions[arr[2]] = idx
                idx += 1
                arr[2] = conversions[arr[2]]
            outData.append(arr[:4])
    print(outData)
    return np.asarray(outData, dtype=int), conversions

def PageRank2(V):
    return np.ones(V.shape[0]) / V.shape[0]

from numpy.linalg import norm
def pageRank(G, d, eps):
    n = G.shape[0]
    ranks = np.ones(n) / n
    # M_weighted = M
    # M_weighted[M_weighted > 0] = 1 / M_weighted[M_weighted > 0]
    r = 0
    constant = (1 - d) * (1 / G.shape[0])
    # constant2 = (1 - d) / G.shape[0])
    # GT = G.T[...,::-1][::-1]
    while True:
        r = r + 1
        print(r)
        ranksP = (G @ ranks * d) + constant
        if norm(ranksP - ranks) < eps**12:
            return ranksP
        else:
            ranks = ranksP
        if r > 10:
            print("maxItr: " + str(r*3))
            return(ranksP)

if __name__ == '__main__':
    dataSource = sys.argv[1]
    d = float(sys.argv[2])
    eps = float(sys.argv[3])
    print(f"Running PageRank with {dataSource}")
    V, conversions = load_data(dataSource)

    allNodes = np.unique(np.concatenate((V[..., 0], V[..., 2])))


    matrix = np.zeros((allNodes.shape[0], allNodes.shape[0]))
    for i, _, j, _ in V:
        matrix[j - 1, i - 1] += 1

    counts = np.count_nonzero(matrix, axis=0)
    counts[counts == 0] = -1
    matrix = matrix * (1/counts)
    print(matrix)
    pageRanki = pageRank(matrix, d, eps)


    sort_idx = np.argsort(pageRanki)[::-1]
    inv_map = {v: k for k, v in conversions.items()}
    ranks_sorted = pageRanki[sort_idx]
    names = [inv_map[x] for x in allNodes[sort_idx]]
    for i, name in enumerate(names[:10]):
        print(f"{ranks_sorted[i]:<.08f} : {name}")
    print(pageRanki)