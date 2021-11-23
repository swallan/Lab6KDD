import sys
from scipy.sparse import csr_matrix as csr
import numpy as np


def load_data(fileName):
    conversions = dict()
    idx = 0
    outData = []
    with open(fileName, 'r') as f:
        for ln in f.readlines():
            arr = ln.strip().replace('"', '').split(',')
            arr = [x.replace(' ', '') for x in arr]
            if "football" in fileName:
                arr[0], arr[2] = arr[2], arr[0]
                # need to swap the order since WINNING teams should come send

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
    return np.asarray(outData, dtype=int), conversions


def PageRank2(V):
    return np.ones(V.shape[0]) / V.shape[0]


from numpy.linalg import norm


def pageRank(G: np.ndarray, d: float, eps: float) -> (list, int):
    n = G.shape[0]
    ranks = np.ones(n) / n
    r = 0
    constant = ((1 - d) / G.shape[0])
    while True:
        r += 1
        ranksP = ((G @ ranks) * d) + constant
        if norm(ranks - ranksP) < eps:
            return ranksP, r
        else:
            ranks = ranksP


if __name__ == '__main__':
    dataSource = sys.argv[1]
    d = float(sys.argv[2])
    eps = float(sys.argv[3])
    s = (f"""Running PageRank with {dataSource}
d={d}, eps={eps}\n""")
    V, conversions = load_data(dataSource)

    allNodes = np.unique(np.concatenate((V[..., 0], V[..., 2])))
    nNodes = len(allNodes)
    matrix = np.zeros((allNodes.shape[0], allNodes.shape[0]))


    csr_m = csr((np.ones(len(V)), (V[...,2], V[...,0])), shape=(nNodes, nNodes))

    # for i, _, j, _ in V:
    #     matrix[j, i] += 1

    counts = np.count_nonzero(matrix, axis=0)
    counts[counts == 0] = 1
    csr_m = csr_m.multiply(1 / counts)
    pageRanki, niterations = pageRank(matrix, d, eps)

    sort_idx = np.argsort(pageRanki)[::-1]
    inv_map = {v: k for k, v in conversions.items()}
    ranks_sorted = pageRanki[sort_idx]
    names = [inv_map[x] for x in allNodes[sort_idx]]
    s += f"After n={niterations} iterations:\n"
    for i, name in enumerate(names[:10]):
        s += f"{ranks_sorted[i]:<.08f} : {name} {i}\n"
    s += f"np.sum(pageRanks) = {np.sum(pageRanki):.10f}\n"
    print(s)
    with open(f"out/pageRank_d{d}_eps{eps}_dense.txt", 'w') as f:
        f.write(s)
