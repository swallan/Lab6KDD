import sys
import time

from scipy.sparse import csr_matrix as csr
from numpy.linalg import norm
import numpy as np

def load_data(fileName: str):
    t0 = time.time()
    print("Loading data...", end="")
    conversions = dict()
    idx = 0
    outData = []
    with open(fileName, 'r') as f:
        for ln in f.readlines():
            idx += 1
            if idx % 1000000 == 0:
                print(f"{idx / 68993773 * 100:.04f}%")
            if '#' not in ln:
                arr = ln.strip().replace('"', '').split(',')
                arr = [x.replace(' ', '') for x in arr]
                if ("wiki-Vote.txt" in fileName or "p2p-Gnutella05" in fileName
                        or 'amazon' in fileName
                        or 'soc-sign-Slashdot081106' in fileName
                        or 'LiveJournal1' in fileName
                        ):
                    l = ln.replace("\n", '').split('\t')
                    arr = [int(l[0]), 0, int(l[1]), 0]

                if "football" in fileName:
                    arr[0], arr[2] = arr[2], arr[0]
                    # need to swap the order since WINNING teams should come send

                # if arr[0] in conversions.keys():
                #     arr[0] = conversions[arr[0]]
                # else:
                #     conversions[arr[0]] = idx
                #     idx += 1
                #     arr[0] = conversions[arr[0]]
                #
                # if arr[2] in conversions.keys():
                #     arr[2] = conversions[arr[2]]
                # else:
                #     conversions[arr[2]] = idx
                #     idx += 1
                #     arr[2] = conversions[arr[2]]
                conversions[arr[2]] = arr[2]
                conversions[arr[0]] = arr[0]
                outData.append(arr[:4])
            else:
                print(ln, end="")

    print("done.")
    return np.asarray(outData, dtype=int), conversions, time.time() - t0


def pageRank(G: np.ndarray, d: float, eps: float) -> (list, int):
    print("Running pagerank....", end="")
    n = G.shape[0]
    ranks = np.ones(n) / n
    r = 0
    constant = ((1 - d) / G.shape[0])
    t0 = time.time()
    while True:
        if r % 100 == 0:
            print(f"iteration: {r}")
        r += 1
        ranksP = ((G @ ranks) * d) + constant
        if norm(ranks - ranksP) < eps:
            print(f"done. time elapsed: {time.time() - t0:.08f}")
            return ranksP, r, time.time() - t0
        else:
            ranks = ranksP


if __name__ == '__main__':
    dataSource = sys.argv[1]
    d = float(sys.argv[2])
    eps = float(sys.argv[3])
    s = (f"""Running PageRank with {dataSource}
d={d}, eps={eps}\n""")
    V, conversions, dataLoadTime = load_data(dataSource)

    allNodes = np.unique(np.concatenate((V[..., 0], V[..., 2])))
    nNodes = len(allNodes)

    # form sparse matrix. Format: data is all ones,
    # data are from input with (data, (row_idx, col_idx))), then shape
    csr_m = csr((np.ones(len(V)), (V[..., 2], V[..., 0])),
                shape=(nNodes, nNodes))

    # calculate the counts per column, then adjust the probabilities
    # inside the matrix
    sp_matrix = csr_m.tocoo()
    col_ind = sp_matrix.col
    counts, unqiues = np.unique(col_ind, return_counts=1)
    all_ones = np.ones(sp_matrix.shape[0])
    all_ones[counts] = unqiues
    csr_m = csr_m.multiply(1 / all_ones)

    # calculate pagerank
    pageRanki, niterations, processingTime = pageRank(csr_m, d, eps)

    # print results
    sort_idx = np.argsort(pageRanki)[::-1]
    inv_map = {v: k for k, v in conversions.items()}
    ranks_sorted = pageRanki[sort_idx]
    names = [inv_map[x] for x in allNodes[sort_idx]]
    s += f"readTime: {dataLoadTime:.02f}s, processTime: {processingTime:.02f}s\n"
    s += f"After n={niterations} iterations:\n"
    for i, name in enumerate(names[:10]):
        s += f"{ranks_sorted[i]:<.08f} : {name} {i}\n"
    s += f"np.sum(pageRanks) = {np.sum(pageRanki):.10f}\n"
    print(s)
    with open(
            f"out/pageRank_d{d}_eps{eps}_sparse_{dataSource.split('/')[-1].replace('.txt', '')}.txt",
            'w') as f:
        f.write(s)
