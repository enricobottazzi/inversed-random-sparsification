import math
import numpy as np
import networkx as nx

def obfuscate_graph_random_sparsification(g, p):

    assert 0 <= p <= 1

    gp = g.copy()
    print(gp.edges())

    for edge in g.edges():
        print(edge)
        be = np.random.binomial(1, p)
        if be == 1:
            gp.remove_edge(*edge)


    return gp

def verify_obfuscation(g, gp, p, k): 

    n = len(g.nodes())

    # Complexity O(n)
    # A is an array of n arrays where A[0] is an array containing the indices of vertices that have degree 0 
    # and A[n-1] is an array containing the indices of vertices that have degree n-1 
    A = [[] for _ in range(n)]
    B = [[] for _ in range(n)]

    g_adj_matrix = nx.to_numpy_array(g)
    gp_adj_matrix = nx.to_numpy_array(gp)

    for i in range(n):
        degree_a = int(np.sum(g_adj_matrix[i]))
        degree_b = int(np.sum(gp_adj_matrix[i]))
        A[degree_a].append(i)
        B[degree_b].append(i)

    F = np.zeros((n, n), dtype=float)
    
    # Complexity is O(A x B), namely number of degree pairs in the cartesian product A x B
    for a in range(n):
        for b in range(n):
            if len(A[a]) == 0 or len(B[b]) == 0:
                continue
            if b > a:
                pr = 0
            else:
                pr = math.comb(a, b) * (1 - p)**b * p**(a - b)
            for i in A[a]:
                for j in B[b]:
                    F[i, j] = pr
    
    ko = float('inf')
    
    # Complexity is O(A x B)
    # each row is a vertex v in g
    for v in range(n):
        f_v = sum(F[v])
        row_entropy = 0
        for u in range(n):
            x_vu = F[u, v] / f_v
            if x_vu > 0:
                row_entropy -= x_vu * math.log2(x_vu)
        k = 2**row_entropy
        if k < ko:
            ko = k

    return ko < k, ko

g = nx.Graph()
g.add_node(0)
g.add_node(1)
g.add_node(2)
g.add_node(3)
g.add_node(4)

g.add_edge(4, 0)
g.add_edge(4, 1)
g.add_edge(4, 2)
g.add_edge(4, 3)
g.add_edge(4, 5)
g.add_edge(4, 6)

p = 0.20
gp = obfuscate_graph_random_sparsification(g, p)

k = 1
_, ko = verify_obfuscation(g, gp, p, k)
print("ko", ko)