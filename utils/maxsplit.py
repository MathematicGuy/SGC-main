import numpy as np
import networkx as nx
import itertools 

def _normalize_edges(n, edges, weights=None):
    if weights is None:
        weights = [1]*len(edges)
    es = []
    for (u,v), w in zip(edges, weights):
        if u == v: continue
        if u > v: u, v = v, u
        es.append((u, v, w))
    return es

def _cut_value(assign, edges):
    val = 0
    for u, v, w in edges:
        if assign[u] != assign[v]:
            val += w
    return val

def max_cut_split(n, edges, weights=None):

    es = _normalize_edges(n, edges, weights)
    best_val, best_assign = -1, None
    
    if n % 2 != 0:
        raise ValueError("")
    
    num_in_group_0 = n // 2
    
    nodes_to_choose_from = range(1, n)
    num_to_choose = num_in_group_0 - 1

    for chosen_nodes in itertools.combinations(nodes_to_choose_from, num_to_choose):
        assign = [1] * n  
        assign[0] = 0     
        for node_idx in chosen_nodes:
            assign[node_idx] = 0 
        
        val = _cut_value(assign, es)
        if val > best_val:
            best_val, best_assign = val, assign[:]
            
    if best_assign is None and n > 0:
        best_assign = [0] * (n//2) + [1] * (n - n//2)
        
    return best_val, best_assign