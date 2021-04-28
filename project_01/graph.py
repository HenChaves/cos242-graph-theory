import numpy as np
import os
import pandas as pd
from tqdm.notebook import tqdm
import copy
import sys

class Graph:
    
    def __init__(self, n):
        self.n_nodes = n
        self.matrix = np.zeros((n, n), dtype="uint8")
        self.nodes = [set() for i in range(n)]
#         self.nodes = np.full(n, fill_value=set(), dtype="object")
    
    
    def add_edge(self, v, w):
        
        if v == w:
            self.matrix[v-1, v-1] = 2
            self.nodes[v-1].add(v)
            
        else:
            self.matrix[v-1, w-1] = 1
            self.matrix[w-1, v-1] = 1
            self.nodes[v-1].add(w)
            self.nodes[w-1].add(v)
    
    def get_node(self, v):
        return self.nodes[v-1]
    
    def get_lists(self):
        return self.nodes
    
    def get_node_edges(self):
        return {i+1:self.nodes[i] for i in range(self.n_nodes)}
    
    def get_matrix(self):
        return self.matrix
    
    def get_matrix_beautiful(self):
        return pd.DataFrame(self.matrix, columns=np.arange(1, self.n_nodes+1), index=np.arange(1, self.n_nodes+1))
    
    def sort_neighbors(self):
        self.nodes = [sorted(n) for n in self.nodes]


def open_graph_txt(filename, extra=False):
    with open(filename, "r") as f:
        lines = f.read().split("\n")
        n_nodes = int(lines[0])
        edges = [tuple(map(lambda i: int(i), line.split(" "))) for line in lines[1:-1]]
    
    graph = Graph(n_nodes)
    for v, w in edges:
        graph.add_edge(v, w)
    
    if extra:
        return graph, n_nodes, edges
    
    return graph

def graph_statistics(graph):
    print("Número de vértices:", graph.n_nodes)
    print("Número de arestas:", graph.get_matrix().sum()/2)
    print("Grau mínimo:", graph.get_matrix().sum(axis=0).min())
    print("Grau máximo:", graph.get_matrix().sum(axis=0).max())
    print("Grau médio:", graph.get_matrix().sum(axis=0).mean())
    print("Mediana do Grau:", np.median(graph.get_matrix().sum(axis=0)))

class BFS:
    def __init__(self, graph, root):
        self.graph = graph
        self.visited = np.zeros(graph.n_nodes, dtype="uint8")
        self.level = np.full(graph.n_nodes, fill_value=-1, dtype="int32")
        self.parent = np.full(graph.n_nodes, fill_value=-1, dtype="int32")
        
        self.level[root-1] = 0
        self.visited[root-1] = 1
        
        self.start_root(root)
        
    def start_root(self, root):
#         self.queue = Queue(size=self.graph.n_nodes)
#         self.queue.add(root)
        self.queue = []
        self.queue.append(root)
        
    def search(self):
#         first = self.queue.next()
        first = self.queue[0]
        for neighbor in self.graph.nodes[first-1]:
            if first == neighbor:
                continue
                
            if not self.visited[neighbor-1]:
                self.visited[neighbor-1] = 1
                self.parent[neighbor-1] = first
                self.level[neighbor-1] = self.level[first-1] + 1
#                 self.queue.add(neighbor)
                self.queue.append(neighbor)
        
#         if not self.queue.is_empty():
        if len(self.queue):
            self.search()

folder = "inputs"
filename = "grafo_3.txt"

path = os.path.join(folder, filename)
graph = open_graph_txt(path)
graph.sort_neighbors()

bfs = BFS(graph, 1)
bfs.search()


