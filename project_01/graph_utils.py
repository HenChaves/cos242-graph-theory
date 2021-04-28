import pandas as pd
import numpy as np
import os

class Graph:
    
    def __init__(self, n):
        self.n_nodes = n
        self.has_matrix = False
        self.matrix_error_size = None
        try:
            self.matrix = np.zeros((n, n), dtype="uint8")
            self.has_matrix = True
        except MemoryError as error:
            print("Cannot create matrix >>", error)
            ea = str(error)
            eb = ea[ea.index("iB for")-1]
            eb_m = 10**3 if eb=="G" else 10**6
            self.matrix_error_size = float(ea[ea.index("allocate")+9:ea.index("iB for")-2]) * eb_m
        self.nodes = [set() for i in range(n)]
    
    
    def add_edge(self, v, w):
        
        if v == w:
            if self.has_matrix:
                self.matrix[v-1, v-1] = 2
            self.nodes[v-1].add(v)
            
        else:
            if self.has_matrix:
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
        if self.has_matrix:
            return self.matrix
        return None
    
    def get_matrix_beautiful(self):
        if self.has_matrix:
            return pd.DataFrame(self.matrix, columns=np.arange(1, self.n_nodes+1), index=np.arange(1, self.n_nodes+1))
        return None
    
    def sort_neighbors(self):
        self.nodes = [sorted(n) for n in self.nodes]

def open_graph_txt(f, extra=False):
    lines = f.read().decode("utf8").split("\n")
    n_nodes = int(lines[0])
    edges = [tuple(map(lambda i: int(i), line.split(" "))) for line in lines[1:-1]]
    
    graph = Graph(n_nodes)
    for v, w in edges:
        graph.add_edge(v, w)
    
    if extra:
        return graph, n_nodes, edges
    
    return graph

class DFSL:
    def __init__(self, graph, root):
        self.graph = graph
        self.visited = np.zeros(graph.n_nodes, dtype="uint8")
        self.level = np.full(graph.n_nodes, fill_value=-1, dtype="int32")
        self.parent = np.full(graph.n_nodes, fill_value=-1, dtype="int32")
        self.level[root-1] = 0
        
        self.start_root(root)
    
    def start_root(self, root):
        self.stack = []
        self.stack.append(root)
    
    def search(self):
        while(len(self.stack) != 0):
            u = self.stack.pop()
            
            if not self.visited[u-1]:
                self.visited[u-1] = 1
                
                for v in self.graph.nodes[u-1][::-1]:
                    if not self.visited[v-1]:
                        self.stack.append(v)
                        self.parent[v-1] = u
                        self.level[v-1] = self.level[u-1] + 1

class BFSL:
    def __init__(self, graph, root):
        self.graph = graph
        self.visited = np.zeros(graph.n_nodes, dtype="uint8")
        self.level = np.full(graph.n_nodes, fill_value=-1, dtype="int32")
        self.parent = np.full(graph.n_nodes, fill_value=-1, dtype="int32")
        
        self.level[root-1] = 0
        self.visited[root-1] = 1
        
        self.start_root(root)
        
    def start_root(self, root):
        self.queue = []
        self.queue.append(root)
        
    def search(self):
        
        while(len(self.queue)):
            v = self.queue.pop(0)
            
            for w in self.graph.nodes[v-1]:
                if v == w:
                        continue
                if not self.visited[w-1]:
                    self.visited[w-1] = 1
                    self.queue.append(w)
                    self.parent[w-1] = v
                    self.level[w-1] = self.level[v-1] + 1 

class MinimumPath:
    
    def __init__(self, graph):
        self.graph = graph
        self.matrix = np.full((graph.n_nodes, graph.n_nodes), fill_value=-1, dtype="int32")
        self.run()
    
    def run(self):
        for v in np.arange(1, self.graph.n_nodes+1):
            bfsl = BFSL(self.graph, v)
            bfsl.search()
            for bfsl_node_index in np.argwhere(bfsl.visited == 1).reshape(-1):
                self.matrix[v-1, bfsl_node_index] = bfsl.level[bfsl_node_index]
            del bfsl
    
    def get_distance(self, u, v):
        return self.matrix[u-1, v-1]
    
    def get_diameter(self):
        return np.max(self.matrix)
    
    def get_matrix(self):
        return self.matrix
    
    def get_matrix_beautiful(self):
        return pd.DataFrame(self.matrix, columns=np.arange(1, self.graph.n_nodes+1), index=np.arange(1, self.graph.n_nodes+1))

class Components:
    
    def __init__(self, graph):
        self.graph = graph
        self.visited = np.zeros(graph.n_nodes, dtype="uint8")
        self.components = []
        
        while np.argwhere(self.visited == 0).reshape(-1).shape[0] > 0:
            root = np.argwhere(self.visited == 0).reshape(-1)[0] + 1

            bfsl = BFSL(self.graph, root)
            bfsl.search()
            
            bfsl_visited_index = np.argwhere(bfsl.visited == 1).reshape(-1)
            
            self.visited[bfsl_visited_index] = 1
            self.components.append((bfsl_visited_index+1).tolist())

    def get_components(self):
        a = sorted(self.components, key=lambda x: len(x), reverse=True)
        b = [len(x) for x in a]
        c = list(zip(b, a))
        return c

