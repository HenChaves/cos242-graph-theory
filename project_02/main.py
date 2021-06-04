import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import sys

class Graph:
    """
    Class to create a Graph.

    ...
    
    Arguments
    ----------
    n : int
        Number of nodes in the Graph.
    mode : str, optional
        The Graph mode, default or weighted. Defaults to `"default"`.
    representation : str, optional
        The Graph representation, matrix or lists. Defaults to `"matrix"`.

    Attributes
    ----------
    n_nodes : int
        Number of nodes in the Graph.
    mode : str
        The Graph mode, default or weighted.
    has_matrix : bool
        Whether the Graph has a matrix representation.
    matrix_error_size: float | None
        The needed memory size for the matrix representation, in case of error.

    Methods
    -------
    add_edge(v, w, weight=1)
        Add edge to Graph.
    get_node(v)
        Gets the neighbors of the v node.
    get_list()
        Gets the full adjacency list of the Graph.
    get_matrix()
        Gets the full matrix of the Graph.
    get_matrix_beautiful()
        Gets the full matrix of the Graph as a Pandas' DataFrame.

    """
    def __init__(self, n, mode="default", representation="matrix"):
        self.n_nodes = n
        self.mode = mode
        self.has_matrix = False
        self.matrix_error_size = None
        
        if representation == "matrix":
            try:
                self.matrix = np.full((n, n), np.inf)
                self.has_matrix = True

            except MemoryError as error:
                print("Cannot create matrix >>", error)
                ea = str(error)
                eb = ea[ea.index("iB for")-1]
                eb_m = 10**3 if eb=="G" else 10**6
                self.matrix_error_size = float(ea[ea.index("allocate")+9:ea.index("iB for")-2]) * eb_m

                self.lists = [list() for i in range(n)]
        
        elif representation == "lists":
            self.lists = [list() for i in range(n)]
        
        else:
            raise KeyError(f"Invalid representation. ({representation})")
    
    
    
    def add_edge(self, v, w, weight=1):
        
        if self.has_matrix:
            if (v != w) and (self.matrix[v-1, w-1] == np.inf):
                self.matrix[v-1, w-1] = weight
                self.matrix[w-1, v-1] = weight
        else:
            if v != w:
                if self.mode == "default":
                    self.lists[v-w].append(w)
                    self.lists[w-1].append(v)
                    
                if self.mode == "weighted":
                    if w not in np.array(self.get_node(v)).reshape(-1, 2)[:, 0].astype("int").tolist():
                        self.lists[v-1].append([w, weight])
                        self.lists[w-1].append([v, weight])

    
    def get_node(self, v):
        
        if not self.has_matrix:
            return self.lists[v-1]
        
        else:
            if self.mode == "default":
                return (np.where(self.matrix[v-1] == 1)[0] + 1).tolist()
            
            if self.mode == "weighted":
                neighbors = (np.where(self.matrix[v-1] < np.inf)[0] + 1).tolist()
                weights = self.matrix[v-1][self.matrix[v-1] < np.inf].tolist()
                return [list(t) for t in zip(neighbors, weights)]
    
    def get_lists(self):
        if not self.has_matrix:
            return self.lists
        else:
            return [self.get_node(n) for n in range(1, self.n_nodes + 1)]
            
    def get_matrix(self):
        if self.has_matrix:
            return self.matrix
        return None
    
    def get_matrix_beautiful(self):
        if self.has_matrix:
            return pd.DataFrame(self.matrix, columns=np.arange(1, self.n_nodes+1), index=np.arange(1, self.n_nodes+1))
        return None
    
    
def open_graph_txt(filename, extra=False, representation="matrix"):
    """
    Function to open a Graph file.

    ...
    
    Arguments
    ----------
    filename : str
        The filename of the file.
    extra : bool, optional
        Whether to return extra values. Defaults to `False`.
    representation : str, optional
        The Graph representation, matrix or lists. Defaults to `"matrix"`.

    """
    with open(filename, "r") as f:
        lines = [line for line in f.read().split("\n") if line != ""]
        n_nodes = int(lines[0])
        
        if np.array([line.split(" ") for line in lines[1:]]).shape[1] < 3:
            mode = "default"
        else:
            mode = "weighted"
        
        edges = [tuple(map(lambda i: int(i), line.split(" ")[:2])) for line in lines[1:]]
        
        if mode == "default":
            weights = [1 for _ in lines[1:]]
        if mode == "weighted":
            weights = [float(line.split(" ")[-1]) for line in lines[1:]]
        
        edges_weights = list(zip(edges, weights))
        
        graph = Graph(n_nodes, mode=mode, representation=representation)
        for (v, w), weight in edges_weights:
            graph.add_edge(v, w, weight=weight)
    
    if extra:
        return graph, n_nodes, edges

    return graph


def graph_statistics(graph):
    """
    Function to get statistics of a weightless Graph.

    ...
    
    Arguments
    ----------
    graph : Graph
        The Graph to get statistics from.

    """
    if graph.mode != "default": raise KeyError(f"Invalid graph mode. ({graph.mode})")
    print("Número de vértices:", graph.n_nodes)
    
    if graph.has_matrix:
        print("Número de arestas:", graph.get_matrix().sum()/2)
        print("Grau mínimo:", graph.get_matrix().sum(axis=0).min())
        print("Grau máximo:", graph.get_matrix().sum(axis=0).max())
        print("Grau médio:", graph.get_matrix().sum(axis=0).mean())
        print("Mediana do Grau:", np.median(graph.get_matrix().sum(axis=0)))
    else:
        print("Número de arestas:", np.sum([len(x) if i+1 not in x else len(x) + 1 for i, x in enumerate(graph.get_lists())])/2)
        print("Grau mínimo:", np.min([len(x) if i+1 not in x else len(x) + 1 for i, x in enumerate(graph.get_lists())]))
        print("Grau máximo:", np.max([len(x) if i+1 not in x else len(x) + 1 for i, x in enumerate(graph.get_lists())]))
        print("Grau médio:", np.mean([len(x) if i+1 not in x else len(x) + 1 for i, x in enumerate(graph.get_lists())]))
        print("Mediana do Grau:", np.median([len(x) if i+1 not in x else len(x) + 1 for i, x in enumerate(graph.get_lists())]))
    print("List: ", sys.getsizeof(graph.get_lists())/(10**6), "MB")
    print("Matrix: ", (str(sys.getsizeof(graph.get_matrix())/(10**6))) if graph.has_matrix else (graph.matrix_error_size), "MB")

    

class DFS:
    """
    Class to run a Depth-first Search in a weightless Graph.

    ...
    
    Attributes
    ----------
    graph : Graph
        Graph to run the DFS.
    root : int
        Root node to start the DFS.

    Attributes
    ----------
    graph : Graph
        Graph used for the DFS.
    visited : np_array
        Array of visited nodes.
    level : np_array
        Array with the level of the nodes.
    parent : np_array
        Array with the parent of the nodes.
    """
    def __init__(self, graph, root):
        self.graph = graph
        if self.graph.mode != "default":
            raise KeyError(f"Invalid mode. ({self.graph.mode})")
        
        self.visited = np.zeros(graph.n_nodes, dtype="uint8")
        self.level = np.full(graph.n_nodes, fill_value=np.inf)
        self.parent = np.full(graph.n_nodes, fill_value=-1, dtype="int32")
        self.level[root-1] = 0
        self.__start_root(root)
        self.__search()
    
    def __start_root(self, root):
        self.stack = []
        self.stack.append(root)
    
    def __search(self):
        while(len(self.stack) != 0):
            u = self.stack.pop()
            
            if not self.visited[u-1]:
                self.visited[u-1] = 1
                
                for v in sorted(self.graph.get_lists()[u-1], reverse=True):
                    if not self.visited[v-1]:
                        self.stack.append(v)
                        self.parent[v-1] = u
                        self.level[v-1] = self.level[u-1] + 1

                        
class BFS:
    """
    Class to run a Breadth-first Search in a weightless Graph.

    ...
    
    Attributes
    ----------
    graph : Graph
        Graph to run the BFS.
    root : int
        Root node to start the BFS.

    Attributes
    ----------
    graph : Graph
        Graph used for the BFS.
    visited : np_array
        Array of visited nodes.
    level : np_array
        Array with the level of the nodes.
    parent : np_array
        Array with the parent of the nodes.
    """
    def __init__(self, graph, root):
        self.graph = graph
        if self.graph.mode != "default":
            raise KeyError(f"Invalid mode. ({self.graph.mode})")
        
        self.visited = np.zeros(graph.n_nodes, dtype="uint8")
        self.level = np.full(graph.n_nodes, fill_value=np.inf)
        self.parent = np.full(graph.n_nodes, fill_value=-1, dtype="int32")
        
        self.level[root-1] = 0
        self.visited[root-1] = 1
        
        self.__start_root(root)
        self.__search()
        
    def __start_root(self, root):
        self.queue = []
        self.queue.append(root)
        
    def __search(self):
        
        while(len(self.queue)):
            v = self.queue.pop(0)
            
            for w in sorted(self.graph.get_lists()[v-1]):
                if v == w:
                        continue
                if not self.visited[w-1]:
                    self.visited[w-1] = 1
                    self.queue.append(w)
                    self.parent[w-1] = v
                    self.level[w-1] = self.level[v-1] + 1

                    
class MinimumPath:
    """
    Class to find the minimum path between all nodes in a weightless Graph.

    ...
    
    Arguments
    ----------
    graph : Graph
        Graph to find the minimum path.

    Attributes
    ----------
    graph : Graph
        Graph used for the minimum path.

    Methods
    ----------
    get_distance(u, v)
        Gets the distance for the minimum path between the nodes u and v.
    get_diameter()
        Gets the value for the diameter of the Graph.
    get_matrix()
        Gets the matrix of the minimum path.
    get_matrix_beautiful()
        Gets the matrix of the minimum path as a Pandas' DataFrame.
    """

    def __init__(self, graph):
        if graph.mode != "default": raise KeyError(f"Invalid graph mode. ({graph.mode})")
        self.graph = graph
        self.__matrix = np.full((graph.n_nodes, graph.n_nodes), fill_value=-1, dtype="int32")
        self.__run()
    
    def __run(self):
        for v in tqdm(range(1, self.graph.n_nodes+1)):
            bfs = BFS(self.graph, v)
            for bfs_node_index in np.argwhere(bfs.visited == 1).reshape(-1):
                self.__matrix[v-1, bfs_node_index] = bfs.level[bfs_node_index]
            del bfs
    
    def get_distance(self, u, v):
        return self.__matrix[u-1, v-1]
    
    def get_diameter(self):
        return np.max(self.__matrix)
    
    def get_matrix(self):
        return self.__matrix
    
    def get_matrix_beautiful(self):
        return pd.DataFrame(self.__matrix, columns=np.arange(1, self.graph.n_nodes+1), index=np.arange(1, self.graph.n_nodes+1))

    
class Components:
    """
    Class to find the all the connected components in a weightless Graph.

    ...
    
    Arguments
    ----------
    graph : Graph
        Graph to find the connected components.

    Attributes
    ----------
    graph : Graph
        Graph used for the connected components.
    visited : np_array
        Array of visited nodes.

    Methods
    ----------
    get_components()
        Gets list of components.
    """

    def __init__(self, graph):
        if graph.mode != "default": raise KeyError(f"Invalid graph mode. ({graph.mode})")
        self.graph = graph
        self.visited = np.zeros(graph.n_nodes, dtype="uint8")
        self.__components = []
        
        while np.argwhere(self.visited == 0).reshape(-1).shape[0] > 0:
            root = np.argwhere(self.visited == 0).reshape(-1)[0] + 1

            bfs = BFS(self.graph, root)
            bfs.search()
            
            bfs_visited_index = np.argwhere(bfs.visited == 1).reshape(-1)
            
            self.visited[bfs_visited_index] = 1
            self.__components.append((bfs_visited_index+1).tolist())

    def get_components(self):
        a = sorted(self.__components, key=lambda x: len(x), reverse=True)
        b = [len(x) for x in a]
        c = list(zip(b, a))
        return c

    
class Dijkstra:
    """
    Class to run the Dijkstra algorithm in a Graph.

    ...
    
    Arguments
    ----------
    graph : Graph
        Graph to run the Dijkstra algorithm.
    root : int
        Root node to run the Dijkstra algorithm.

    Attributes
    ----------
    graph : Graph
        Graph used for the Dijkstra algorithm.
    root : int
        Root node for the Dijkstra algorithm.
    distance : np_array
        Array of distances from the root node to another node.
    parent : np_array
        Array with the parent of the nodes.

    Methods
    ----------
    minpath(t)
        Gets the minimum path between the root node and the target node t.
    """
    def __init__(self, graph, root):
        if graph.mode == "default":
            self.graph = graph
            self.root = root
            self.__bfs = BFS(self.graph, self.root)
            self.distance = self.__bfs.level
            self.parent = self.__bfs.parent

        elif graph.mode == "weighted":
            self.graph = graph
            self.root = root
            
            self.distance = np.full(graph.n_nodes, fill_value=np.inf)
            self.__explored = np.zeros(graph.n_nodes, dtype="uint8")
            self.parent = np.full(graph.n_nodes, fill_value=-1, dtype="int32")
            self.__found = {self.root}

            self.distance[self.root-1] = 0
            self.parent[self.root-1] = -1
            
            self.__find()
        
        else:
            raise KeyError(f"Invalid graph mode. ({graph.mode})")
    
    def __find(self):
        while len(self.__found) != 0:
            u = np.argmin(np.where(self.__explored == 0, self.distance, np.inf)) + 1
            self.__found.remove(u)
            self.__explored[u-1] = 1
            for v, weight in self.graph.get_node(u):
                if weight < 0: raise ValueError(f"Weight cannot be negative. (weight = {weight})")
                if self.__explored[v-1] == 1: continue
                self.__found.add(v)
                if self.distance[v-1] > weight + self.distance[u-1]:
                    self.distance[v-1] = weight + self.distance[u-1]
                    self.parent[v-1] = u
    
    def minpath(self, t):
        path = [t]
        parent = self.parent[t-1]
        while parent != -1:
            path.append(parent)
            parent = self.parent[parent-1]
        if path[-1] == t: raise ValueError(f"Cannot find path between source and target. ({self.root}->{t})")
        return path[::-1]


def dijkstra_df_output(dijkstra, save=False):
    """
    Function to convert the Dijkstra Algorithm to a Pandas' DataFrame.

    ...
    
    Arguments
    ----------
    dijkstra : Dijkstra
        The Dijkstra to convert.
    save : bool, optional
        Whether to export the DataFrame as a csv file.

    """
    dijkstra_df = pd.DataFrame(list(zip(range(1, dijkstra.graph.n_nodes+1), dijkstra.distance, dijkstra.parent)), columns=["node", "distance", "parent"], index=np.arange(1, dijkstra.graph.n_nodes+1))
    if save:
        dijkstra_df.to_csv("outputs/dijkstra_out.csv")

    return dijkstra_df


class Prim:
    """
    Class to run the Prim algorithm in a Graph.

    ...
    
    Arguments
    ----------
    graph : Graph
        Graph to run the Prim algorithm.
    root : int
        Root node to run the Prim algorithm.

    Attributes
    ----------
    graph : Graph
        Graph used for the Prim algorithm.
    root : int
        Root node for the Prim algorithm.
    cost : np_array
        Array with the cost of the nodes.
    parent : np_array
        Array with the parent of the nodes.
    """
    def __init__(self, graph, root):
        if graph.mode == "default":
            self.graph = graph
            self.root = root
            self.__bfs = BFS(graph, root)
            self.cost = np.where(self.__bfs.level < np.inf, 1, np.inf)
            self.cost[root-1] = 0
            self.parent = self.__bfs.parent

        elif graph.mode == "weighted":
            self.graph = graph
            self.root = root
            self.cost = np.full(graph.n_nodes, fill_value=np.inf)
            self.__explored = np.zeros(graph.n_nodes, dtype="uint8")
            self.parent = np.full(graph.n_nodes, fill_value=-1, dtype="int32")
            self.__found = {root}

            self.cost[root-1] = 0
            self.parent[root-1] = -1
            
            self.__find()
        
        else:
            raise KeyError(f"Invalid graph mode. ({graph.mode})")
    
    def __find(self):
        while len(self.__found) != 0:
            u = np.argmin(np.where(self.__explored == 0, self.cost, np.inf)) + 1
            self.__found.remove(u)
            self.__explored[u-1] = 1
            for v, weight in self.graph.get_node(u):
                if weight < 0: raise ValueError(f"Weight cannot be negative. (weight = {weight})")
                if self.__explored[v-1] == 1: continue
                self.__found.add(v)
                if self.cost[v-1] > weight:
                    self.cost[v-1] = weight
                    self.parent[v-1] = u


def prim_df_output(prim, save=False):
    """
    Function to convert the Prim Algorithm to a Pandas' DataFrame.

    ...
    
    Arguments
    ----------
    prim : Prim
        The Prim to convert.
    save : bool, optional
        Whether to export the DataFrame as a csv file.

    """
    prim_df = pd.DataFrame(list(zip(range(1, prim.graph.n_nodes+1), prim.cost, prim.parent)), columns=["node", "cost", "parent"], index=np.arange(1, prim.graph.n_nodes+1))
    prim_df = prim_df[prim_df["parent"] != -1][["parent", "node", "cost"]].sort_values(by="parent")
    
    if save:
        with open("outputs/prim_out.txt", mode="w") as o:
            o.write(str(len(np.unique(prim_df[["parent", "node"]].values.ravel())))+" "+str(prim_df["cost"].sum()))
            o.write("\n"+prim_df.to_string(header=False, index=False, float_format="{:.2f}".format))

    return prim_df


class Eccentricity:
    """
    Class to find the eccentricity of a Graph node.

    ...
    
    Arguments
    ----------
    graph : Graph
        Graph to find the eccentricity.
    root : int
        Root node to find the eccentricity.

    Attributes
    ----------
    dijkstra : Dijkstra
        Dijkstra used to find the eccentricity.
    value : float
        The calculated eccentricity.
    """
    def __init__(self, graph, root):
        self.dijkstra = Dijkstra(graph, root)
        self.value = np.max(self.dijkstra.distance)
    def __repr__(self):
        return str(self.value)

