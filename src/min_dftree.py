from copy import deepcopy
from math import inf
from pyvis.network import Network
from random import randint


class Graph:
    def __init__(self, num_vertices: int):
        """Creates an undirected graph with the specified number of vertices.
        Vertices are stored in an adjacency matrix, and the number of vertices
        cannot be modified. Vertices are index from [0, num_vertices).

        Args:
            num_vertices (int): The number of vertices in the graph
        """
        self.num_vertices = num_vertices
        self.__adj_matrix = [
            [None] * num_vertices for _ in range(num_vertices)]

        self.highlight_edges = []

    @staticmethod
    def generate_random(num_vertices: int) -> Graph:
        """Creates a simple, connected graph.

        Args:
            num_vertices (int): The number of vertices in the graph

        Returns:
            Graph: A simple, connected graph
        """
        graph = Graph(num_vertices)
        for origin in range(1, num_vertices):
            # Each new node creates at least 1 edge to the rest of the connected
            # subgraph
            for _ in range(randint(1, 3)):
                graph.add_edge(
                    origin,
                    randint(0, origin - 1),
                    randint(1, 12)  # Arbitrary range for weights
                )

        return graph

    def copy(self) -> Graph:
        """Returns a duplicate copy of the graph.

        Returns:
            Graph: A copy of the graph
        """
        graph = Graph(self.num_vertices)

        graph.__adj_matrix = deepcopy(self.__adj_matrix)
        graph.highlight_edges = deepcopy(self.highlight_edges)

        return graph

    def add_edge(self, start: int, end: int, weight: int):
        """Inserts an undirected edge with the specified weight between two
          unique vertices in the graph. Reversing the order of the two vertices
          has no effect. If the edge is are the same a loop connecting a vertex
          to itself, the edge is not is created.

        Args:
            start (int): An vertex the edge connects to
            end (int): The other vertex the edge connects to
            weight (int): The weight of the edge

        Raises:
            IndexError: If any of the vertices is considered out of the 
            expected bounds of [0, num_vertices)
            ValueError: If edge is a loop connecting a vertex to itself
        """
        if start < 0 or start >= self.num_vertices:
            raise IndexError(f'Start vertex {start} is out of bound.')

        if end < 0 or end >= self.num_vertices:
            raise IndexError(f'End vertex {end} is out of bound.')

        if start == end:
            return

        # Undirected edge
        self.__adj_matrix[start][end] = weight
        self.__adj_matrix[end][start] = weight

    def get_edges(self, vertex: int) -> list[tuple[int, int]]:
        """Returns a list of neighbor-weight pairs representing all the existing
        edges connected to a vertex.

        Args:
            vertex (int): The vertex to search for edges

        Raises:
            IndexError: If the vertex is considered out of the expected bounds
            of [0, num_vertices)

        Returns:
            list[int]: A list of existing edges connected to a vertex
        """
        if vertex < 0 or vertex >= self.num_vertices:
            raise IndexError(f'Vertex {vertex} is out of bound.')

        return [
            (neighbor, weight)
            for neighbor, weight in enumerate(self.__adj_matrix[vertex])
            if weight != None
        ]

    def get_all_edges(self) -> list[tuple[int, int, int]]:
        """Returns all unique vertices in the graph as a list of 
        (vertex1, vertex2, weight) tuples

        Returns:
            list[tuple[int, int, int]]: A list of all edges
        """
        edges = []
        for start in range(self.num_vertices):
            for stop in range(start, self.num_vertices):
                if (weight := self.__adj_matrix[start][stop]) == None:
                    continue

                edges.append((start, stop, weight))

        return edges

    def get_weight(self, start: int, end: int):
        return self.__adj_matrix[start][end]

    def get_min_tree(self, min_tree_algorithm: function) -> Graph:
        min_tree_edges = min_tree_algorithm(self)
        min_tree = self.copy()

        # Removes non-spanning tree edges
        for start, end, _ in self.get_all_edges():
            if ((start, end) not in min_tree_edges
                    and (end, start) not in min_tree_edges):
                min_tree.add_edge(start, end, None)

        return min_tree

    def display(self, output_path='graph.html'):
        """Displays the graph as an html file with the specified file path.

        Args:
            output_path (str, optional): The file path of the html output. 
            Defaults to 'graph.html'.
        """
        network = Network(
            height='100vh',
            bgcolor='#222',
            font_color='#eee')

        # Inserts all vertices
        vertices = [i for i in range(self.num_vertices)]
        network.add_nodes(vertices, label=list(map(str, vertices)))

        # Inserts all edges
        for start, stop, weight in self.get_all_edges():
            # Red highlight for special edges
            color = None
            if ((start, stop) in self.highlight_edges
                    or (stop, start) in self.highlight_edges):
                color = '#ec0b43'

            network.add_edge(
                start,
                stop,
                weight=weight,
                label=str(weight),
                color=color
            )

        network.save_graph(output_path)


def get_total_weight(graph: Graph, edges: list[tuple[int, int]]) -> int:
    """Gets the total weights of all the specified edges in a graph

    Args:
        graph (Graph): A graph to get edges
        edges (list[tuple[int, int]]): List of specified edges

    Returns:
        int: The total weight of the edges
    """
    return sum([graph.get_weight(*edge) for edge in edges])


def prim(graph: Graph) -> list[tuple[int, int]]:
    """Runs Prim's Algorithm on a connected graph and the edges of the minimum
    spanning tree as well as gets total weight of the edges in the tree.

    Args:
        graph (Graph): A connected graph

    Returns:
        list[set[int, int]]: The edges in the minimum spanning tree and the 
        total weight of all its edges
    """
    cheapest_edge = [None] * graph.num_vertices
    cheapest_cost = [inf] * graph.num_vertices

    cheapest_cost[0] = 0

    unexplored = set(range(graph.num_vertices))
    while unexplored:
        # Gets vertex-cost pairs which vertices are unexplored
        unexplored_cheapest = [
            (vertex, cost)
            for vertex, cost in enumerate(cheapest_cost)
            if vertex in unexplored
        ]

        current_vertex, _ = min(unexplored_cheapest, key=lambda pair: pair[1])

        # Marks the vertex as explored
        unexplored.remove(current_vertex)

        # Searches all edges adjacent to unexplored vertices and updates if the
        # edge is cheaper
        for neighbor, weight in graph.get_edges(current_vertex):
            if neighbor in unexplored and weight < cheapest_cost[neighbor]:
                cheapest_cost[neighbor] = weight
                cheapest_edge[neighbor] = (current_vertex, neighbor)

    # Gets cheapest edges that exists
    min_tree_edges = [edge for edge in cheapest_edge if edge]
    return min_tree_edges


def min_dftree(graph: Graph) -> list[tuple[int, int]]:
    predecessors = [None] * graph.num_vertices
    weights = [inf] * graph.num_vertices

    stack = [(None, 0, None)]  # Predecessor-vertex-weight tuples

    while stack:
        predecessor, vertex, weight = stack.pop()

        # Pushes unexplored neighors in descending-sorted order by weight
        unexplored_edges = sorted(
            [
                (neighbor, weight)
                for neighbor, weight in graph.get_edges(vertex)
                if predecessors[neighbor] == None  # Gets unexplored neighbors
            ],
            key=lambda pair: pair[1],  # Sorts by weight
            reverse=True
        )

        stack.extend([(vertex, neighbor, weight)
                     for neighbor, weight in unexplored_edges])

        # Updates best predecessor and weight
        if predecessor != None:
            if weight < weights[vertex]:
                weights[vertex] = weight
                predecessors[vertex] = predecessor

    min_tree_edges = [
        (start, end)
        for start, end in enumerate(predecessors)
        if end != None
    ]

    return min_tree_edges


if __name__ == '__main__':
    num_tests = 1
    for _ in range(num_tests):
        graph = Graph.generate_random(20)
        
        prim_edges = prim(graph)
        prim_weight = get_total_weight(graph, prim_edges) 

        dft_edges = min_dftree(graph)
        dft_weight = get_total_weight(graph, dft_edges)

        graph.highlight_edges = prim_edges
        graph.display()

        graph.highlight_edges = dft_edges
        graph.display('min_tree.html')

        # if success := (prim_weight != dft_weight):
        #     graph.highlight_edges = prim_edges
        #     graph.display()

        #     graph.highlight_edges = dft_edges
        #     graph.display('min_tree.html')
        #     break

    print(f'{num_tests} runs: {success}')
        

    
