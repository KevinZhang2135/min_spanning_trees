from graph import Graph
from math import inf


def get_total_weight(graph: Graph, edges: list[tuple[int, int]]) -> int:
    """Gets the total weights of all the specified edges in a graph

    Args:
        graph (Graph): A graph to get edges
        edges (list[tuple[int, int]]): List of specified edges

    Returns:
        int: The total weight of the edges
    """
    return sum([graph.get_weight(*edge) for edge in edges])


def shortest_path(graph: Graph) -> tuple[list[int], list[int]]:
    """Determines the shortest path of a graph, disregarding weights. May yield
    unexpected results for a disconnected graph.

    Args:
        graph (Graph): A graph to search for shortest paths.

    Returns:
        tuple[list[int], list[int]]: A list of total distances to each vertex
        and a list of precedessors to each vertex
    """
    queue = [0] # Arbituarily selects 0 as the starting vertex
    distances = [inf] * graph.num_vertices
    distances[0] = 0

    predecessors = [None] * graph.num_vertices

    # Continuous checks queued vertices
    while queue:
        vertex = queue.pop(0)
        neighbors, _ = zip(*graph.get_edges(vertex))  # Ignores weights

        # Queues unexplored neighbors and determines their distances
        for neighbor in neighbors:
            if predecessors[neighbor] != None:
                distances = distances[vertex] + 1
                predecessors[neighbor] = vertex

                queue.append(neighbor)

    return distances, predecessors


def breadth_first(graph: Graph) -> list[int]:
    """Performs breath-first-traversal of a graph and returns the traversal 
    order.

    Args:
        graph (Graph): A graph to traverse

    Returns:
        list[int]: The traversal order of vertices
    """
    queue = [0]  # Arbituarily selects 0 as the starting vertex
    explored = [False] * graph.num_vertices
    explored[0] = True

    visit_order = []

    # Continuously checks each queued vertex
    while queue:
        vertex = queue.pop(0)
        neighbors, _ = zip(*graph.get_edges(vertex))  # Ignores weights

        # Queues unexplored neighbors and marks them as explored
        for neighbor in neighbors:
            if not explored[neighbor]:
                visit_order.append(neighbor)
                explored[neighbor] = True

                queue.append(neighbor)

    return visit_order


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
    explored = set()

    while stack:
        predecessor, vertex, weight = stack.pop()

        # Pushes unexplored neighors in descending-sorted order by weight
        unexplored_edges = sorted(
            [
                (neighbor, weight)
                for neighbor, weight in graph.get_edges(vertex)
                if neighbor not in explored  # Gets unexplored neighbors
            ],
            key=lambda pair: pair[1],  # Sorts by weight
            reverse=True
        )

        explored.add(vertex)
        stack.extend([
            (vertex, neighbor, weight)
            for neighbor, weight in unexplored_edges
        ])

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


def test_dft_tree(show_graph=False):
    num_tests = 1000
    successes = num_tests

    for test_num in range(num_tests):
        graph = Graph.generate_random(6)

        # Computes the total weights
        prim_edges = prim(graph)
        prim_weight = get_total_weight(graph, prim_edges)

        dft_edges = min_dftree(graph)
        dft_weight = get_total_weight(graph, dft_edges)

        # Tracks failures
        fail_flag = 'fail' if prim_weight != dft_weight else ''
        successes -= prim_weight != dft_weight

        print(
            f'Test num {test_num:4}: {prim_weight:4} {dft_weight:4} {fail_flag}')

        # Displays a graph to inspect failures
        if show_graph and fail_flag:
            graph.highlight_edges = prim_edges
            graph.display()

            graph.highlight_edges = dft_edges
            graph.display('min_tree.html')
            break

    # Summary
    if not show_graph:
        print(f'Successful tests: {successes}/{num_tests}')


if __name__ == '__main__':
    graph = Graph.generate_random(10)
    print(breadth_first(graph))
    graph.display()
