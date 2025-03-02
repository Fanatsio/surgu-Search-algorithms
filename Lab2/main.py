import heapq
import time
import random
import matplotlib.pyplot as plt
import networkx as nx

def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    order = [start]
    for next_vertex in sorted(graph[start] - visited):
        order.extend(dfs(graph, next_vertex, visited))
    return order

def bfs(graph, start, target=None):
    visited = set()
    queue = [start]
    order = []
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            if target is not None and vertex == target:
                break
            queue.extend(sorted(graph[vertex] - visited))
    return order

def dijkstra(graph, start, target=None):
    visited = set()
    pq = [(0, start)]
    order = []
    while pq:
        _, vertex = heapq.heappop(pq)
        if vertex in visited:
            continue
        visited.add(vertex)
        order.append(vertex)
        if target is not None and vertex == target:
            break
        for neighbor in sorted(graph[vertex]):
            if neighbor not in visited:
                heapq.heappush(pq, (0, neighbor))
    return order

def visualize_graph(graph, title="Граф"):
    G = nx.Graph()
    for vertex in graph:
        for neighbor in graph[vertex]:
            G.add_edge(vertex, neighbor)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    plt.title(title)
    plt.show()

def measure_time(algorithm, graph, start):
    start_time = time.perf_counter()
    result = algorithm(graph, start)
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, result

small_graph = {
    0: {1, 2},
    1: {0, 3, 4},
    2: {0, 4},
    3: {1, 4},
    4: {1, 2, 3}
}

medium_graph = {
    0: {1, 2, 3},
    1: {0, 4, 5},
    2: {0, 6, 7},
    3: {0, 8, 9},
    4: {1, 10, 11},
    5: {1, 12, 13},
    6: {2, 14, 15},
    7: {2, 16, 17},
    8: {3, 18, 19},
    9: {3, 10, 11},
    10: {4, 9, 12},
    11: {4, 9, 13},
    12: {5, 10, 14},
    13: {5, 11, 15},
    14: {6, 12, 16},
    15: {6, 13, 17},
    16: {7, 14, 18},
    17: {7, 15, 19},
    18: {8, 16, 19},
    19: {8, 17, 18}
}

for graph, name in [(small_graph, "Малый граф"), (medium_graph, "Средний граф")]:
    print(f"\n{name}:")
    visualize_graph(graph, title=name)
    for algorithm, label in [(bfs, "BFS"), (dfs, "DFS"), (dijkstra, "Дейкстра")]:
        time_taken, order = measure_time(algorithm, graph, 0)
        print(f"{label}: {time_taken:.6f} сек, Порядок обхода: {order}")

sizes = [10, 20, 40, 60, 80, 100]
bfs_times = []
dfs_times = []
dijkstra_times = []

for size in sizes:
    random_graph = {i: set(random.sample(range(size), random.randint(1, min(5, size-1)))) for i in range(size)}
    print(f"\nСлучайный граф из {size} вершин")
    for algorithm, times_list, label in [(bfs, bfs_times, "BFS"), (dfs, dfs_times, "DFS"), (dijkstra, dijkstra_times, "Дейкстра")]:
        time_taken, _ = measure_time(algorithm, random_graph, 0)
        times_list.append(time_taken)
        print(f"{label}: {time_taken:.6f} сек")

plt.figure(figsize=(10, 6))
plt.plot(sizes, bfs_times, marker='o', label='BFS', linestyle='-', color='r')
plt.plot(sizes, dfs_times, marker='o', label='DFS', linestyle='-', color='g')
plt.plot(sizes, dijkstra_times, marker='o', label='Дейкстра', linestyle='-', color='b')
plt.xlabel('Размер графа (количество вершин)')
plt.ylabel('Время выполнения (сек)')
plt.title('Сравнение времени выполнения BFS, DFS и Дейкстры')
plt.legend()
plt.grid(True)
plt.show()
