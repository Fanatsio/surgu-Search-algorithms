import heapq
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import time
import numpy as np

# Примеры графов

# Малый невзвешенный граф
small_graph = {
    0: {1, 2},
    1: {0, 3, 4},
    2: {0, 4},
    3: {1, 4},
    4: {1, 2, 3}
}

# Средний невзвешенный граф (ручное задание связей)
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

# Граф со взвешенными рёбрами
weighted_graph = {
    0: {1: 4, 2: 1},
    1: {0: 4, 3: 1},
    2: {0: 1, 3: 2},
    3: {1: 1, 2: 2, 4: 3},
    4: {3: 3}
}

# Функции алгоритмов

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    order = []
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            queue.extend(graph[vertex] - visited)
    return order

def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    order = []
    stack = [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            stack.extend(graph[vertex] - visited)
    return order

def dijkstra(graph, start):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)
        if current_distance > distances[current_vertex]:
            continue
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    return distances

def draw_graph(graph, title="Граф"):
    G = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if isinstance(neighbors, dict):
                G.add_edge(node, neighbor, weight=neighbors[neighbor])
            else:
                G.add_edge(node, neighbor)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    if any(isinstance(neighbors, dict) for neighbors in graph.values()):
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title(title)
    plt.show()

# Функция замера времени выполнения алгоритмов
def measure_time(func, graph, start):
    start_time = time.time()
    result = func(graph, start)
    end_time = time.time()
    return result, end_time - start_time

# Выполнение алгоритмов и измерение времени
bfs_small, time_bfs_small = measure_time(bfs, small_graph, 0)
dfs_small, time_dfs_small = measure_time(dfs, small_graph, 0)
bfs_medium, time_bfs_medium = measure_time(bfs, medium_graph, 0)
dfs_medium, time_dfs_medium = measure_time(dfs, medium_graph, 0)
dijkstra_weighted, time_dijkstra_weighted = measure_time(dijkstra, weighted_graph, 0)

# Вывод результатов
print("BFS (малый граф):", bfs_small, "Время:", time_bfs_small)
print("DFS (малый граф):", dfs_small, "Время:", time_dfs_small)
print("BFS (средний граф):", bfs_medium, "Время:", time_bfs_medium)
print("DFS (средний граф):", dfs_medium, "Время:", time_dfs_medium)
print("Дейкстра (взвешенный граф):", dijkstra_weighted, "Время:", time_dijkstra_weighted)

# Визуализация графиков времени выполнения
algorithms = ["BFS Small", "DFS Small", "BFS Medium", "DFS Medium", "Dijkstra"]
times = [time_bfs_small, time_dfs_small, time_bfs_medium, time_dfs_medium, time_dijkstra_weighted]
plt.figure(figsize=(8, 5))
plt.bar(algorithms, times, color=['blue', 'green', 'blue', 'green', 'red'])
plt.xlabel("Алгоритмы")
plt.ylabel("Время (сек)")
plt.title("Сравнение времени выполнения алгоритмов")
plt.show()

# Визуализация графов
draw_graph(small_graph, "Малый невзвешенный граф")
draw_graph(medium_graph, "Средний невзвешенный граф")
draw_graph(weighted_graph, "Взвешенный граф")
