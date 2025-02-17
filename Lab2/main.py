import heapq
import time
import random
import matplotlib.pyplot as plt
import networkx as nx


# Алгоритм обхода в ширину (BFS)
def bfs(graph, start):
    visited = set()
    queue = [start]
    order = []
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            order.append(vertex)
            queue.extend(graph[vertex] - visited)
    return order


# Алгоритм обхода в глубину (DFS)
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    if start not in order:
        order = [start]
    for next_vertex in graph[start] - visited:  # Проходим по соседям, которые ещё не были посещены
        order.extend(dfs(graph, next_vertex, visited))  # Рекурсивный вызов для непосещённых соседей
    return order



# Алгоритм Дейкстры для нахождения кратчайших путей
def dijkstra(graph, start):
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]
    visited = set()
    order = []
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        if current_vertex in visited:
            continue
        visited.add(current_vertex)
        order.append(current_vertex)
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return order


# Генерация случайного графа
def generate_graph(num_vertices, density=0.5, weighted=False):
    if num_vertices <= 0:
        raise ValueError("Количество вершин должно быть положительным числом.")
    if weighted:
        graph = {i: {} for i in range(num_vertices)}
    else:
        graph = {i: set() for i in range(num_vertices)}

    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < density:
                if weighted:
                    weight = random.randint(1, 10)
                    graph[i][j] = weight
                    graph[j][i] = weight
                else:
                    graph[i].add(j)
                    graph[j].add(i)
    return graph


# Визуализация графа
def visualize_graph(graph, visited=None, title="Граф"):
    G = nx.Graph()
    for vertex in graph:
        for neighbor in graph[vertex]:
            if isinstance(graph[vertex], dict):
                weight = graph[vertex][neighbor]
                G.add_edge(vertex, neighbor, weight=weight)
            else:
                G.add_edge(vertex, neighbor)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))

    node_colors = ['lightgreen' if node in visited else 'lightblue' for node in G.nodes] if visited else 'lightblue'

    if isinstance(graph[list(graph.keys())[0]], dict):
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    else:
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=10)

    plt.title(title)
    plt.show()


# Измерение времени выполнения алгоритма
def measure_time(algorithm, graph, start):
    start_time = time.perf_counter()
    result = algorithm(graph, start)
    elapsed_time = time.perf_counter() - start_time
    return result, elapsed_time


# Основная функция
def main():
    sizes = [5, 10, 20]
    densities = [0.7]
    bfs_times = []
    dfs_times = []
    dijkstra_times = []

    for density in densities:
        print(f"\nТестирование с плотностью {density}:")

        for size in sizes:
            print(f"\nГраф из {size} вершин:")
            graph = generate_graph(size, density=density, weighted=True)
            unweighted_graph = generate_graph(size, density=density)

            bfs_result, bfs_time = measure_time(bfs, unweighted_graph, 0)
            bfs_times.append(bfs_time)
            print(f"BFS (посещённые вершины): {bfs_result}")
            print(f"BFS: {bfs_time:.6f} сек")
            visualize_graph(unweighted_graph, visited=bfs_result, title=f"BFS: Посещённые вершины для {size} вершин")

            dfs_result, dfs_time = measure_time(dfs, unweighted_graph, 0)
            dfs_times.append(dfs_time)
            print(f"DFS (посещённые вершины): {dfs_result}")
            print(f"DFS: {dfs_time:.6f} сек")
            visualize_graph(unweighted_graph, visited=dfs_result, title=f"DFS: Посещённые вершины для {size} вершин")

            dijkstra_result, dijkstra_time = measure_time(dijkstra, graph, 0)
            dijkstra_times.append(dijkstra_time)
            print(f"Дейкстра (посещённые вершины): {dijkstra_result}")
            print(f"Дейкстра: {dijkstra_time:.6f} сек")
            visualize_graph(graph, visited=dijkstra_result, title=f"Дейкстра: Посещённые вершины для {size} вершин")

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(sizes * len(densities), bfs_times, marker='o', label='BFS', color='blue')
    plt.plot(sizes * len(densities), dfs_times, marker='o', label='DFS', color='green')
    plt.plot(sizes * len(densities), dijkstra_times, marker='o', label='Дейкстра', color='red')
    plt.xlabel('Размер графа (количество вершин)')
    plt.ylabel('Время выполнения (сек)')
    plt.title('Сравнение времени выполнения алгоритмов')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
