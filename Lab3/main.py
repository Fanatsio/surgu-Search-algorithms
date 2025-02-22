import json
import heapq
import math
import time
import matplotlib.pyplot as plt
import numpy as np

def load_maze(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def heuristic_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def heuristic_euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def a_star_search(maze_data, heuristic):
    width, height = maze_data['width'], maze_data['height']
    start, goal = tuple(maze_data['start']), tuple(maze_data['goal'])
    maze = maze_data['maze']
    
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    visited_nodes = 0
    
    while open_list:
        _, current = heapq.heappop(open_list)
        visited_nodes += 1
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, visited_nodes
        
        x, y = current
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        
        for neighbor in neighbors:
            nx, ny = neighbor
            if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 0:
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
    
    return None, visited_nodes  # No path found

def visualize_maze(maze_data, path):
    maze = np.array(maze_data['maze'])
    fig, ax = plt.subplots()
    
    ax.imshow(maze, cmap='gray_r')
    
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if (x, y) == tuple(maze_data['start']):
                ax.text(x, y, 'S', ha='center', va='center', color='green', fontsize=12, fontweight='bold')
            elif (x, y) == tuple(maze_data['goal']):
                ax.text(x, y, 'G', ha='center', va='center', color='red', fontsize=12, fontweight='bold')
    
    if path:
        for (x, y) in path:
            ax.add_patch(plt.Circle((x, y), 0.3, color='blue', alpha=0.6))
    
    plt.show()

def compare_heuristics(file_path):
    maze_data = load_maze(file_path)
    
    print("Testing Manhattan heuristic...")
    start_time = time.time()
    path_manhattan, visited_manhattan = a_star_search(maze_data, heuristic_manhattan)
    manhattan_time = time.time() - start_time
    print(f"Visited nodes: {visited_manhattan}, Time: {manhattan_time:.6f} sec")
    visualize_maze(maze_data, path_manhattan)
    
    print("\nTesting Euclidean heuristic...")
    start_time = time.time()
    path_euclidean, visited_euclidean = a_star_search(maze_data, heuristic_euclidean)
    euclidean_time = time.time() - start_time
    print(f"Visited nodes: {visited_euclidean}, Time: {euclidean_time:.6f} sec")
    visualize_maze(maze_data, path_euclidean)
    
    return {
        "manhattan": {"path_length": len(path_manhattan) if path_manhattan else 0, "visited_nodes": visited_manhattan, "time": manhattan_time},
        "euclidean": {"path_length": len(path_euclidean) if path_euclidean else 0, "visited_nodes": visited_euclidean, "time": euclidean_time}
    }

# Запуск сравнения
if __name__ == "__main__":
    file_path = "./Lab3/maze.json"  # Укажи путь к файлу с лабиринтом
    results = compare_heuristics(file_path)
    print("\nComparison results:")
    print(results)
