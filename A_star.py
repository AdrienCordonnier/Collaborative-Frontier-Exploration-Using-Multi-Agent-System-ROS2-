import heapq

# Define the Cell class
class Cell:
	def __init__(self):
		self.parent_i = 0     # Parent cell's row index
		self.parent_j = 0     # Parent cell's column index
		self.f = float('inf') # Total cost of the cell (g + h)
		self.g = float('inf') # Cost from start to this cell
		self.h = 0            # Heuristic cost from this cell to destination

def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Get neighboring cells of a given cell
def get_neighbors(grid, obstacles, cell):
    x, y = cell
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]  # Assuming 4-connectivity
    valid_neighbors = []
    for neighbor in neighbors:
        if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and neighbor not in obstacles:
            valid_neighbors.append(neighbor)
    return valid_neighbors
    
# A* algorithm to find the path from start to goal
def astar(grid, obstacles, start, goal):
    start = start
    goal = goal
    open_set = []
    heapq.heappush(open_set, (0, start))  # Push start node into the open set with priority 0
    came_from = {}
    # Initialize g_score dictionary with default values
    g_score = {(x, y): float('inf') for x in range(len(grid)) for y in range(len(grid[0]))}
    g_score[start] = 0

    f_score = {cell: float('inf') for row in grid for cell in row}
    f_score[start] = heuristic(start, goal)

    while open_set:
        current_f, current = heapq.heappop(open_set)  # Pop the node with the lowest f score from the open set

        if current == goal:
            return reconstruct_path(came_from, current)[1:]  # Goal reached, reconstruct and return the path

        for neighbor in get_neighbors(grid, obstacles, current):  # Iterate through neighbors of current node
            tentative_g_score = g_score[current] + 1  # Assuming uniform cost for simplicity

            if tentative_g_score < g_score[neighbor]:  # If this path is better than previously recorded
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))  # Add neighbor to open set with updated priority

    return None  # No path found