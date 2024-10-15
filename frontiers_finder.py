from statistics import median
from queue import Queue

def is_valid_point(point, rows, cols):
    """This funtion return True or false if the point is in the limits of the border or not"""
    return 1 < point[0] < rows and 1 < point[1] < cols

def get_adjacent_points(point):
    """This function return all the adjacent points of a given point"""
    x, y = point
    return [[x-1, y], [x+1, y], [x, y-1], [x, y+1], [x-1, y-1], [x+1, y-1], [x-1, y+1], [x+1, y+1]]

def is_frontier(point, matrix):
    """Cheking if the given point is a frontier or not. A frontier is an unknown point which has at least 1 free space neighbor"""
    if matrix[point[0]][point[1]] == -1:                               # if unkown point
        for neighbor in get_adjacent_points(point):
            x, y = neighbor
            if not is_valid_point((x,y), len(matrix), len(matrix[0])): # cheking if valid point
                continue
            if matrix[x][y] == 0:                                      # returning true if there's at least 1 free space in the neighbors of the given point
                return True
    return False

def detect_frontiers(matrix, robot_pose):
    "This function returns all the frontiers of a map(matrix)"

    rows = len(matrix)
    cols = len(matrix[0])
    matrix1 = [list(row) for row in matrix] # copying the map
    matrix2 = [list(row) for row in matrix] # copying the map

    frontiers = []                          # frontiers

    queue_m = Queue()
    queue_m.put(robot_pose)                 # putting the robot pose at first

    while not queue_m.empty():
        p = queue_m.get()                   # taking an elemenet of the queue
        if matrix1[p[0]][p[1]] == "Map-Close-List":
            continue
        frontier_found = False

        for adj_point in get_adjacent_points(p):
            if not is_valid_point(adj_point, rows, cols):
                continue
            if matrix2[adj_point[0]][adj_point[1]] == -1:  # if one of the adjacent point is an uneplored cell
                frontier_found = True
                break

        if frontier_found:
            queue_f = Queue()
            new_frontier = []
            queue_f.put(p)#we add the point to the frontier queue
            matrix1[p[0]][p[1]] = "Frontier-Open-List"#updating the copied map
            
            while not queue_f.empty():#while the frontier list not empty
                q = queue_f.get()#we take an element of the list
                if matrix1[q[0]][q[1]] in ["Map-Close-List", "Frontier-Close-List"]:
                    continue
                if is_frontier((q[0],q[1]),matrix2):
                    new_frontier.append(q)  # if the point is a frontier, we add it to the list
                    for adj_point in get_adjacent_points(q):#now searching in the adjacent points of the frontier point
                        if not is_valid_point(adj_point, rows, cols):
                            continue
                        if matrix1[adj_point[0]][adj_point[1]] not in ["Frontier-Open-List", "Frontier-Close-List", "Map-Close-List"]:
                            queue_f.put(adj_point)#adding the adjacent point to the queue to analize it after
                            matrix1[adj_point[0]][adj_point[1]] = "Frontier-Open-List"#updating the copied matrix
                matrix1[q[0]][q[1]] = "Frontier-Close-List"

            if new_frontier:#add the frontier only if it's not empty
                frontiers.append(new_frontier)
            for frontier_point in new_frontier:
                matrix1[frontier_point[0]][frontier_point[1]] = "Map-Close-List"#closing all the points added to the frontiers

        for v in get_adjacent_points(p):
            if not is_valid_point(v, rows, cols):
                continue
            if matrix1[v[0]][v[1]] not in ["Map-Open-List", "Map-Close-List"]:
                has_open_neighbor = any(matrix2[adj[0]][adj[1]] == 0 for adj in get_adjacent_points(v) if is_valid_point(adj, rows, cols))
                if has_open_neighbor:
                    queue_m.put(v)  # ENQUEUE(queue_m, v)
                    matrix1[v[0]][v[1]] = "Map-Open-List"  # mark v as "Map-Open-List"

        matrix1[p[0]][p[1]] = "Map-Close-List"

    return frontiers

def compute_median_point(points):
    """Return the median point of a group of points"""
    i_coords = [point[0] for point in points]
    j_coords = [point[1] for point in points]
    median_i = int(median(i_coords))
    median_j = int(median(j_coords))
    return (median_i, median_j)

def median_point_Frontiers(points):
    """Return the median points of all the groups of adjacent points"""
    median_points = [compute_median_point(group) for group in points]
    return median_points

def calculate_distance(robot_position, frontier):
    """Calculate the distance between the robot position and a frontier"""
    x_robot, y_robot = robot_position
    x_frontier, y_frontier = frontier                             
    return abs(x_robot - x_frontier) + abs(y_robot - y_frontier)  

def assign_best_frontier(robot_position, frontiers):
    """Assign the best frontier points according to the robot position"""
    distances = [(calculate_distance(robot_position, frontier_point), frontier_point) for frontier_point in frontiers]
    sorted_frontier_points = sorted(distances, key=lambda x: x[0])
    return [frontier_point for _, frontier_point in sorted_frontier_points]
