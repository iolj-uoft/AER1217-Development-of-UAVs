"""Example utility module.

Please use a file like this one to add extra functions.

"""
import numpy as np
import random
import math

def get_gate_edge_buffers(x, y, yaw, width=0.64, thickness=0.15, buffer_radius=0.10):
    """
    Returns a list of (x, y, r) circles that approximate buffered gate edges.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])

    hw, ht = width / 2, thickness / 2
    corners = np.array([
        [-hw, -ht],
        [ hw, -ht],
        [ hw,  ht],
        [-hw,  ht]
    ])
    world_corners = np.dot(corners, R.T) + np.array([x, y])
    edge_buffers = []

    for i in range(4):
        p1 = world_corners[i]
        p2 = world_corners[(i + 1) % 4]
        # Discretize each edge into N points
        N = 10
        for t in np.linspace(0, 1, N):
            px = p1[0] + t * (p2[0] - p1[0])
            py = p1[1] + t * (p2[1] - p1[1])
            edge_buffers.append((px, py, buffer_radius))
    
    return edge_buffers

def get_gate_edges(x, y, yaw, width=0.64, thickness=0.12):
    """
    Returns a list of 4 line segments representing a rectangular gate frame
    centered at (x, y) with rotation yaw.
    Each segment is a tuple: ((x1, y1), (x2, y2))
    """
    import numpy as np

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])

    hw, ht = width / 2, thickness / 2
    corners = np.array([
        [-hw, -ht],
        [ hw, -ht],
        [ hw,  ht],
        [-hw,  ht]
    ])
    world_corners = np.dot(corners, R.T) + np.array([x, y])

    edges = []
    for i in range(4):
        p1 = world_corners[i]
        p2 = world_corners[(i + 1) % 4]
        edges.append((tuple(p1), tuple(p2)))

    return edges

def is_segment_intersect(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

class RRTStarNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0
        
def offset_from_yaw(x, y, yaw, dist=0.3):
    """Returns a point dist meters before and after (x, y) along -yaw direction."""
    yaw -= np.pi / 2
    dx = dist * math.cos(yaw)
    dy = dist * math.sin(yaw)
    return (x + dx, y + dy), (x - dx, y - dy)

def is_line_collision_free(p1, p2, obstacles, gate_edges=[]):
    SAFE_MARGIN = 0.0
    x1, y1 = p1
    x2, y2 = p2
    steps = min(1000, max(1, int(np.hypot(x2 - x1, y2 - y1) / 0.05)))

    for i in range(steps + 1):
        x = x1 + (x2 - x1) * i / steps
        y = y1 + (y2 - y1) * i / steps
        for ox, oy, r in obstacles:
            if np.hypot(x - ox, y - oy) <= (r + SAFE_MARGIN):
                return False

    for edge in gate_edges:
        if is_segment_intersect(p1, p2, edge[0], edge[1]):
            return False

    return True

def shortcut_smooth(path, obstacles, is_collision_free_fn):
    new_path = [path[0]]
    i = 0
    while i < len(path) - 1:
        for j in reversed(range(i + 1, len(path))):
            if is_collision_free_fn(path[i], path[j], obstacles):
                new_path.append(path[j])
                i = j
                break
    return new_path

def plan_path_rrtstar(start, goal, obstacles, bounds,
                      max_iter=500,
                      step_size=0.1,
                      goal_sample_rate=0.1,
                      search_radius=1.0):
    min_x, max_x = bounds[0]
    min_y, max_y = bounds[1]
    
    start_node = RRTStarNode(start[0], start[1])
    goal_node = RRTStarNode(goal[0], goal[1])
    nodes = [start_node]

    def sample():
        if random.random() < goal_sample_rate:
            return goal_node
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        return RRTStarNode(x, y)

    def distance(n1, n2):
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def steer(from_node, to_node):
        d = distance(from_node, to_node)
        if d <= step_size:
            return to_node
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_x = from_node.x + step_size * math.cos(theta)
        new_y = from_node.y + step_size * math.sin(theta)
        new_node = RRTStarNode(new_x, new_y)
        new_node.parent = from_node
        new_node.cost = from_node.cost + step_size
        return new_node

    def is_collision_free(n1, n2):
        steps = max(1, int(distance(n1, n2) / 0.05))
        for i in range(steps + 1):
            x = n1.x + (n2.x - n1.x) * i / steps
            y = n1.y + (n2.y - n1.y) * i / steps
            for ox, oy, r in obstacles:
                if math.hypot(x - ox, y - oy) <= r:
                    return False
        return True

    def find_nearest(node_list, new_node):
        return min(node_list, key=lambda n: distance(n, new_node))

    def find_nearby(node_list, new_node):
        return [n for n in node_list if distance(n, new_node) <= search_radius]

    for i in range(max_iter):
        print(f"[RRT*] Sampling node {i}")
        rnd = sample()
        print(f"[RRT*] Finding nearest")
        nearest = find_nearest(nodes, rnd)
        new_node = steer(nearest, rnd)

        if not is_collision_free(nearest, new_node):
            continue

        neighbors = find_nearby(nodes, new_node)
        min_cost = nearest.cost + distance(nearest, new_node)
        best_parent = nearest

        for neighbor in neighbors:
            cost = neighbor.cost + distance(neighbor, new_node)
            if is_collision_free(neighbor, new_node) and cost < min_cost:
                best_parent = neighbor
                min_cost = cost

        new_node.parent = best_parent
        new_node.cost = min_cost
        nodes.append(new_node)

        # Rewiring
        for neighbor in neighbors:
            cost_through_new = new_node.cost + distance(new_node, neighbor)
            if is_collision_free(new_node, neighbor) and cost_through_new < neighbor.cost:
                neighbor.parent = new_node
                neighbor.cost = cost_through_new

        # Check for goal connection
        if distance(new_node, goal_node) < step_size and is_collision_free(new_node, goal_node):
            goal_node.parent = new_node
            goal_node.cost = new_node.cost + distance(new_node, goal_node)
            nodes.append(goal_node)
            break

    # Retrieve path
    path = []
    node = goal_node if goal_node.parent else min(nodes, key=lambda n: distance(n, goal_node))
    while node is not None:
        path.append((node.x, node.y))
        node = node.parent
    path.reverse()
    return path