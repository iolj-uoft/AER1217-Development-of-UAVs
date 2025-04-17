"""Example utility module.

Please use a file like this one to add extra functions.

"""
import numpy as np
import random
import math

class RRTStarNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0
        
def offset_from_yaw(x, y, yaw, dist=0.4):
    """Returns a point dist meters before (x, y) along -yaw direction."""
    yaw -= np.pi / 2
    dx = dist * math.cos(yaw)
    dy = dist * math.sin(yaw)
    return (x + dx, y + dy)

def double_offset_from_yaw(x, y, yaw, d1=0.5, d2=0.25):
    yaw -= np.pi / 2
    dx1 = d1 * np.cos(yaw)
    dy1 = d1 * np.sin(yaw)
    dx2 = d2 * np.cos(yaw)
    dy2 = d2 * np.sin(yaw)
    return (x + dx1, y + dy1), (x + dx2, y + dy2)

def is_line_collision_free(p1, p2, obstacles):
    x1, y1 = p1
    x2, y2 = p2
    steps = max(1, int(np.hypot(x2 - x1, y2 - y1) / 0.05))
    for i in range(steps + 1):
        x = x1 + (x2 - x1) * i / steps
        y = y1 + (y2 - y1) * i / steps
        for ox, oy, r in obstacles:
            if np.hypot(x - ox, y - oy) <= r:
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
                      max_iter=1000,
                      step_size=0.2,
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

    for _ in range(max_iter):
        rnd = sample()
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