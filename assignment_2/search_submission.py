# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

from __future__ import division

import heapq
import os
import pickle
import math


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
        current (int): The index of the current node in the queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
        self.REMOVED = '<removed-task>'
        self.count = 0

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        while self.queue:
            priority, count, node = heapq.heappop(self.queue)
            return priority, node

        raise KeyError('Pop from empty Priority queue')

        # TODO: finish this function!
        raise NotImplementedError

    def remove(self, node_id):
        """
        Remove a node from the queue.

        This is a hint, you might require this in ucs,
        however, if you choose not to use it, you are free to
        define your own method and not use it.

        Args:
            node_id (int): Index of node in queue.
        """

        self.queue[node_id][2] = self.REMOVED
        # raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        self.count += 1
        heapq.heappush(self.queue, [node[0], self.count, node[1]])

        # TODO: finish this function!
        # raise NotImplementedError

    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n for _, n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self == other

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in the queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    path = []
    child_parent = {}
    if start == goal:
        return path

    count = 0
    found = 0
    entry_finder = set()
    frontier = PriorityQueue()
    explored = set()

    curr_node = start
    entry_finder.add(curr_node)
    while curr_node is not goal:
        children = graph[curr_node]
        for child in children:
            if child not in entry_finder:
                count += 1
                frontier.append((count, child))
                entry_finder.add(child)
                child_parent[child] = curr_node
                if child == goal:
                    found = 1
                    break
        explored.add(curr_node)
        if found == 1:
            break
        priority, curr_node = frontier.pop()

    node = goal
    while node is not start:
        path.append(node)
        node = child_parent[node]
    path.append(start)
    path.reverse()

    return path

    # TODO: finish this function!
    raise NotImplementedError


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    path = []
    child_parent = {}
    if start == goal:
        return path

    entry_finder = set()
    frontier = PriorityQueue()
    explored = {}
    curr_node = start
    entry_finder.add(curr_node)
    g = 0
    while 1:
        children = graph[curr_node]
        for child in children:
            if child in entry_finder:
                for i in range(len(frontier.queue)):
                    entry = frontier.queue[i]
                    if entry[2] == child:
                        child_path_cost = g + graph[curr_node][child]['weight']
                        if child_path_cost < entry[0]:
                            frontier.remove(i)
                            frontier.append((child_path_cost, child))
                            child_parent[child] = curr_node
                        break
            else:
                entry_finder.add(child)
                child_path_cost = g + graph[curr_node][child]['weight']
                frontier.append((child_path_cost, child))
                child_parent[child] = curr_node

        explored[curr_node] = g

        if len(frontier.queue):
            priority, curr_node = frontier.pop()
            while curr_node is frontier.REMOVED:
                priority, curr_node = frontier.pop()

        parent = child_parent[curr_node]
        g = explored[parent] + graph[parent][curr_node]['weight']

        if curr_node == goal:
            break

    node = goal
    while node is not start:
        path.append(node)
        node = child_parent[node]
    path.append(start)

    path.reverse()
    return path

    # TODO: finish this function!
    raise NotImplementedError


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    x1, y1 = graph.node[v]['pos']
    x2, y2 = graph.node[goal]['pos']
    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return distance
    # TODO: finish this function!
    raise NotImplementedError


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    path = []
    child_parent = {}
    if start == goal:
        return path

    entry_finder = set()
    frontier = PriorityQueue()
    explored = {}

    curr_node = start
    entry_finder.add(curr_node)
    g = 0
    h = euclidean_dist_heuristic(graph, start, goal)
    f = g + h
    while 1:
        children = graph[curr_node]
        for child in children:
            if child in entry_finder:
                for i in range(len(frontier.queue)):
                    entry = frontier.queue[i]
                    if entry[2] == child:
                        f = g + graph[curr_node][child]['weight'] + heuristic(graph, child, goal)
                        if f < entry[0]:
                            frontier.remove(i)
                            frontier.append((f, child))
                            child_parent[child] = curr_node
                        break
            else:
                entry_finder.add(child)
                f = g + graph[curr_node][child]['weight'] + heuristic(graph, child, goal)
                frontier.append((f, child))
                child_parent[child] = curr_node

        explored[curr_node] = g
        if len(frontier.queue):
            priority, curr_node = frontier.pop()
            while curr_node is frontier.REMOVED:
                priority, curr_node = frontier.pop()

        parent = child_parent[curr_node]
        g = explored[parent] + graph[parent][curr_node]['weight']

        if curr_node == goal:
            break

    node = goal
    while node is not start:
        path.append(node)
        node = child_parent[node]
    path.append(start)

    path.reverse()
    return path

    # TODO: finish this function!
    raise NotImplementedError


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    path = []
    start_path = []
    goal_path = []
    start_child_parent = {}
    goal_child_parent = {}
    if start == goal:
        return path

    start_entry_finder = set()
    goal_entry_finder = set()
    start_frontier = PriorityQueue()
    goal_frontier = PriorityQueue()
    start_explored = {}
    goal_explored = {}
    start_curr_node = start
    goal_curr_node = goal
    start_entry_finder.add(start_curr_node)
    goal_entry_finder.add(goal_curr_node)
    start_g = 0
    goal_g = 0
    while 1:
        start_children = graph[start_curr_node]
        for child in start_children:
            if child in start_entry_finder:
                for i in range(len(start_frontier.queue)):
                    entry = start_frontier.queue[i]
                    if entry[2] == child:
                        child_path_cost = start_g + graph[start_curr_node][child]['weight']
                        if child_path_cost < entry[0]:
                            start_frontier.remove(i)
                            start_frontier.append((child_path_cost, child))
                            start_child_parent[child] = start_curr_node
                        break
            else:
                start_entry_finder.add(child)
                child_path_cost = start_g + graph[start_curr_node][child]['weight']
                start_frontier.append((child_path_cost, child))
                start_child_parent[child] = start_curr_node

        start_explored[start_curr_node] = start_g

        if len(start_frontier.queue):
            priority, start_curr_node = start_frontier.pop()
            while start_curr_node is start_frontier.REMOVED:
                priority, start_curr_node = start_frontier.pop()

        parent = start_child_parent[start_curr_node]
        start_g = start_explored[parent] + graph[parent][start_curr_node]['weight']

        goal_children = graph[goal_curr_node]
        for child in goal_children:
            if child in goal_entry_finder:
                for i in range(len(goal_frontier.queue)):
                    entry = goal_frontier.queue[i]
                    if entry[2] == child:
                        child_path_cost = goal_g + graph[goal_curr_node][child]['weight']
                        if child_path_cost < entry[0]:
                            goal_frontier.remove(i)
                            goal_frontier.append((child_path_cost, child))
                            goal_child_parent[child] = goal_curr_node
                        break
            else:
                goal_entry_finder.add(child)
                child_path_cost = goal_g + graph[goal_curr_node][child]['weight']
                goal_frontier.append((child_path_cost, child))
                goal_child_parent[child] = goal_curr_node

        goal_explored[goal_curr_node] = goal_g

        if len(goal_frontier.queue):
            priority, goal_curr_node = goal_frontier.pop()
            while goal_curr_node is goal_frontier.REMOVED:
                priority, goal_curr_node = goal_frontier.pop()

        parent = goal_child_parent[goal_curr_node]
        goal_g = goal_explored[parent] + graph[parent][goal_curr_node]['weight']

        start_explored_set = set(start_explored.keys())
        goal_explored_set = set(goal_explored.keys())
        intersection = start_explored_set.intersection(goal_explored_set)

        intersection = list(intersection)
        if intersection:
            inter_set = []
            for node in intersection:
                cost = start_explored[node] + goal_explored[node]
                inter_set.append([cost, node])
            inter_set = sorted(inter_set)
            intersection = inter_set[0][1]

            a = 0
            b = 0
            if intersection in start_explored:
                a = start_explored[intersection]
            if intersection in goal_explored:
                b = goal_explored[intersection]
            mu = a + b

            start_frontier_goal_explored = []
            for element in start_frontier.queue:
                if element[2] in goal_explored:
                    cost = element[0] + goal_explored[element[2]]
                    start_frontier_goal_explored.append(cost)
            goal_frontier_start_explored = []
            for element in goal_frontier.queue:
                if element[2] in start_explored:
                    cost = element[0] + start_explored[element[2]]
                    goal_frontier_start_explored.append(cost)

            x = float("inf")
            y = float("inf")
            if start_frontier_goal_explored:
                x = min(start_frontier_goal_explored)
            if goal_frontier_start_explored:
                y = goal_frontier_start_explored
            check = min(x, y)

            if check >= mu:
                break

    node = intersection
    while intersection in goal_child_parent:
        goal_path.append(node)
        node = goal_child_parent[node]
        if node == goal:
            break
    goal_path.append(goal)

    node = intersection
    while intersection in start_child_parent:
        start_path.append(node)
        node = start_child_parent[node]
        if node == start:
            break
    start_path.append(start)
    start_path.reverse()

    goal_path.pop(0)

    path = start_path + goal_path
    return path

    # TODO: finish this function!
    raise NotImplementedError


def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    path = []
    start_path = []
    goal_path = []
    start_child_parent = {}
    goal_child_parent = {}
    if start == goal:
        return path

    start_entry_finder = set()
    goal_entry_finder = set()
    start_frontier = PriorityQueue()
    goal_frontier = PriorityQueue()
    start_explored = {}
    goal_explored = {}
    start_curr_node = start
    goal_curr_node = goal
    start_entry_finder.add(start_curr_node)
    goal_entry_finder.add(goal_curr_node)
    start_g = 0
    goal_g = 0
    start_h = euclidean_dist_heuristic(graph, start, goal)
    goal_h = euclidean_dist_heuristic(graph, goal, start)
    start_f = start_g + start_h
    goal_f = goal_g + goal_h

    while 1:
        start_children = graph[start_curr_node]
        for child in start_children:
            if child in start_entry_finder:
                for i in range(len(start_frontier.queue)):
                    entry = start_frontier.queue[i]
                    if entry[2] == child:
                        start_f = start_g + graph[start_curr_node][child]['weight'] + heuristic(graph, child, goal)
                        if start_f < entry[0]:
                            start_frontier.remove(i)
                            start_frontier.append((start_f, child))
                            start_child_parent[child] = start_curr_node
                        break
            else:
                start_entry_finder.add(child)
                start_f = start_g + graph[start_curr_node][child]['weight'] + heuristic(graph, child, goal)
                start_frontier.append((start_f, child))
                start_child_parent[child] = start_curr_node

        start_explored[start_curr_node] = start_g

        if len(start_frontier.queue):
            priority, start_curr_node = start_frontier.pop()
            while start_curr_node is start_frontier.REMOVED:
                priority, start_curr_node = start_frontier.pop()

        parent = start_child_parent[start_curr_node]
        start_g = start_explored[parent] + graph[parent][start_curr_node]['weight']

        goal_children = graph[goal_curr_node]
        for child in goal_children:
            if child in goal_entry_finder:
                for i in range(len(goal_frontier.queue)):
                    entry = goal_frontier.queue[i]
                    if entry[2] == child:
                        goal_f = goal_g + graph[goal_curr_node][child]['weight'] + heuristic(graph, child, start)
                        if goal_f < entry[0]:
                            goal_frontier.remove(i)
                            goal_frontier.append((goal_f, child))
                            goal_child_parent[child] = goal_curr_node
                        break
            else:
                goal_entry_finder.add(child)
                goal_f = goal_g + graph[goal_curr_node][child]['weight'] + heuristic(graph, child, start)
                goal_frontier.append((goal_f, child))
                goal_child_parent[child] = goal_curr_node

        goal_explored[goal_curr_node] = goal_g

        if len(goal_frontier.queue):
            priority, goal_curr_node = goal_frontier.pop()
            while goal_curr_node is goal_frontier.REMOVED:
                priority, goal_curr_node = goal_frontier.pop()

        parent = goal_child_parent[goal_curr_node]
        goal_g = goal_explored[parent] + graph[parent][goal_curr_node]['weight']

        start_explored_set = set(start_explored.keys())
        goal_explored_set = set(goal_explored.keys())
        intersection = start_explored_set.intersection(goal_explored_set)

        intersection = list(intersection)
        if intersection:
            inter_set = []
            for node in intersection:
                cost = start_explored[node] + goal_explored[node]
                inter_set.append([cost, node])
            inter_set = sorted(inter_set)
            intersection = inter_set[0][1]

            a = 0
            b = 0
            if intersection in start_explored:
                a = start_explored[intersection]
            if intersection in goal_explored:
                b = goal_explored[intersection]
            mu = a + b

            start_frontier_goal_explored = []
            for element in start_frontier.queue:
                if element[2] in goal_explored:
                    cost = element[0] + goal_explored[element[2]]
                    start_frontier_goal_explored.append(cost)
            goal_frontier_start_explored = []
            for element in goal_frontier.queue:
                if element[2] in start_explored:
                    cost = element[0] + start_explored[element[2]]
                    goal_frontier_start_explored.append(cost)
            x = float("inf")
            y = float("inf")
            if start_frontier_goal_explored:
                x = min(start_frontier_goal_explored)
            if goal_frontier_start_explored:
                y = goal_frontier_start_explored
            check = min(x, y)

            if check >= mu:
                break

    node = intersection
    while intersection in goal_child_parent:
        goal_path.append(node)
        node = goal_child_parent[node]
        if node == goal:
            break
    goal_path.append(goal)

    node = intersection
    while intersection in start_child_parent:
        start_path.append(node)
        node = start_child_parent[node]
        if node == start:
            break
    start_path.append(start)
    start_path.reverse()

    goal_path.pop(0)

    path = start_path + goal_path
    return path

    # TODO: finish this function!
    raise NotImplementedError


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search
    See README.MD for exercise description.
    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """

    goal12 = 0
    goal23 = 0
    goal31 = 0
    path = []
    path12 = []
    path23 = []
    path31 = []
    goal1_path = []
    start1_path = []
    goal2_path = []
    start2_path = []

    goal1 = goals[0]
    goal2 = goals[1]
    goal3 = goals[2]

    if goal1 == goal2:
        goal12 = 1
    if goal2 == goal3:
        goal23 = 1
    if goal3 == goal1:
        goal31 = 1

    goal1_child_parent = {}
    goal2_child_parent = {}
    goal3_child_parent = {}
    goal1_entry_finder = set()
    goal2_entry_finder = set()
    goal3_entry_finder = set()
    goal1_frontier = PriorityQueue()
    goal2_frontier = PriorityQueue()
    goal3_frontier = PriorityQueue()
    goal1_explored = {}
    goal2_explored = {}
    goal3_explored = {}
    goal1_curr_node = goal1
    goal2_curr_node = goal2
    goal3_curr_node = goal3
    goal1_entry_finder.add(goal1_curr_node)
    goal2_entry_finder.add(goal2_curr_node)
    goal3_entry_finder.add(goal3_curr_node)
    goal1_g = 0
    goal2_g = 0
    goal3_g = 0

    while 1:
        if not (goal12 and goal31):
            goal1_children = graph[goal1_curr_node]
            for child in goal1_children:
                if child in goal1_entry_finder:
                    for i in range(len(goal1_frontier.queue)):
                        entry = goal1_frontier.queue[i]
                        if entry[2] == child:
                            child_path_cost = goal1_g + graph[goal1_curr_node][child]['weight']
                            if child_path_cost < entry[0]:
                                goal1_frontier.remove(i)
                                goal1_frontier.append((child_path_cost, child))
                                goal1_child_parent[child] = goal1_curr_node
                            break
                else:
                    goal1_entry_finder.add(child)
                    child_path_cost = goal1_g + graph[goal1_curr_node][child]['weight']
                    goal1_frontier.append((child_path_cost, child))
                    goal1_child_parent[child] = goal1_curr_node

            goal1_explored[goal1_curr_node] = goal1_g

            if len(goal1_frontier.queue):
                priority, goal1_curr_node = goal1_frontier.pop()
                while goal1_curr_node is goal1_frontier.REMOVED:
                    priority, goal1_curr_node = goal1_frontier.pop()

            parent = goal1_child_parent[goal1_curr_node]
            goal1_g = goal1_explored[parent] + graph[parent][goal1_curr_node]['weight']

        if not (goal23 and goal12):
            goal2_children = graph[goal2_curr_node]
            for child in goal2_children:
                if child in goal2_entry_finder:
                    for i in range(len(goal2_frontier.queue)):
                        entry = goal2_frontier.queue[i]
                        if entry[2] == child:
                            child_path_cost = goal2_g + graph[goal2_curr_node][child]['weight']
                            if child_path_cost < entry[0]:
                                goal2_frontier.remove(i)
                                goal2_frontier.append((child_path_cost, child))
                                goal2_child_parent[child] = goal2_curr_node
                            break
                else:
                    goal2_entry_finder.add(child)
                    child_path_cost = goal2_g + graph[goal2_curr_node][child]['weight']
                    goal2_frontier.append((child_path_cost, child))
                    goal2_child_parent[child] = goal2_curr_node

            goal2_explored[goal2_curr_node] = goal2_g

            if len(goal2_frontier.queue):
                priority, goal2_curr_node = goal2_frontier.pop()
                while goal2_curr_node is goal2_frontier.REMOVED:
                    priority, goal2_curr_node = goal2_frontier.pop()

            parent = goal2_child_parent[goal2_curr_node]
            goal2_g = goal2_explored[parent] + graph[parent][goal2_curr_node]['weight']

        if not (goal31 and goal23):
            goal3_children = graph[goal3_curr_node]
            for child in goal3_children:
                if child in goal3_entry_finder:
                    for i in range(len(goal3_frontier.queue)):
                        entry = goal3_frontier.queue[i]
                        if entry[2] == child:
                            child_path_cost = goal3_g + graph[goal3_curr_node][child]['weight']
                            if child_path_cost < entry[0]:
                                goal3_frontier.remove(i)
                                goal3_frontier.append((child_path_cost, child))
                                goal3_child_parent[child] = goal3_curr_node
                            break
                else:
                    goal3_entry_finder.add(child)
                    child_path_cost = goal3_g + graph[goal3_curr_node][child]['weight']
                    goal3_frontier.append((child_path_cost, child))
                    goal3_child_parent[child] = goal3_curr_node

            goal3_explored[goal3_curr_node] = goal3_g

            if len(goal3_frontier.queue):
                priority, goal3_curr_node = goal3_frontier.pop()
                while goal3_curr_node is goal3_frontier.REMOVED:
                    priority, goal3_curr_node = goal3_frontier.pop()

            parent = goal3_child_parent[goal3_curr_node]
            goal3_g = goal3_explored[parent] + graph[parent][goal3_curr_node]['weight']

        goal1_explored_set = set(goal1_explored.keys())
        goal2_explored_set = set(goal2_explored.keys())
        goal3_explored_set = set(goal3_explored.keys())

        intersection12 = goal1_explored_set.intersection(goal2_explored_set)
        intersection23 = goal2_explored_set.intersection(goal3_explored_set)
        intersection31 = goal3_explored_set.intersection(goal1_explored_set)

        if intersection12:
            inter_set12 = []
            for node in intersection12:
                cost12 = goal1_explored[node] + goal2_explored[node]
                inter_set12.append([cost12, node])
            inter_set12 = sorted(inter_set12)
            if inter_set12:
                intersection12 = inter_set12[0][1]
                cost12 = inter_set12[0][0]
            elif not goal12:
                cost12 = float("inf")
            elif goal12:
                cost12 = 0

            a = 0
            b = 0
            if intersection12 in goal1_explored:
                a = goal1_explored[intersection12]
            if intersection12 in goal2_explored:
                b = goal2_explored[intersection12]
            mu = a + b

            goal1_frontier_goal2_explored = []
            for element in goal1_frontier.queue:
                if element[2] in goal2_explored:
                    cost = element[0] + goal2_explored[element[2]]
                    goal1_frontier_goal2_explored.append(cost)
            goal2_frontier_goal1_explored = []
            for element in goal2_frontier.queue:
                if element[2] in goal1_explored:
                    cost = element[0] + goal1_explored[element[2]]
                    goal2_frontier_goal1_explored.append(cost)
            x = float("inf")
            y = float("inf")
            if goal1_frontier_goal2_explored:
                x = min(goal1_frontier_goal2_explored)
            if goal2_frontier_goal1_explored:
                y = goal2_frontier_goal1_explored
            check = min(x, y)

            if check >= mu:
                goal12 = 1

        if intersection23:
            inter_set23 = []
            for node in intersection23:
                cost23 = goal2_explored[node] + goal3_explored[node]
                inter_set23.append([cost23, node])
            inter_set23 = sorted(inter_set23)
            if inter_set23:
                intersection23 = inter_set23[0][1]
                cost23 = inter_set23[0][0]
            elif not goal23:
                cost23 = float("inf")
            elif goal23:
                cost23 = 0

            a = 0
            b = 0
            if intersection23 in goal2_explored:
                a = goal2_explored[intersection23]
            if intersection23 in goal3_explored:
                b = goal3_explored[intersection23]
            mu = a + b

            goal2_frontier_goal3_explored = []
            for element in goal2_frontier.queue:
                if element[2] in goal3_explored:
                    cost = element[0] + goal3_explored[element[2]]
                    goal2_frontier_goal3_explored.append(cost)
            goal3_frontier_goal2_explored = []
            for element in goal3_frontier.queue:
                if element[2] in goal2_explored:
                    cost = element[0] + goal2_explored[element[2]]
                    goal3_frontier_goal2_explored.append(cost)
            x = float("inf")
            y = float("inf")
            if goal2_frontier_goal3_explored:
                x = min(goal2_frontier_goal3_explored)
            if goal3_frontier_goal2_explored:
                y = goal3_frontier_goal2_explored
            check = min(x, y)

            if check >= mu:
                goal23 = 1

        if intersection31:
            inter_set31 = []
            for node in intersection31:
                cost31 = goal3_explored[node] + goal1_explored[node]
                inter_set31.append([cost31, node])
            inter_set31 = sorted(inter_set31)
            if inter_set31:
                intersection31 = inter_set31[0][1]
                cost31 = inter_set31[0][0]
            elif not goal31:
                cost31 = float("inf")
            elif goal31:
                cost31 = 0

            a = 0
            b = 0
            if intersection31 in goal3_explored:
                a = goal3_explored[intersection31]
            if intersection31 in goal1_explored:
                b = goal1_explored[intersection31]
            mu = a + b

            goal3_frontier_goal1_explored = []
            for element in goal3_frontier.queue:
                if element[2] in goal1_explored:
                    cost = element[0] + goal1_explored[element[2]]
                    goal3_frontier_goal1_explored.append(cost)
            goal1_frontier_goal3_explored = []
            for element in goal1_frontier.queue:
                if element[2] in goal3_explored:
                    cost = element[0] + goal3_explored[element[2]]
                    goal1_frontier_goal3_explored.append(cost)
            x = float("inf")
            y = float("inf")
            if goal3_frontier_goal1_explored:
                x = min(goal3_frontier_goal1_explored)
            if goal1_frontier_goal3_explored:
                y = goal1_frontier_goal3_explored
            check = min(x, y)

            if check >= mu:
                goal31 = 1

        if goal12 and goal23 and goal31:
            break

    if (goal1 == goal2) or (goal2 == goal3) or (goal3 == goal1):

        if (goal1 == goal2) and (goal2 == goal3):
            path = []
            return path

        if goal1 == goal2:
            path12 = []
            node = intersection23
            while intersection23 in goal3_child_parent:
                goal2_path.append(node)
                node = goal3_child_parent[node]
                if node == goal3:
                    break
            goal2_path.append(goal3)

            node = intersection23
            while intersection23 in goal2_child_parent:
                start2_path.append(node)
                node = goal2_child_parent[node]
                if node == goal2:
                    break
            start2_path.append(goal2)
            start2_path.reverse()

            goal2_path.pop(0)

            path23 = start2_path + goal2_path

            path = path12 + path23
            return path

        if goal2 == goal3:
            path23 = []
            node = intersection31
            while intersection31 in goal1_child_parent:
                goal1_path.append(node)
                node = goal1_child_parent[node]
                if node == goal1:
                    break
            goal1_path.append(goal1)

            node = intersection31
            while intersection31 in goal3_child_parent:
                start1_path.append(node)
                node = goal3_child_parent[node]
                if node == goal3:
                    break
            start1_path.append(goal3)
            start1_path.reverse()

            goal1_path.pop(0)

            path31 = start1_path + goal1_path

            path = path23 + path31
            return path

        if goal1 == goal3:
            path31 = []
            node = intersection12
            while intersection12 in goal2_child_parent:
                goal1_path.append(node)
                node = goal2_child_parent[node]
                if node == goal2:
                    break
            goal1_path.append(goal2)

            node = intersection12
            while intersection12 in goal1_child_parent:
                start1_path.append(node)
                node = goal1_child_parent[node]
                if node == goal1:
                    break
            start1_path.append(goal1)
            start1_path.reverse()

            goal1_path.pop(0)

            path12 = start1_path + goal1_path

            path = path31 + path12
            return path

    if (cost12 <= cost31) and (cost23 <= cost31):
        node = intersection12
        while intersection12 in goal2_child_parent:
            goal1_path.append(node)
            node = goal2_child_parent[node]
            if node == goal2:
                break
        goal1_path.append(goal2)

        node = intersection12
        while intersection12 in goal1_child_parent:
            start1_path.append(node)
            node = goal1_child_parent[node]
            if node == goal1:
                break
        start1_path.append(goal1)
        start1_path.reverse()

        goal1_path.pop(0)

        path12 = start1_path + goal1_path

        node = intersection23
        while intersection23 in goal3_child_parent:
            goal2_path.append(node)
            node = goal3_child_parent[node]
            if node == goal3:
                break
        goal2_path.append(goal3)

        node = intersection23
        while intersection23 in goal2_child_parent:
            start2_path.append(node)
            node = goal2_child_parent[node]
            if node == goal2:
                break
        start2_path.reverse()

        goal2_path.pop(0)

        path23 = start2_path + goal2_path

        path = path12 + path23
        return path

    if (cost12 <= cost23) and (cost31 <= cost23):
        node = intersection31
        while intersection31 in goal1_child_parent:
            goal1_path.append(node)
            node = goal1_child_parent[node]
            if node == goal1:
                break
        goal1_path.append(goal1)

        node = intersection31
        while intersection31 in goal3_child_parent:
            start1_path.append(node)
            node = goal3_child_parent[node]
            if node == goal3:
                break
        start1_path.append(goal3)
        start1_path.reverse()

        goal1_path.pop(0)

        path31 = start1_path + goal1_path

        node = intersection12
        while intersection12 in goal2_child_parent:
            goal2_path.append(node)
            node = goal2_child_parent[node]
            if node == goal2:
                break
        goal2_path.append(goal2)

        node = intersection12
        while intersection12 in goal1_child_parent:
            start2_path.append(node)
            node = goal1_child_parent[node]
            if node == goal1:
                break
        start2_path.reverse()

        goal2_path.pop(0)

        path12 = start2_path + goal2_path

        path = path31 + path12
        return path

    if (cost23 <= cost12) and (cost31 <= cost12):
        node = intersection23
        while intersection23 in goal3_child_parent:
            goal1_path.append(node)
            node = goal3_child_parent[node]
            if node == goal3:
                break
        goal1_path.append(goal3)

        node = intersection23
        while intersection23 in goal2_child_parent:
            start1_path.append(node)
            node = goal2_child_parent[node]
            if node == goal2:
                break
        start1_path.append(goal2)
        start1_path.reverse()

        goal1_path.pop(0)

        path23 = start1_path + goal1_path

        node = intersection31
        while intersection31 in goal1_child_parent:
            goal2_path.append(node)
            node = goal1_child_parent[node]
            if node == goal1:
                break
        goal2_path.append(goal1)

        node = intersection31
        while intersection31 in goal3_child_parent:
            start2_path.append(node)
            node = goal3_child_parent[node]
            if node == goal3:
                break
        start2_path.reverse()

        goal2_path.pop(0)

        path31 = start2_path + goal2_path

        path = path23 + path31
        return path


    # TODO: finish this function
    raise NotImplementedError


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
    """
    Exercise 3: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """

    goal12 = 0
    goal23 = 0
    goal31 = 0
    path = []
    path12 = []
    path23 = []
    path31 = []
    goal1_path = []
    start1_path = []
    goal2_path = []
    start2_path = []

    goal1 = goals[0]
    goal2 = goals[1]
    goal3 = goals[2]

    if goal1 == goal2:
        goal12 = 1
    if goal2 == goal3:
        goal23 = 1
    if goal3 == goal1:
        goal31 = 1

    goal1_child_parent = {}
    goal2_child_parent = {}
    goal3_child_parent = {}
    goal1_entry_finder = set()
    goal2_entry_finder = set()
    goal3_entry_finder = set()
    goal1_frontier = PriorityQueue()
    goal2_frontier = PriorityQueue()
    goal3_frontier = PriorityQueue()
    goal1_explored = {}
    goal2_explored = {}
    goal3_explored = {}
    goal1_curr_node = goal1
    goal2_curr_node = goal2
    goal3_curr_node = goal3
    goal1_entry_finder.add(goal1_curr_node)
    goal2_entry_finder.add(goal2_curr_node)
    goal3_entry_finder.add(goal3_curr_node)
    goal1_g = 0
    goal2_g = 0
    goal3_g = 0
    goal1_h2 = euclidean_dist_heuristic(graph, goal1, goal2)
    goal1_h3 = euclidean_dist_heuristic(graph, goal1, goal3)
    goal2_h1 = euclidean_dist_heuristic(graph, goal2, goal1)
    goal2_h3 = euclidean_dist_heuristic(graph, goal2, goal3)
    goal3_h1 = euclidean_dist_heuristic(graph, goal3, goal1)
    goal3_h2 = euclidean_dist_heuristic(graph, goal3, goal2)
    f12 = goal1_g + goal1_h2
    f13 = goal1_g + goal1_h3
    f23 = goal2_g + goal2_h3
    f21 = goal2_g + goal2_h1
    f31 = goal3_g + goal3_h1
    f32 = goal3_g + goal3_h2

    while 1:
        if not (goal12 and goal31):
            goal1_children = graph[goal1_curr_node]
            for child in goal1_children:
                if child in goal1_entry_finder:
                    for i in range(len(goal1_frontier.queue)):
                        entry = goal1_frontier.queue[i]
                        if entry[2] == child:
                            f12 = goal1_g + graph[goal1_curr_node][child]['weight'] + heuristic(graph, child, goal2)
                            f13 = goal1_g + graph[goal1_curr_node][child]['weight'] + heuristic(graph, child, goal3)
                            f = min(f12, f13)
                            if f < entry[0]:
                                goal1_frontier.remove(i)
                                goal1_frontier.append((f, child))
                                goal1_child_parent[child] = goal1_curr_node
                            break
                else:
                    goal1_entry_finder.add(child)
                    f12 = goal1_g + graph[goal1_curr_node][child]['weight'] + heuristic(graph, child, goal2)
                    f13 = goal1_g + graph[goal1_curr_node][child]['weight'] + heuristic(graph, child, goal3)
                    f = min(f12, f13)
                    goal1_frontier.append((f, child))
                    goal1_child_parent[child] = goal1_curr_node

            goal1_explored[goal1_curr_node] = goal1_g

            if len(goal1_frontier.queue):
                priority, goal1_curr_node = goal1_frontier.pop()
                while goal1_curr_node is goal1_frontier.REMOVED:
                    priority, goal1_curr_node = goal1_frontier.pop()

            parent = goal1_child_parent[goal1_curr_node]
            goal1_g = goal1_explored[parent] + graph[parent][goal1_curr_node]['weight']

        if not (goal23 and goal12):
            goal2_children = graph[goal2_curr_node]
            for child in goal2_children:
                if child in goal2_entry_finder:
                    for i in range(len(goal2_frontier.queue)):
                        entry = goal2_frontier.queue[i]
                        if entry[2] == child:
                            f23 = goal2_g + graph[goal2_curr_node][child]['weight'] + heuristic(graph, child, goal3)
                            f21 = goal2_g + graph[goal2_curr_node][child]['weight'] + heuristic(graph, child, goal1)
                            f = min(f23, f21)
                            if f < entry[0]:
                                goal2_frontier.remove(i)
                                goal2_frontier.append((f, child))
                                goal2_child_parent[child] = goal2_curr_node
                            break
                else:
                    goal2_entry_finder.add(child)
                    f23 = goal2_g + graph[goal2_curr_node][child]['weight'] + heuristic(graph, child, goal3)
                    f21 = goal2_g + graph[goal2_curr_node][child]['weight'] + heuristic(graph, child, goal1)
                    f = min(f23, f21)
                    goal2_frontier.append((f, child))
                    goal2_child_parent[child] = goal2_curr_node

            goal2_explored[goal2_curr_node] = goal2_g

            if len(goal2_frontier.queue):
                priority, goal2_curr_node = goal2_frontier.pop()
                while goal2_curr_node is goal2_frontier.REMOVED:
                    priority, goal2_curr_node = goal2_frontier.pop()

            parent = goal2_child_parent[goal2_curr_node]
            goal2_g = goal2_explored[parent] + graph[parent][goal2_curr_node]['weight']

        if not (goal31 and goal23):
            goal3_children = graph[goal3_curr_node]
            for child in goal3_children:
                if child in goal3_entry_finder:
                    for i in range(len(goal3_frontier.queue)):
                        entry = goal3_frontier.queue[i]
                        if entry[2] == child:
                            f31 = goal3_g + graph[goal3_curr_node][child]['weight'] + heuristic(graph, child, goal1)
                            f32 = goal3_g + graph[goal3_curr_node][child]['weight'] + heuristic(graph, child, goal2)
                            f = min(f31, f32)
                            if f < entry[0]:
                                goal3_frontier.remove(i)
                                goal3_frontier.append((f, child))
                                goal3_child_parent[child] = goal3_curr_node
                            break
                else:
                    goal3_entry_finder.add(child)
                    f31 = goal3_g + graph[goal3_curr_node][child]['weight'] + heuristic(graph, child, goal1)
                    f32 = goal3_g + graph[goal3_curr_node][child]['weight'] + heuristic(graph, child, goal2)
                    f = min(f31, f32)
                    goal3_frontier.append((f, child))
                    goal3_child_parent[child] = goal3_curr_node

            goal3_explored[goal3_curr_node] = goal3_g

            if len(goal3_frontier.queue):
                priority, goal3_curr_node = goal3_frontier.pop()
                while goal3_curr_node is goal3_frontier.REMOVED:
                    priority, goal3_curr_node = goal3_frontier.pop()

            parent = goal3_child_parent[goal3_curr_node]
            goal3_g = goal3_explored[parent] + graph[parent][goal3_curr_node]['weight']

        goal1_explored_set = set(goal1_explored.keys())
        goal2_explored_set = set(goal2_explored.keys())
        goal3_explored_set = set(goal3_explored.keys())

        intersection12 = goal1_explored_set.intersection(goal2_explored_set)
        intersection23 = goal2_explored_set.intersection(goal3_explored_set)
        intersection31 = goal3_explored_set.intersection(goal1_explored_set)

        if intersection12:
            inter_set12 = []
            for node in intersection12:
                cost12 = goal1_explored[node] + goal2_explored[node]
                inter_set12.append([cost12, node])
            inter_set12 = sorted(inter_set12)
            if inter_set12:
                intersection12 = inter_set12[0][1]
                cost12 = inter_set12[0][0]
            elif not goal12:
                cost12 = float("inf")
            elif goal12:
                cost12 = 0

            a = 0
            b = 0
            if intersection12 in goal1_explored:
                a = goal1_explored[intersection12]
            if intersection12 in goal2_explored:
                b = goal2_explored[intersection12]
            mu = a + b

            goal1_frontier_goal2_explored = []
            for element in goal1_frontier.queue:
                if element[2] in goal2_explored:
                    cost = element[0] + goal2_explored[element[2]]
                    goal1_frontier_goal2_explored.append(cost)
            goal2_frontier_goal1_explored = []
            for element in goal2_frontier.queue:
                if element[2] in goal1_explored:
                    cost = element[0] + goal1_explored[element[2]]
                    goal2_frontier_goal1_explored.append(cost)
            x = float("inf")
            y = float("inf")
            if goal1_frontier_goal2_explored:
                x = min(goal1_frontier_goal2_explored)
            if goal2_frontier_goal1_explored:
                y = goal2_frontier_goal1_explored
            check = min(x, y)

            if check >= mu:
                goal12 = 1

        if intersection23:
            inter_set23 = []
            for node in intersection23:
                cost23 = goal2_explored[node] + goal3_explored[node]
                inter_set23.append([cost23, node])
            inter_set23 = sorted(inter_set23)
            if inter_set23:
                intersection23 = inter_set23[0][1]
                cost23 = inter_set23[0][0]
            elif not goal23:
                cost23 = float("inf")
            elif goal23:
                cost23 = 0

            a = 0
            b = 0
            if intersection23 in goal2_explored:
                a = goal2_explored[intersection23]
            if intersection23 in goal3_explored:
                b = goal3_explored[intersection23]
            mu = a + b

            goal2_frontier_goal3_explored = []
            for element in goal2_frontier.queue:
                if element[2] in goal3_explored:
                    cost = element[0] + goal3_explored[element[2]]
                    goal2_frontier_goal3_explored.append(cost)
            goal3_frontier_goal2_explored = []
            for element in goal3_frontier.queue:
                if element[2] in goal2_explored:
                    cost = element[0] + goal2_explored[element[2]]
                    goal3_frontier_goal2_explored.append(cost)
            x = float("inf")
            y = float("inf")
            if goal2_frontier_goal3_explored:
                x = min(goal2_frontier_goal3_explored)
            if goal3_frontier_goal2_explored:
                y = goal3_frontier_goal2_explored
            check = min(x, y)

            if check >= mu:
                goal23 = 1

        if intersection31:
            inter_set31 = []
            for node in intersection31:
                cost31 = goal3_explored[node] + goal1_explored[node]
                inter_set31.append([cost31, node])
            inter_set31 = sorted(inter_set31)
            if inter_set31:
                intersection31 = inter_set31[0][1]
                cost31 = inter_set31[0][0]
            elif not goal31:
                cost31 = float("inf")
            elif goal31:
                cost31 = 0

            a = 0
            b = 0
            if intersection31 in goal3_explored:
                a = goal3_explored[intersection31]
            if intersection31 in goal1_explored:
                b = goal1_explored[intersection31]
            mu = a + b

            goal3_frontier_goal1_explored = []
            for element in goal3_frontier.queue:
                if element[2] in goal1_explored:
                    cost = element[0] + goal1_explored[element[2]]
                    goal3_frontier_goal1_explored.append(cost)
            goal1_frontier_goal3_explored = []
            for element in goal1_frontier.queue:
                if element[2] in goal3_explored:
                    cost = element[0] + goal3_explored[element[2]]
                    goal1_frontier_goal3_explored.append(cost)
            x = float("inf")
            y = float("inf")
            if goal3_frontier_goal1_explored:
                x = min(goal3_frontier_goal1_explored)
            if goal1_frontier_goal3_explored:
                y = goal1_frontier_goal3_explored
            check = min(x, y)

            if check >= mu:
                goal31 = 1

        if goal12 and goal23 and goal31:
            break

    if (goal1 == goal2) or (goal2 == goal3) or (goal3 == goal1):

        if (goal1 == goal2) and (goal2 == goal3):
            path = []
            return path

        if goal1 == goal2:
            path12 = []
            node = intersection23
            while intersection23 in goal3_child_parent:
                goal2_path.append(node)
                node = goal3_child_parent[node]
                if node == goal3:
                    break
            goal2_path.append(goal3)

            node = intersection23
            while intersection23 in goal2_child_parent:
                start2_path.append(node)
                node = goal2_child_parent[node]
                if node == goal2:
                    break
            start2_path.append(goal2)
            start2_path.reverse()

            goal2_path.pop(0)

            path23 = start2_path + goal2_path

            path = path12 + path23
            return path

        if goal2 == goal3:
            path23 = []
            node = intersection31
            while intersection31 in goal1_child_parent:
                goal1_path.append(node)
                node = goal1_child_parent[node]
                if node == goal1:
                    break
            goal1_path.append(goal1)

            node = intersection31
            while intersection31 in goal3_child_parent:
                start1_path.append(node)
                node = goal3_child_parent[node]
                if node == goal3:
                    break
            start1_path.append(goal3)
            start1_path.reverse()

            goal1_path.pop(0)

            path31 = start1_path + goal1_path

            path = path23 + path31
            return path

        if goal1 == goal3:
            path31 = []
            node = intersection12
            while intersection12 in goal2_child_parent:
                goal1_path.append(node)
                node = goal2_child_parent[node]
                if node == goal2:
                    break
            goal1_path.append(goal2)

            node = intersection12
            while intersection12 in goal1_child_parent:
                start1_path.append(node)
                node = goal1_child_parent[node]
                if node == goal1:
                    break
            start1_path.append(goal1)
            start1_path.reverse()

            goal1_path.pop(0)

            path12 = start1_path + goal1_path

            path = path31 + path12
            return path

    if (cost12 <= cost31) and (cost23 <= cost31):
        node = intersection12
        while intersection12 in goal2_child_parent:
            goal1_path.append(node)
            node = goal2_child_parent[node]
            if node == goal2:
                break
        goal1_path.append(goal2)

        node = intersection12
        while intersection12 in goal1_child_parent:
            start1_path.append(node)
            node = goal1_child_parent[node]
            if node == goal1:
                break
        start1_path.append(goal1)
        start1_path.reverse()

        goal1_path.pop(0)

        path12 = start1_path + goal1_path

        node = intersection23
        while intersection23 in goal3_child_parent:
            goal2_path.append(node)
            node = goal3_child_parent[node]
            if node == goal3:
                break
        goal2_path.append(goal3)

        node = intersection23
        while intersection23 in goal2_child_parent:
            start2_path.append(node)
            node = goal2_child_parent[node]
            if node == goal2:
                break
        start2_path.reverse()

        goal2_path.pop(0)

        path23 = start2_path + goal2_path

        path = path12 + path23
        return path

    if (cost12 <= cost23) and (cost31 <= cost23):
        node = intersection31
        while intersection31 in goal1_child_parent:
            goal1_path.append(node)
            node = goal1_child_parent[node]
            if node == goal1:
                break
        goal1_path.append(goal1)

        node = intersection31
        while intersection31 in goal3_child_parent:
            start1_path.append(node)
            node = goal3_child_parent[node]
            if node == goal3:
                break
        start1_path.append(goal3)
        start1_path.reverse()

        goal1_path.pop(0)

        path31 = start1_path + goal1_path

        node = intersection12
        while intersection12 in goal2_child_parent:
            goal2_path.append(node)
            node = goal2_child_parent[node]
            if node == goal2:
                break
        goal2_path.append(goal2)

        node = intersection12
        while intersection12 in goal1_child_parent:
            start2_path.append(node)
            node = goal1_child_parent[node]
            if node == goal1:
                break
        start2_path.reverse()

        goal2_path.pop(0)

        path12 = start2_path + goal2_path

        path = path31 + path12
        return path

    if (cost23 <= cost12) and (cost31 <= cost12):
        node = intersection23
        while intersection23 in goal3_child_parent:
            goal1_path.append(node)
            node = goal3_child_parent[node]
            if node == goal3:
                break
        goal1_path.append(goal3)

        node = intersection23
        while intersection23 in goal2_child_parent:
            start1_path.append(node)
            node = goal2_child_parent[node]
            if node == goal2:
                break
        start1_path.append(goal2)
        start1_path.reverse()

        goal1_path.pop(0)

        path23 = start1_path + goal1_path

        node = intersection31
        while intersection31 in goal1_child_parent:
            goal2_path.append(node)
            node = goal1_child_parent[node]
            if node == goal1:
                break
        goal2_path.append(goal1)

        node = intersection31
        while intersection31 in goal3_child_parent:
            start2_path.append(node)
            node = goal3_child_parent[node]
            if node == goal3:
                break
        start2_path.reverse()

        goal2_path.pop(0)

        path31 = start2_path + goal2_path

        path = path23 + path31
        return path

    # TODO: finish this function
    raise NotImplementedError


def return_your_name():
    """Return your name from this function"""

    return "Dhaneshwaran Jotheeswaran"
    # TODO: finish this function
    raise NotImplementedError


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    path = []
    start_path = []
    goal_path = []
    start_child_parent = {}
    goal_child_parent = {}
    if start == goal:
        return path

    start_entry_finder = set()
    goal_entry_finder = set()
    start_frontier = PriorityQueue()
    goal_frontier = PriorityQueue()
    start_explored = {}
    goal_explored = {}
    start_curr_node = start
    goal_curr_node = goal
    start_entry_finder.add(start_curr_node)
    goal_entry_finder.add(goal_curr_node)
    start_g = 0
    goal_g = 0
    start_h = euclidean_dist_heuristic(graph, start, goal)
    goal_h = euclidean_dist_heuristic(graph, goal, start)
    start_f = start_g + start_h
    goal_f = goal_g + goal_h

    while 1:
        start_children = graph[start_curr_node]
        for child in start_children:
            if child in start_entry_finder:
                child = str(child)
                for i in range(len(start_frontier.queue)):
                    entry = start_frontier.queue[i]
                    if entry[2] == child:
                        start_f = start_g + graph[start_curr_node][child]['weight'] + heuristic(graph, child, goal)
                        if start_f < entry[0]:
                            start_frontier.remove(i)
                            start_frontier.append((start_f, child))
                            start_child_parent[child] = start_curr_node
                        break
            else:
                start_entry_finder.add(child)
                start_f = start_g + graph[start_curr_node][child]['weight'] + heuristic(graph, child, goal)
                start_frontier.append((start_f, child))
                start_child_parent[child] = start_curr_node

        start_explored[start_curr_node] = start_g

        if len(start_frontier.queue):
            priority, start_curr_node = start_frontier.pop()
            while start_curr_node is start_frontier.REMOVED:
                priority, start_curr_node = start_frontier.pop()

        parent = start_child_parent[start_curr_node]
        start_g = start_explored[parent] + graph[parent][start_curr_node]['weight']

        goal_children = graph[goal_curr_node]
        for child in goal_children:
            child = str(child)
            if child in goal_entry_finder:
                for i in range(len(goal_frontier.queue)):
                    entry = goal_frontier.queue[i]
                    if entry[2] == child:
                        goal_f = goal_g + graph[goal_curr_node][child]['weight'] + heuristic(graph, child, start)
                        if goal_f < entry[0]:
                            goal_frontier.remove(i)
                            goal_frontier.append((goal_f, child))
                            goal_child_parent[child] = goal_curr_node
                        break
            else:
                goal_entry_finder.add(child)
                goal_f = goal_g + graph[goal_curr_node][child]['weight'] + heuristic(graph, child, start)
                goal_frontier.append((goal_f, child))
                goal_child_parent[child] = goal_curr_node

        goal_explored[goal_curr_node] = goal_g

        if len(goal_frontier.queue):
            priority, goal_curr_node = goal_frontier.pop()
            while goal_curr_node is goal_frontier.REMOVED:
                priority, goal_curr_node = goal_frontier.pop()

        parent = goal_child_parent[goal_curr_node]
        goal_g = goal_explored[parent] + graph[parent][goal_curr_node]['weight']

        start_explored_set = set(start_explored.keys())
        goal_explored_set = set(goal_explored.keys())
        intersection = start_explored_set.intersection(goal_explored_set)

        intersection = list(intersection)
        if intersection:
            inter_set = []
            for node in intersection:
                cost = start_explored[node] + goal_explored[node]
                inter_set.append([cost, node])
            inter_set = sorted(inter_set)
            intersection = inter_set[0][1]

            a = 0
            b = 0
            if intersection in start_explored:
                a = start_explored[intersection]
            if intersection in goal_explored:
                b = goal_explored[intersection]
            mu = a + b

            start_frontier_goal_explored = []
            for element in start_frontier.queue:
                if element[2] in goal_explored:
                    cost = element[0] + goal_explored[element[2]]
                    start_frontier_goal_explored.append(cost)
            goal_frontier_start_explored = []
            for element in goal_frontier.queue:
                if element[2] in start_explored:
                    cost = element[0] + start_explored[element[2]]
                    goal_frontier_start_explored.append(cost)
            x = float("inf")
            y = float("inf")
            if start_frontier_goal_explored:
                x = min(start_frontier_goal_explored)
            if goal_frontier_start_explored:
                y = goal_frontier_start_explored
            check = min(x, y)

            if check >= mu:
                break

    node = intersection
    while intersection in goal_child_parent:
        goal_path.append(node)
        node = goal_child_parent[node]
        if node == goal:
            break
    goal_path.append(goal)

    node = intersection
    while intersection in start_child_parent:
        start_path.append(node)
        node = start_child_parent[node]
        if node == start:
            break
    start_path.append(start)
    start_path.reverse()

    goal_path.pop(0)

    path = start_path + goal_path
    return path

    # TODO: finish this function!
    raise NotImplementedError


def load_data():
    """
    Loads data from data.pickle and return the data object that is passed to
    the custom_search method.

    Will be called only once. Feel free to modify.

    Returns:
         The data loaded from the pickle file.
    """

    dir_name = os.path.dirname(os.path.realpath(__file__))
    pickle_file_path = os.path.join(dir_name, "data.pickle")
    data = pickle.load(open(pickle_file_path, 'rb'))
    return data
