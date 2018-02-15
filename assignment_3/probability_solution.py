"""Testing pbnt. Run this before anything else to get pbnt to work!"""
import sys

if('pbnt/combined' not in sys.path):
    sys.path.append('pbnt/combined')
from exampleinference import inferenceExample

inferenceExample()
# Should output:
# ('The marginal probability of sprinkler=false:', 0.80102921)
#('The marginal probability of wetgrass=false | cloudy=False, rain=True:', 0.055)

'''
WRITE YOUR CODE BELOW. DO NOT CHANGE ANY FUNCTION HEADERS FROM THE NOTEBOOK.
'''


from Node import BayesNode
from Graph import BayesNet
from numpy import zeros, float32
import Distribution
from Distribution import DiscreteDistribution, ConditionalDiscreteDistribution
from Inference import JunctionTreeEngine, EnumerationEngine
import random


def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge",
    "temperature". (for the tests to work.)
    """
    nodenames = ["temperature", "faulty gauge", "gauge", "faulty alarm", "alarm"]

    T_node = BayesNode(0, 2, name=nodenames[0])
    Fg_node = BayesNode(1, 2, name=nodenames[1])
    G_node = BayesNode(2, 2, name=nodenames[2])
    Fa_node = BayesNode(3, 2, name=nodenames[3])
    A_node = BayesNode(4, 2, name=nodenames[4])

    T_node.add_child(Fg_node)
    T_node.add_child(G_node)
    Fg_node.add_parent(T_node)
    Fg_node.add_child(G_node)
    G_node.add_parent(Fg_node)
    G_node.add_parent(T_node)
    G_node.add_child(A_node)
    Fa_node.add_child(A_node)
    A_node.add_parent(G_node)
    A_node.add_parent(Fa_node)

    nodes = [T_node, Fg_node, G_node, Fa_node, A_node]

    return BayesNet(nodes)

    # TODO: finish this function    
    raise NotImplementedError


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system."""    
    A_node = bayes_net.get_node_by_name("alarm")
    F_A_node = bayes_net.get_node_by_name("faulty alarm")
    G_node = bayes_net.get_node_by_name("gauge")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")
    nodes = [A_node, F_A_node, G_node, F_G_node, T_node]

    T_distribution = DiscreteDistribution(T_node)
    index = T_distribution.generate_index([], [])
    T_distribution[index] = [0.80, 0.20]
    T_node.set_dist(T_distribution)

    dist = zeros([T_node.size(), F_G_node.size()], dtype=float32)
    dist[0, :] = [0.95, 0.05]
    dist[1, :] = [0.20, 0.80]
    F_G_distribution = ConditionalDiscreteDistribution(nodes=[T_node, F_G_node], table=dist)
    F_G_node.set_dist(F_G_distribution)

    dist = zeros([T_node.size(), F_G_node.size(), G_node.size()], dtype=float32)
    dist[0, 0, :] = [0.95, 0.05]
    dist[0, 1, :] = [0.20, 0.80]
    dist[1, 0, :] = [0.05, 0.95]
    dist[1, 1, :] = [0.80, 0.20]
    G_distribution = ConditionalDiscreteDistribution(nodes=[T_node, F_G_node, G_node], table=dist)
    G_node.set_dist(G_distribution)

    F_A_distribution = DiscreteDistribution(F_A_node)
    index = F_A_distribution.generate_index([], [])
    F_A_distribution[index] = [0.85, 0.15]
    F_A_node.set_dist(F_A_distribution)

    dist = zeros([G_node.size(), F_A_node.size(), A_node.size()], dtype=float32)
    dist[0, 0, :] = [0.90, 0.10]
    dist[0, 1, :] = [0.55, 0.45]
    dist[1, 0, :] = [0.10, 0.90]
    dist[1, 1, :] = [0.45, 0.55]
    A_distribution = ConditionalDiscreteDistribution(nodes=[G_node, F_A_node, A_node], table=dist)
    A_node.set_dist(A_distribution)

    bayes_net = BayesNet(nodes)

    return bayes_net

    # TODO: set the probability distribution for each node
    raise NotImplementedError


def get_alarm_prob(bayes_net, alarm_rings):
    """Calculate the marginal 
    probability of the alarm 
    ringing (T/F) in the 
    power plant system."""

    A_node = bayes_net.get_node_by_name("alarm")
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(A_node)[0]
    index = Q.generate_index([True], range(Q.nDims))
    alarm_prob = Q[index]

    return alarm_prob

    # TODO: finish this function
    raise NotImplementedError


def get_gauge_prob(bayes_net, gauge_hot):
    """Calculate the marginal
    probability of the gauge 
    showing hot (T/F) in the 
    power plant system."""

    G_node = bayes_net.get_node_by_name("gauge")
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(G_node)[0]
    index = Q.generate_index([True], range(Q.nDims))
    gauge_prob = Q[index]

    return gauge_prob

    # TODO: finish this function
    raise NotImplementedError


def get_temperature_prob(bayes_net,temp_hot):
    """Calculate the conditional probability 
    of the temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""

    T_node = bayes_net.get_node_by_name("temperature")
    A_node = bayes_net.get_node_by_name("alarm")
    F_A_node = bayes_net.get_node_by_name("faulty alarm")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[A_node] = True
    engine.evidence[F_A_node] = False
    engine.evidence[F_G_node] = False
    Q = engine.marginal(T_node)[0]
    index = Q.generate_index([True], range(Q.nDims))
    temp_prob = Q[index]

    return temp_prob

    # TODO: finish this function
    raise NotImplementedError


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """

    nodenames = ["A", "B", "C", "AvB", "BvC", "CvA"]

    A_node = BayesNode(0, 4, name=nodenames[0])
    B_node = BayesNode(1, 4, name=nodenames[1])
    C_node = BayesNode(2, 4, name=nodenames[2])
    AvB_node = BayesNode(3, 3, name=nodenames[3])
    BvC_node = BayesNode(4, 3, name=nodenames[4])
    CvA_node = BayesNode(5, 3, name=nodenames[5])

    A_node.add_child(AvB_node)
    A_node.add_child(CvA_node)
    B_node.add_child(AvB_node)
    B_node.add_child(BvC_node)
    C_node.add_child(BvC_node)
    C_node.add_child(CvA_node)
    AvB_node.add_parent(A_node)
    AvB_node.add_parent(B_node)
    BvC_node.add_parent(B_node)
    BvC_node.add_parent(C_node)
    CvA_node.add_parent(C_node)
    CvA_node.add_parent(A_node)

    A_distribution = DiscreteDistribution(A_node)
    index = A_distribution.generate_index([], [])
    A_distribution[index] = [0.15, 0.45, 0.30, 0.10]
    A_node.set_dist(A_distribution)

    B_distribution = DiscreteDistribution(B_node)
    index = B_distribution.generate_index([], [])
    B_distribution[index] = [0.15, 0.45, 0.30, 0.10]
    B_node.set_dist(B_distribution)

    C_distribution = DiscreteDistribution(C_node)
    index = C_distribution.generate_index([], [])
    C_distribution[index] = [0.15, 0.45, 0.30, 0.10]
    C_node.set_dist(C_distribution)

    dist = zeros([A_node.size(), B_node.size(), AvB_node.size()], dtype=float32)
    dist[0, 0, :] = [0.10, 0.10, 0.80]
    dist[0, 1, :] = [0.20, 0.60, 0.20]
    dist[0, 2, :] = [0.15, 0.75, 0.10]
    dist[0, 3, :] = [0.05, 0.90, 0.05]

    dist[1, 0, :] = [0.60, 0.20, 0.20]
    dist[1, 1, :] = [0.10, 0.10, 0.80]
    dist[1, 2, :] = [0.20, 0.60, 0.20]
    dist[1, 3, :] = [0.15, 0.75, 0.10]

    dist[2, 0, :] = [0.75, 0.15, 0.10]
    dist[2, 1, :] = [0.60, 0.20, 0.20]
    dist[2, 2, :] = [0.10, 0.10, 0.80]
    dist[2, 3, :] = [0.20, 0.60, 0.20]

    dist[3, 0, :] = [0.90, 0.05, 0.05]
    dist[3, 1, :] = [0.75, 0.15, 0.10]
    dist[3, 2, :] = [0.60, 0.20, 0.20]
    dist[3, 3, :] = [0.10, 0.10, 0.80]
    AvB_distribution = ConditionalDiscreteDistribution(nodes=[A_node, B_node, AvB_node], table=dist)
    AvB_node.set_dist(AvB_distribution)

    dist = zeros([B_node.size(), C_node.size(), BvC_node.size()], dtype=float32)
    dist[0, 0, :] = [0.10, 0.10, 0.80]
    dist[0, 1, :] = [0.20, 0.60, 0.20]
    dist[0, 2, :] = [0.15, 0.75, 0.10]
    dist[0, 3, :] = [0.05, 0.90, 0.05]

    dist[1, 0, :] = [0.60, 0.20, 0.20]
    dist[1, 1, :] = [0.10, 0.10, 0.80]
    dist[1, 2, :] = [0.20, 0.60, 0.20]
    dist[1, 3, :] = [0.15, 0.75, 0.10]

    dist[2, 0, :] = [0.75, 0.15, 0.10]
    dist[2, 1, :] = [0.60, 0.20, 0.20]
    dist[2, 2, :] = [0.10, 0.10, 0.80]
    dist[2, 3, :] = [0.20, 0.60, 0.20]

    dist[3, 0, :] = [0.90, 0.05, 0.05]
    dist[3, 1, :] = [0.75, 0.15, 0.10]
    dist[3, 2, :] = [0.60, 0.20, 0.20]
    dist[3, 3, :] = [0.10, 0.10, 0.80]
    BvC_distribution = ConditionalDiscreteDistribution(nodes=[B_node, C_node, BvC_node], table=dist)
    BvC_node.set_dist(BvC_distribution)

    dist = zeros([C_node.size(), A_node.size(), CvA_node.size()], dtype=float32)
    dist[0, 0, :] = [0.10, 0.10, 0.80]
    dist[0, 1, :] = [0.20, 0.60, 0.20]
    dist[0, 2, :] = [0.15, 0.75, 0.10]
    dist[0, 3, :] = [0.05, 0.90, 0.05]

    dist[1, 0, :] = [0.60, 0.20, 0.20]
    dist[1, 1, :] = [0.10, 0.10, 0.80]
    dist[1, 2, :] = [0.20, 0.60, 0.20]
    dist[1, 3, :] = [0.15, 0.75, 0.10]

    dist[2, 0, :] = [0.75, 0.15, 0.10]
    dist[2, 1, :] = [0.60, 0.20, 0.20]
    dist[2, 2, :] = [0.10, 0.10, 0.80]
    dist[2, 3, :] = [0.20, 0.60, 0.20]

    dist[3, 0, :] = [0.90, 0.05, 0.05]
    dist[3, 1, :] = [0.75, 0.15, 0.10]
    dist[3, 2, :] = [0.60, 0.20, 0.20]
    dist[3, 3, :] = [0.10, 0.10, 0.80]
    CvA_distribution = ConditionalDiscreteDistribution(nodes=[C_node, A_node, CvA_node], table=dist)
    CvA_node.set_dist(CvA_distribution)

    nodes = [A_node, B_node, C_node, AvB_node, BvC_node, CvA_node]

    return BayesNet(nodes)

    # TODO: fill this out
    raise NotImplementedError


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""

    posterior = [0, 0, 0]
    AvB_node = bayes_net.get_node_by_name("AvB")
    CvA_node = bayes_net.get_node_by_name("CvA")
    BvC_node = bayes_net.get_node_by_name("BvC")
    engine = EnumerationEngine(bayes_net)
    engine.evidence[AvB_node] = 0
    engine.evidence[CvA_node] = 2
    Q = engine.marginal(BvC_node)[0]
    index = Q.generate_index([0], range(Q.nDims))
    posterior[0] = Q[index]
    index = Q.generate_index([1], range(Q.nDims))
    posterior[1] = Q[index]
    index = Q.generate_index([2], range(Q.nDims))
    posterior[2] = Q[index]

    return posterior

    # TODO: finish this function    
    raise NotImplementedError


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """

    if len(initial_state) == 0:
        for i in range(0, 6):
            if i < 3:
                initial_state.append(random.randint(0, 3))
            else:
                initial_state.append(random.randint(0, 2))
        initial_state[3] = 0
        initial_state[5] = 2

    A_node = bayes_net.get_node_by_name("A")
    B_node = bayes_net.get_node_by_name("B")
    C_node = bayes_net.get_node_by_name("C")
    AvB_node = bayes_net.get_node_by_name("AvB")
    BvC_node = bayes_net.get_node_by_name("BvC")
    CvA_node = bayes_net.get_node_by_name("CvA")

    A_table = A_node.dist.table
    B_table = B_node.dist.table
    C_table = C_node.dist.table
    AvB_table = AvB_node.dist.table
    BvC_table = BvC_node.dist.table
    CvA_table = CvA_node.dist.table

    sample = initial_state
    sample[3] = 0
    sample[5] = 2

    change = random.randint(0, 3)
    if change == 0:
        # "A"
        item0 = A_table[0] * B_table[initial_state[1]] * C_table[initial_state[2]] * \
                AvB_table[0, initial_state[1], 0] * BvC_table[initial_state[1], initial_state[2], initial_state[4]] * \
                CvA_table[initial_state[2], 0, 2]

        item1 = A_table[1] * B_table[initial_state[1]] * C_table[initial_state[2]] * \
                AvB_table[1, initial_state[1], 0] * BvC_table[initial_state[1], initial_state[2], initial_state[4]] * \
                CvA_table[initial_state[2], 1, 2]

        item2 = A_table[2] * B_table[initial_state[1]] * C_table[initial_state[2]] * \
                AvB_table[2, initial_state[1], 0] * BvC_table[initial_state[1], initial_state[2], initial_state[4]] * \
                CvA_table[initial_state[2], 2, 2]

        item3 = A_table[3] * B_table[initial_state[1]] * C_table[initial_state[2]] * \
                AvB_table[3, initial_state[1], 0] * BvC_table[initial_state[1], initial_state[2], initial_state[4]] * \
                CvA_table[initial_state[2], 3, 2]

        item_sum = item0 + item1 + item2 + item3

        prob0 = item0 / item_sum
        prob1 = item1 / item_sum
        prob2 = item2 / item_sum
        prob3 = item3 / item_sum

        rand = random.uniform(0, 1)
        if rand < prob0:
            sample[0] = 0
        elif rand < prob0 + prob1:
            sample[0] = 1
        elif rand < prob0 + prob1 + prob2:
            sample[0] = 2
        elif rand <= prob0 + prob1 + prob2 + prob3:
            sample[0] = 3

    elif change == 1:
        # "B"
        item0 = A_table[initial_state[0]] * B_table[0] * C_table[initial_state[2]] * \
                AvB_table[initial_state[0], 0, 0] * BvC_table[0, initial_state[2], initial_state[4]] * \
                CvA_table[initial_state[2], initial_state[0], 2]

        item1 = A_table[initial_state[0]] * B_table[1] * C_table[initial_state[2]] * \
                AvB_table[initial_state[0], 1, 0] * BvC_table[1, initial_state[2], initial_state[4]] * \
                CvA_table[initial_state[2], initial_state[0], 2]

        item2 = A_table[initial_state[0]] * B_table[2] * C_table[initial_state[2]] * \
                AvB_table[initial_state[0], 2, 0] * BvC_table[2, initial_state[2], initial_state[4]] * \
                CvA_table[initial_state[2], initial_state[0], 2]

        item3 = A_table[initial_state[0]] * B_table[3] * C_table[initial_state[2]] * \
                AvB_table[initial_state[0], 3, 0] * BvC_table[3, initial_state[2], initial_state[4]] * \
                CvA_table[initial_state[2], initial_state[0], 2]

        item_sum = item0 + item1 + item2 + item3

        prob0 = item0 / item_sum
        prob1 = item1 / item_sum
        prob2 = item2 / item_sum
        prob3 = item3 / item_sum

        rand = random.uniform(0, 1)
        if rand < prob0:
            sample[1] = 0
        elif rand < prob0 + prob1:
            sample[1] = 1
        elif rand < prob0 + prob1 + prob2:
            sample[1] = 2
        elif rand <= prob0 + prob1 + prob2 + prob3:
            sample[1] = 3

    elif change == 2:
        # "C"
        item0 = A_table[initial_state[0]] * B_table[initial_state[1]] * C_table[0] * \
                AvB_table[initial_state[0], initial_state[1], 0] * BvC_table[initial_state[1], 0, initial_state[4]] * \
                CvA_table[0, initial_state[0], 2]

        item1 = A_table[initial_state[0]] * B_table[initial_state[1]] * C_table[1] * \
                AvB_table[initial_state[0], initial_state[1], 0] * BvC_table[initial_state[1], 1, initial_state[4]] * \
                CvA_table[1, initial_state[0], 2]

        item2 = A_table[initial_state[0]] * B_table[initial_state[1]] * C_table[2] * \
                AvB_table[initial_state[0], initial_state[1], 0] * BvC_table[initial_state[1], 2, initial_state[4]] * \
                CvA_table[2, initial_state[0], 2]

        item3 = A_table[initial_state[0]] * B_table[initial_state[1]] * C_table[3] * \
                AvB_table[initial_state[0], initial_state[1], 0] * BvC_table[initial_state[1], 3, initial_state[4]] * \
                CvA_table[3, initial_state[0], 2]

        item_sum = item0 + item1 + item2 + item3

        prob0 = item0 / item_sum
        prob1 = item1 / item_sum
        prob2 = item2 / item_sum
        prob3 = item3 / item_sum

        rand = random.uniform(0, 1)
        if rand < prob0:
            sample[2] = 0
        elif rand < prob0 + prob1:
            sample[2] = 1
        elif rand < prob0 + prob1 + prob2:
            sample[2] = 2
        elif rand <= prob0 + prob1 + prob2 + prob3:
            sample[2] = 3

    elif change == 3:
        # "BvC"
        item0 = A_table[initial_state[0]] * B_table[initial_state[1]] * C_table[initial_state[2]] * \
                AvB_table[initial_state[0], initial_state[1], 0] * BvC_table[initial_state[1], initial_state[2], 0] * \
                CvA_table[initial_state[2], initial_state[0], 2]

        item1 = A_table[initial_state[0]] * B_table[initial_state[1]] * C_table[initial_state[2]] * \
                AvB_table[initial_state[0], initial_state[1], 0] * BvC_table[initial_state[1], initial_state[2], 1] * \
                CvA_table[initial_state[2], initial_state[0], 2]

        item2 = A_table[initial_state[0]] * B_table[initial_state[1]] * C_table[initial_state[2]] * \
                AvB_table[initial_state[0], initial_state[1], 0] * BvC_table[initial_state[1], initial_state[2], 2] * \
                CvA_table[initial_state[2], initial_state[0], 2]

        item_sum = item0 + item1 + item2

        prob0 = item0 / item_sum
        prob1 = item1 / item_sum
        prob2 = item2 / item_sum

        rand = random.uniform(0, 1)
        if rand < prob0:
            sample[4] = 0
        elif rand < prob0 + prob1:
            sample[4] = 1
        elif rand < prob0 + prob1 + prob2:
            sample[4] = 2

    sample = tuple(sample)
    return sample

    # TODO: finish this function
    raise NotImplementedError


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """

    if initial_state == None or initial_state == []:
        initial_state = [0, 0, 0, 0, 0, 2]

    A = bayes_net.get_node_by_name("A")
    A_table = A.dist.table
    B = bayes_net.get_node_by_name("B")
    B_table = B.dist.table
    C = bayes_net.get_node_by_name("C")
    C_table = C.dist.table
    AvB = bayes_net.get_node_by_name("AvB")
    AvB_table = AvB.dist.table
    BvC = bayes_net.get_node_by_name("BvC")
    BvC_table = BvC.dist.table
    CvA = bayes_net.get_node_by_name("CvA")
    CvA_table = CvA.dist.table

    new_state = initial_state[:]
    new_state[0] = random.choice([0, 1, 2, 3])
    new_state[1] = random.choice([0, 1, 2, 3])
    new_state[2] = random.choice([0, 1, 2, 3])
    new_state[4] = random.choice([0, 1, 2])

    prob_initial = A_table[initial_state[0]] * B_table[initial_state[1]] * C_table[initial_state[2]] * AvB_table[
        initial_state[0], initial_state[1], 0] * BvC_table[initial_state[1], initial_state[2], initial_state[4]] * \
                   CvA_table[initial_state[2], initial_state[0], 2]
    prob_new = A_table[new_state[0]] * B_table[new_state[1]] * C_table[new_state[2]] * AvB_table[
        new_state[0], new_state[1], 0] * BvC_table[new_state[1], new_state[2], new_state[4]] * CvA_table[
                   new_state[2], new_state[0], 2]

    if prob_new >= prob_initial:
        initial_state = new_state
    else:
        prob_ratio = random.uniform(0, 1)
        if prob_ratio < (prob_new / prob_initial):
            initial_state = new_state

    sample = tuple(initial_state)
    # TODO: finish this function
    # raise NotImplementedError
    return sample

    # if len(initial_state) == 0:
    #     for i in range(0, 6):
    #         if i < 3:
    #             initial_state.append(random.randint(0, 3))
    #         else:
    #             initial_state.append(random.randint(0, 2))
    #     initial_state[3] = 0
    #     initial_state[5] = 2
    #
    # A_node = bayes_net.get_node_by_name("A")
    # B_node = bayes_net.get_node_by_name("B")
    # C_node = bayes_net.get_node_by_name("C")
    # AvB_node = bayes_net.get_node_by_name("AvB")
    # BvC_node = bayes_net.get_node_by_name("BvC")
    # CvA_node = bayes_net.get_node_by_name("CvA")
    #
    # A_table = A_node.dist.table
    # B_table = B_node.dist.table
    # C_table = C_node.dist.table
    # AvB_table = AvB_node.dist.table
    # BvC_table = BvC_node.dist.table
    # CvA_table = CvA_node.dist.table
    #
    # sample = []
    # for i in range(0, 6):
    #     if i < 3:
    #         sample.append(random.randint(0, 3))
    #     else:
    #         sample.append(random.randint(0, 2))
    # sample[3] = 0
    # sample[5] = 2
    #
    # init_prob = A_table[initial_state[0]] * B_table[initial_state[1]] * C_table[initial_state[2]] * \
    #             AvB_table[initial_state[0], initial_state[1], 0] * \
    #             BvC_table[initial_state[1], initial_state[2], initial_state[4]] * \
    #             CvA_table[initial_state[2], initial_state[0], 2]
    #
    # samp_prob = A_table[sample[0]] * B_table[sample[1]] * C_table[sample[2]] * \
    #             AvB_table[sample[0], sample[1], 0] * BvC_table[sample[1], sample[2], sample[4]] * \
    #             CvA_table[sample[2], sample[0], 2]
    #
    # if samp_prob >= init_prob:
    #     sample = tuple(sample)
    # else:
    #     alpha = min(1, samp_prob/init_prob)
    #     u = random.uniform(0, 1)
    #     if u < alpha:
    #         sample = tuple(sample)
    #     else:
    #         sample = tuple(initial_state)
    #
    # return sample
    #
    # # TODO: finish this function
    # raise NotImplementedError


def compare_sampling(bayes_net, initial_state, delta):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0.0, 0.0, 0.0]
    MH_convergence = [0.0, 0.0, 0.0]

    Gibbs_flag = 1
    MH_flag = 1
    Gibbs_samples = [0.0, 0.0, 0.0]
    MH_samples = [0.0, 0.0, 0.0]
    Gibbs_ref = [0.0, 0.0, 0.0]
    MH_ref = [0.0, 0.0, 0.0]
    N = 10000
    N_count = 0
    initial_state1 = list(initial_state)
    initial_state2 = list(initial_state)

    while Gibbs_flag:
        sample = Gibbs_sampler(bayes_net, initial_state1)
        Gibbs_count += 1
        Gibbs_samples[sample[4]] += 1
        s = sum(Gibbs_samples)
        Gibbs_convergence = [Gibbs_samples[0]/s, Gibbs_samples[1]/s, Gibbs_samples[2]/s]

        if (Gibbs_convergence[0] - Gibbs_ref[0]) < delta and (Gibbs_convergence[1] - Gibbs_ref[1]) < delta and \
            (Gibbs_convergence[2] - Gibbs_ref[2]) < delta:
            N_count += 1
            if N == N_count:
                Gibbs_flag = 0
        else:
            Gibbs_ref = Gibbs_convergence
            N_count = 0
        initial_state1 = list(sample)

    N_count = 0

    while MH_flag:
        sample = list(MH_sampler(bayes_net, initial_state2))
        if sample == initial_state2:
            MH_rejection_count += 1
        else:
            MH_count += 1
        MH_samples[sample[4]] += 1
        s = sum(MH_samples)
        MH_convergence = [MH_samples[0]/s, MH_samples[1]/s, MH_samples[2]/s]

        if (MH_convergence[0] - MH_ref[0]) < delta and (MH_convergence[1] - MH_ref[1]) < delta and \
            (MH_convergence[2] - MH_ref[2]) < delta:
            N_count += 1
            if N == N_count:
                MH_flag = 0
        else:
            MH_ref = MH_convergence
            N_count = 0
        initial_state2 = list(sample)

    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count

    # TODO: finish this function
    raise NotImplementedError


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    choice = 1
    options = ['Gibbs', 'Metropolis-Hastings']
    factor = 5.2
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    return "Dhaneshwaran Jotheeswaran"
    # TODO: finish this function
    raise NotImplementedError
