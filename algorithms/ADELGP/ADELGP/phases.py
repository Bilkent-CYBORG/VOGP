import copy
from typing import List

import cvxpy as cp
import numpy as np

from algorithms.ADELGP.ADELGP.Hyperrectangle import Hyperrectangle
from algorithms.ADELGP.ADELGP.Polyhedron import Polyhedron
from algorithms.ADELGP.ADELGP.DesignPoint import DesignPoint, DesignPointAD
from algorithms.ADELGP.ADELGP.ad_utils import calculate_vh


def modeling(A: List[DesignPoint], gp, beta):
    pts = np.array([point.x for point in A])
    mus, covs = gp.predict(pts)
    for point, mu, cov in zip(A, mus, covs):
        mu = mu.reshape(-1)
        std = np.sqrt(np.diag(cov.squeeze()))

        # High probability lower and upper bound, B
        L = mu - std * beta
        U = mu + std * beta

        # Confidence hyperrectangle, Q
        Q = Hyperrectangle(L.tolist(), U.tolist())

        # Cumulative confidence hyperrectangle, R
        point.R = point.R.intersect(Q)
        point.mu = mu

def discard(S, P, D, C, epsilon, u_star):
    p_pess = pess(S, P, C, u_star)
    difference = set_diff(S, p_pess)  # undecided points that are not in pessimistic pareto set.

    if (C.A.shape[0] == C.A.shape[1]) and np.allclose(C.A, np.eye(C.A.shape[0])):
        dom_func = dominated_by_fastest
    else:
        dom_func = dominated_by_opt3

    to_be_discarded_pess = []
    to_be_discarded_norm = []

    for point in difference:
        for point_prime in p_pess:
            # Function to check if  ∃z' in R(x') such that R(x) <_C z + u, where u < epsilon
            if dom_func(point, point_prime, C, epsilon, u_star):
                to_be_discarded_pess.append(point)
                # S.remove(point)
                break
    
    for node in to_be_discarded_pess:
        if node in S:
            S.remove(node)
            D.append(node)

def epsiloncovering(S, P, C, epsilon, max_depth, u_star):
    # Don't do epsilon covering during adaptive discretization
    # unless every point in the undecided set are at maximum depth
    for point in S:
        if isinstance(point, DesignPointAD) and point.depth != max_depth:
            return
    
    if (C.A.shape[0] == C.A.shape[1]) and np.allclose(C.A, np.eye(C.A.shape[0])):
        ecover_func = ecovered_fastest
    else:
        ecover_func = ecovered_faster_new

    A = S + P

    is_index_pareto = []
    for point in S:
        for point_prime in A:
            if point == point_prime:
                continue

            if ecover_func(point, point_prime, C, epsilon, u_star):
                is_index_pareto.append(False)
                break
        else:
            is_index_pareto.append(True)

    tmp_S = copy.deepcopy(S)
    for is_pareto, point in zip(is_index_pareto, tmp_S):
        if is_pareto:
            S.remove(point)
            P.append(point)

def evaluate(W: List[DesignPoint], model, beta, batch_size) -> DesignPoint:
    if batch_size:
        raise NotImplementedError
    else:
        largest = 0
        to_observe = None
        for x in W:
            diameter = x.R.diameter
            if diameter > largest:
                largest = diameter
                to_observe = x

        print(f"Observing point {to_observe}. It has diameter {largest}")

        return [to_observe]

def pess_helper(point_i, point_set, C):
    for point_j in point_set:
        if point_j == point_i:
            continue

        if check_dominates(point_j.R, point_i.R, C):
            return False
    
    return True

def pess(
    point_set: List[DesignPoint], pareto_set: List[DesignPoint], C: Polyhedron, u_star: np.ndarray
) -> List[DesignPoint]:
    """
    The set of Pessimistic Pareto set of a set of DesignPoint objects.
    :param point_set: List of DesignPoint objects.
    :param C: The ordering cone.
    :return: List of Node objects.
    """
    if (C.A.shape[0] == C.A.shape[1]) and np.allclose(C.A, np.eye(C.A.shape[0])):
        dom_func = check_dominates_fastest
    else:
        dom_func = check_dominates_new

    W = point_set + pareto_set
    point_set = W

    pess_set = []
    for point_i in point_set:
        for point_j in W:
            if point_j == point_i:
                continue

            # Check if there is another point j that dominates i, if so,
            # do not include i in the pessimistic set
            if dom_func(point_j.R, point_i.R, C, u_star):
                break
        else:
            pess_set.append(point_i)

    # pess_set += pareto_set  # Pareto points are by definition pessimistic Pareto

    return pess_set

def check_dominates(polyhedron1: Hyperrectangle, polyhedron2: Hyperrectangle, cone: Polyhedron) -> bool:
    """
    Check if polyhedron1 dominates polyhedron2.
    Check if polyhedron1 is a subset of polyhedron2 + cone (by checking each vertex of polyhedron1).

    :param polyhedron1: The first polyhedron.
    :param polyhedron2: The second polyhedron.
    :param cone: The ordering cone.
    :return: Dominating condition.
    """ 

    condition = True
    n = cone.A.shape[1]  # Variable shape of x
    c = np.zeros(n)

    x = cp.Variable(n)
    y = cp.Variable(n)

    vertices = polyhedron1.get_vertices()

    for vertex in vertices:
        """
        Checking if vertices can be represented by summation of a help vector from cone (y in this case) and zx
        """
        x = cp.Variable(n)
        y = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(0),
                          [x + y == vertex, #@: Together with two lines below, this enforces vertextes of first
                                            # hyperrec to be sum of help from cone (y) plus point from polyhedron2.
                           polyhedron2.A @ x >= polyhedron2.b, #@: This enforces x to be in polyhedron2
                           cone.A @ y >= cone.b]) #@: This enforces y to be in cone
        try:                   
            prob.solve(solver = "ECOS")
        except cp.error.SolverError:
            prob.solve(solver = "SCIPY")

        if prob.status == 'infeasible':
            condition = False
            break

    return condition

def check_dominates_fastest(
    polyhedron1: Hyperrectangle, polyhedron2: Hyperrectangle, cone: Polyhedron, u_star: np.ndarray
) -> bool:
    """
    Check if polyhedron1 dominates polyhedron2.
    Check if polyhedron1 is a subset of polyhedron2 + cone
    (by checking lower corner of polyhedron1 and upper corner of polyhedron2).

    If returns True, polyhedron2 is not added to pessimistic set.
    """

    vertex = polyhedron1.get_lower()
    vertex_prime = polyhedron2.get_lower()
    
    return ((vertex-vertex_prime) >= 0).all()

def set_diff(s1, s2):  # Discarding
    #@: implements s1-s2  where the shared subsets are removed from s1.
    """
    Set difference of two sets.

    :param s1: List of DesignPoint objects.
    :param s2: List of DesignPoint objects.
    :return: List of DesignPoint objects.
    """

    tmp = copy.deepcopy(s1)

    for node in s2:
        if node in tmp:
            tmp.remove(node)

    return tmp

def dominated_by_opt3(point, point_prime, C, epsilon, u_star): #Line 11 of the algorithm
    # implementing the discarding rule
    # Define and solve the CVXPY problem.

    n = C.A.shape[1]
    # u = epsilon * (np.ones(n) / np.sqrt(n))
    W = C.A

    # Check each vertex in R(x)
    vertices = point.R.get_vertices()
    vertices_prime = point_prime.R.get_vertices()

    vertices = vertices.astype(np.float64) @ W.transpose()
    vertices_prime = (vertices_prime.astype(np.float64) + u_star) @ W.transpose()

    for row in vertices:
        for row_prime in vertices_prime:
            if not (row <= row_prime).all():
                return False

    return True

def dominated_by_fastest(point: DesignPoint, point_prime: DesignPoint, C, epsilon: float, u_star: np.ndarray):
    # Line 11 of the algorithm
    # implementing the discarding rule
    # point will be eliminated if this returns true

    vertices = point.R.get_upper()
    vertices_prime = point_prime.R.get_lower()

    return (((vertices_prime + epsilon) - vertices) >= 0).all()

def ecovered(point, point_prime, C, epsilon): 
    """

    :param point: DesignPoint x.
    :param point_prime: Design Point x'.
    :param C: Polyhedron C.
    :param epsilon:
    :return:
    """
    n = C.A.shape[1]

    z = cp.Variable(n)
    z_point = cp.Variable(n)
    z_point2 = cp.Variable(n)
    c_point = cp.Variable(n)
    c_point2 = cp.Variable(n)
    u = epsilon * (np.ones(n) / np.sqrt(n))

    W_point = point.R.A
    W_point_prime = point_prime.R.A
    W_C = C.A

    b_point = point.R.b
    b_point_prime = point_prime.R.b
    b_C = C.b

    P = np.eye(n)
                                                        #@: Here, they use the intersection version 
    prob = cp.Problem(cp.Minimize(cp.sum(P)), #@: This minimizes the  norm of u
                      [z == z_point + u + c_point, 
                       z == z_point2 - c_point2,       #@: z is meant to be the intersection point
                       W_point @ z_point >= b_point,  #@: these two enforce the hyperrectangles
                       W_point_prime @ z_point2 >= b_point_prime,
                       W_C @ c_point >= b_C, #@: These two enforces c points to be from the cone
                       W_C @ c_point2 >= b_C])
                       #W_C @ u >= b_C])
    try:                   
        prob.solve(solver = "OSQP")
    except :#cp.error.SolverError:
        prob.solve(solver = "ECOS")

    condition = prob.status == 'optimal'  
    return condition

def ecovered_faster(point, point_prime, C, epsilon, u_star): 
    """
    :param point: DesignPoint x.
    :param point_prime: Design Point x'.
    :param C: Polyhedron C.
    :param epsilon:
    :return:
    """
    n = C.A.shape[1]

    z_point = cp.Variable(n)
    z_point2 = cp.Variable(n)
    # u = epsilon * (np.ones(n) / np.sqrt(n))

    W_point = point.R.A
    W_point_prime = point_prime.R.A
    W_C = C.A

    b_point = point.R.b
    b_point_prime = point_prime.R.b
    b_C = C.b
    P = np.eye(n)
    #@: Here, they use the intersection version 
    prob = cp.Problem(
        cp.Minimize(cp.sum(P)), #@: This minimizes the  norm of u
        [
            W_point @ z_point >= b_point,  #@: these two enforce the hyperrectangles
            W_point_prime @ z_point2 >= b_point_prime,
            W_C @ (z_point2-z_point-u_star)>= b_C
        ]
    )
    try:
        prob.solve(solver = "OSQP",max_iter=10000)#,verbose=True)
    except :#cp.error.SolverError:
        prob.solve(solver = "ECOS")

    if prob.status==None:
        return True

    condition = prob.status == 'optimal'  

    return condition

def ecovered_fastest(point: DesignPoint, point_prime: DesignPoint, C, epsilon, u_star):
    """
    :param point: DesignPoint x.
    :param point_prime: Design Point x'.
    :param C: Polyhedron C.
    :param epsilon:
    :return:

    If returns True, point is not added to the pareto set.
    """

    vertex = point.R.get_lower()
    vertex_prime = point_prime.R.get_upper()
    
    return ((vertex_prime - (vertex + epsilon)) >= 0).all()

#################

def evaluating_refining_AD(W, S, P, gp, beta, depth_max):
    largest = 0
    to_observe = None

    for x in W:
        diameter = x.R.diameter
        if diameter > largest:
            largest = diameter
            to_observe = x

    mu, cov = gp.predict(to_observe.x.reshape(-1, gp.input_dim))
    std = np.sqrt(np.diag(cov.squeeze()))
    
    x_in_S = to_observe in S
    vh = calculate_vh(to_observe.depth, gp, gp.output_dim, gp.input_dim, depth_offset=0)

    diff = beta * np.linalg.norm(std) - np.linalg.norm(vh)
    print(f'The difference in comparison at depth {to_observe.depth}: {diff.tolist()}')

    to_observe_list = []
    if np.all(beta * np.linalg.norm(std) <= np.linalg.norm(vh)) and to_observe.depth < depth_max:
        if x_in_S:
            S.remove(to_observe)
            children = to_observe.get_child_list()
            for child in children:
                S.append(child)
        else:
            P.remove(to_observe)
            children = to_observe.get_child_list()
            for child in children:
                P.append(child)
    else:
        to_observe_list.append(to_observe)
        print(f"Observing point {to_observe}. It has diameter {largest}")

    return [to_observe]

#################

def line_seg_pt_intersect_at_dim(P1, P2, target_pt, target_dim):
    t = (target_pt[target_dim] - P1[target_dim]) / (P2[target_dim] - P1[target_dim])

    if t < 0 or t > 1:
        # No intersection
        return None

    point_on_line = P1 + t * (P2 - P1)
    return point_on_line

def is_pt_in_extended_polytope(pt, polytope, invert_extension=False):
    dim = polytope.shape[1]
    
    if invert_extension:
        comp_func = lambda x, y: x >= y
    else:
        comp_func = lambda x, y: x <= y

    # Vertex is trivially an element
    for vert in polytope:
        if comp_func(vert, pt).all():
            return True

    # Check intersections with polytope. If any intersection is dominated, then an element.
    for dim_i in range(dim):
        edges_of_interest = np.empty((0, 2, dim), dtype=np.float64)
        for vert_i, vert1 in enumerate(polytope):
            for vert_j, vert2 in enumerate(polytope):
                if vert_i == vert_j:
                    continue

                if vert1[dim_i] <= pt[dim_i] and pt[dim_i] <= vert2[dim_i]:
                    edges_of_interest = np.vstack((
                        edges_of_interest,
                        np.expand_dims(np.vstack((vert1, vert2)), axis=0)
                    ))

        for edge in edges_of_interest:
            intersection = line_seg_pt_intersect_at_dim(edge[0], edge[1], pt, dim_i)
            if intersection is not None and comp_func(intersection, pt).all():
                # Vertex is an element due to the intersection point
                return True

    return False

def check_dominates_new(polyhedron1: Hyperrectangle, polyhedron2: Hyperrectangle, cone: Polyhedron, u_star: np.ndarray) -> bool:

    W = cone.A
    xprime = polyhedron1.get_vertices().astype(np.float64) @ W.transpose()
    x = polyhedron2.get_vertices().astype(np.float64) @ W.transpose()
    
    # For every vertex of x', check if element of x+C. Return False if any vertex is not.
    for ref_point in xprime:
        if is_pt_in_extended_polytope(ref_point, x) is False:
            return False
    
    return True

def ecovered_faster_new(point, point_prime, C, epsilon, u_star):
    W = C.A
    m = W.shape[1]

    polyhedron1 = point_prime.R
    polyhedron2 = point.R
    xprime = polyhedron1.get_vertices().astype(np.float64) @ W.transpose()
    x = (polyhedron2.get_vertices().astype(np.float64) + u_star) @ W.transpose()
    
    # For every vertex of x', check if element of x+C. Return False if any vertex is not.
    for ref_point in xprime:
        if is_pt_in_extended_polytope(ref_point, x) is True:
            return True
    
    for ref_point in x:
        if is_pt_in_extended_polytope(ref_point, xprime, invert_extension=True) is True:
            return True
    
    return False
