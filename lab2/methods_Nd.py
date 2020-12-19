from typing import Union, Callable, Tuple, Dict

from plot_utils import graph_full

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

_AVAILABLE_METHODS = [
    'steepest-descent', 
    'coordinate-descent', 
    'projected-descent',
    'ravine-descent'
]

_NUM_POINTS = 50
_MAX_STEP = 0.1
_MIN_PDIST = 1e-4
_EPS = 1e-5
_DEFAULT_LR = 2e-3
_MAX_ITERS = 50000


def minimize_Nd(
    function: Union[Dict, Tuple],
    method: str, 
    visualize: bool = True
) -> np.ndarray:
    """ First-order minimization techniques """
    
    _check_method_name(method) 
    
    if isinstance(function, tuple):
        path = _NAME_TO_FN[method](function[0], function[1]) 
        if visualize:
            graph_full(function[0], path[-1], function[0]['bounds'], path=path)

    else:
        path = _NAME_TO_FN[method](function)
        if visualize:
            graph_full(function, path[-1], function['bounds'], path=path)
        
    return path


def plot_log_e_to_num_iters(
    function: Dict, method: str
):
    """ Plots log e to num iters graph """
    
    epsilons = [10 ** -eps for eps in range(1, 10)]
    calcs = list()
        
    _check_method_name(method)

    for e in epsilons:
        if isinstance(function, tuple):
            path = _NAME_TO_FN[method](function[0], function[1], eps=e)
            title = function[0]['title']
            
        else:
            path = _NAME_TO_FN[method](function, eps=e)
            title = function['title']
            
        calcs.append(len(path))
            
    plt.title(title)
    plt.plot(np.log2(epsilons), calcs)
    plt.ylabel("Number of iters")
    plt.xlabel("log2(eps)")
    plt.show()
    
    
def _golden_ring(f, a, b, e=_EPS):
    k = (3 - math.sqrt(5)) / 2

    fmin = 0

    x1 = a + (b - a) * k
    x2 = b - (b - a) * k

    fx1 = f(x1)
    fx2 = f(x2)

    while b - a > e:

        if fx1 < fx2:
            b = x2
            fmin = fx1

            x2 = x1
            x1 = a + (b - a) * k

            fx2 = fx1
            fx1 = f(x1)

        else:
            a = x1
            fmin = fx2

            x1 = x2
            x2 = b - (b - a) * k

            fx1 = fx2
            fx2 = f(x2)

    xmin = a if f(a) == fmin else b
    return xmin
    
    
def _steepest_descent(function_data: Dict, eps=_MIN_PDIST) -> np.ndarray:   
    f = function_data['func']
    df = function_data['deriv']
    P = function_data['initPoint']

    path = [P]

    for i in range(_MAX_ITERS):
        grad = df(P)

        g = lambda alpha: f(P - alpha * grad)
        alpha = _golden_ring(g, 0.0, _MAX_STEP)
        P = P - alpha * grad

        path.append(P)
        if len(path) >= 2 and np.linalg.norm(path[-2] - path[-1]) < eps:
            break

    path = np.array(path)
    return path
    
    
def _coordinate_descent(function_data: Dict, eps=_MIN_PDIST) -> np.ndarray:
    f = function_data['func']
    df = function_data['deriv']
    P = function_data['initPoint']
    dims = function_data['dims']

    path = [P]

    for i in range(_MAX_ITERS):        
        grad = np.zeros(dims)
        grad[i % dims] = 1
        
        # next two lineas calculate lr for steepest descent.. do we need it here?
        # could be just:
        # P = P - _DEFAULT_LR * grad

        g = lambda alpha: f(P - alpha * grad)
        alpha = _golden_ring(g, 0, _MAX_STEP)

        P = P - alpha * grad

        path.append(P)
        if len(path) >= 2 and np.linalg.norm(path[-2] - path[-1]) < eps:
            break
            
    path = np.array(path)
    return path
    
    
def _projected_descent(function_data: Dict, project_fn: Callable, eps=_MIN_PDIST) -> np.ndarray:
    f = function_data['func']
    df = function_data['deriv']
    P = function_data['initPoint']

    path = [P]

    for i in range(_MAX_ITERS):
        grad = df(P)

        g = lambda alpha: f(P - alpha * grad)
        alpha = _golden_ring(g, 0.0, _MAX_STEP)
        P = P - alpha * grad
        
        P = project_fn(P)

        path.append(P)
        if len(path) >= 2 and np.linalg.norm(path[-2] - path[-1]) < eps:
            break

    path = np.array(path)
    return path
    
    
def _ravine_descent(function_data: Dict, eps=_EPS, h=1e-4, C=5) -> np.ndarray:
    f = function_data['func']
    df = function_data['deriv']
    v_k, v_k1 = function_data['initPointsForRavine']
    
    path = list()
    
    def gradient_step(v_k):
        x_k = v_k - _DEFAULT_LR * df(v_k)
        return x_k
    
    x_k = gradient_step(v_k)
    x_k1 = gradient_step(v_k1)
    path.append(x_k)
    
    for i in range(_MAX_ITERS):
        path.append(x_k1)

        v_k2 = x_k1 - (x_k1 - x_k) / np.linalg.norm(x_k1 - x_k) * h * np.sign(f(x_k1) - f(x_k))
        x_k2 = gradient_step(v_k2)
        
        
        cos_a_k2 = (np.dot(v_k2 - x_k1, x_k2 - x_k1)) / np.linalg.norm(v_k2 - x_k1) / np.linalg.norm(x_k2 - x_k1)
        cos_a_k1 = (np.dot(v_k1 - x_k, x_k1 - x_k)) / (np.linalg.norm(v_k1 - x_k) * (np.linalg.norm(x_k1 - x_k)))
        h = h * C ** (cos_a_k2 - cos_a_k1)
        
        v_k = v_k1
        v_k1 = v_k2
        
        x_k = x_k1
        x_k1 = x_k2
           
            
        if np.linalg.norm(x_k1 - x_k) < eps or abs(f(x_k1) - f(x_k)) < eps:
            path.append(x_k1)
            break

    path = np.array(path)
    return path
    
    
_NAME_TO_FN = {
    'steepest-descent':_steepest_descent, 
    'coordinate-descent':_coordinate_descent, 
    'projected-descent':_projected_descent, 
    'ravine-descent':_ravine_descent, 
}


def _check_method_name(method_name: str):
    if method_name not in _AVAILABLE_METHODS:
        raise ValueError(
            f"{method_name} not supported. \n" +
            f"Choose one of {_AVAILABLE_METHODS}")
