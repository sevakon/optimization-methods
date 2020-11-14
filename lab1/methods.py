from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

_AVAILABLE_METHODS = [
    'dichotomy', 
    'golden-section', 
    'fibonacci', 
    'parabolic', 
    'brent',
]

_NUM_POINTS = 50
_MIN_E = 0.00001
_MAX_E = 0.5


def minimize_1d(
    function: Callable, method: str, 
    a: float, b: float, e: float,
    title: str, visualize: bool = True
):
    """ Perform 1d minimization based on the specified method """

    _check_method_name(method)
        
    history, x_min, y_min, num_calcs = _NAME_TO_FN[method](
        function, a, b, e)

    x = np.linspace(a, b, _NUM_POINTS)
    y = list(map(function, x))
        
    if visualize:
        plt.title(title)
        plt.plot(x, y)
        plt.scatter([x_min], [y_min])
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()        
        
    return pd.DataFrame(history), x_min, y_min


def plot_log_e_to_num_iters(
    function: Callable, method: str, 
    a: float, b: float, title: str, 
):
    """ Plots log e to num iters graph """
    epsilons = np.linspace(_MIN_E, _MAX_E, 25)
    calcs = list()
    
    _check_method_name(method)
    
    for e in epsilons:
        _, _, _, num_calcs = _NAME_TO_FN[method](
            function, a, b, e)
        calcs.append(num_calcs)
            
    plt.title(title)
    plt.plot(np.log2(epsilons), calcs)
    plt.ylabel("Number of iters")
    plt.xlabel("log2(eps)")
    plt.show()
            
# --- Private functions --- #
        
        
def _dichotomy_min(f, a, b, e):
    history = list()
    
    delta = e / 4
    n = int(math.log((b - a) / e) / math.log(2))
    fmin = 0

    for _ in range(n):        
        x1 = (a + b) / 2 - delta
        x2 = (a + b) / 2 + delta

        fx1 = f(x1)
        fx2 = f(x2)
        if fx1 < fx2:
            b = x2
            fmin = fx1
        else:
            a = x1
            fmin = fx2
           
        history.append({
            "a":a, "b":b, "abs(a-b)":abs(a-b),
            "x_1":x1, "x_2":x2, "f(x_1)":fx1, "f(x_2)":fx2,
        })
        
    xmin = a if abs(f(a) - fmin) <= abs(f(b) - fmin) else b
    num_calcs = n * 2
        
    return history, xmin, fmin, num_calcs


def _golden_section_min(f, a, b, e):
    history = list()
    k = (3 - math.sqrt(5)) / 2
    num_calcs = 0
    
    fmin = 0

    x1 = a + (b - a) * k
    x2 = b - (b - a) * k

    fx1 = f(x1)
    fx2 = f(x2)
    num_calcs += 2
    
    history.append({
        "a":a, "b":b, "abs(a-b)":abs(a-b), 
        "x_1":x1, "x_2":x2, "f(x_1)":fx1, "f(x_2)":fx2
    })

    while b - a > e:

        if fx1 < fx2:
            b = x2
            fmin = fx1

            x2 = x1
            x1 = a + (b - a) * k
            
            fx2 = fx1
            fx1 = f(x1)
            
            num_calcs += 1

        else:
            a = x1
            fmin = fx2

            x1 = x2
            x2 = b - (b - a) * k

            fx1 = fx2
            fx2 = f(x2)
            num_calcs += 1
            
        history.append({
            "a":a, "b":b, "abs(a-b)":abs(a-b), 
            "x_1":x1, "x_2":x2, "f(x_1)":fx1, "f(x_2)":fx2
        })
            
    xmin = a if f(a) == fmin else b
    
    return history, xmin, fmin, num_calcs


def _fibonacci_min(f, a, b, e):
    history = list()
    
    def fibonacci(n):
        return (1 / math.sqrt(5)) * (((1 + math.sqrt(5)) / 2) ** n - ((1 - math.sqrt(5)) / 2) ** n)
    
    def find_dot(a, b, n, k, is_first):
        if is_first:
            return a + (fibonacci(n - k - 1) * (b - a)) / fibonacci(n - k + 1)
        else:
            return a + (fibonacci(n - k) * (b - a)) / fibonacci(n - k + 1)
        
    n = 0
    num_calcs = 0
    while (b - a) / e >= fibonacci(n + 2):
        n += 1
        
    x1 = find_dot(a, b, n, 1, True)
    x2 = find_dot(a, b, n, 1, False)
    
    history.append({
        "a":a, "b":b, "abs(a-b)":abs(a-b), 
        "x_1":x1, "x_2":x2, "f(x_1)":f(x1), "f(x_2)":f(x2)
    })
        
    for k in range(2, n + 1):
        
        if f(x1) > f(x2):
            a = x1
            x1 = x2
            x2 = find_dot(a, b, n, k, False)
        else:
            b = x2
            x2 = x1
            x1 = find_dot(a, b, n, k, True)
            
        history.append({
            "a":a, "b":b, "abs(a-b)":abs(a-b), 
            "x_1":x1, "x_2":x2, "f(x_1)":f(x1), "f(x_2)":f(x2)
        })
        
        num_calcs += 2
            
        if k == n - 2:
            break
            
    x_min = (x1 + x2) / 2
    f_min = f(x_min)
    
    return history, x_min, f_min, num_calcs


def _parabolic_min(f, a, b, e):
    history = list()
    num_calcs = 3
    
    x1, x2, x3 = a, (a + b) / 2, b
    y1, y2, y3 = f(x1), f(x2), f(x3)
    
    history.append({
        "x1":x1, "x2":x2, "x3":x3, 
        "abs(x3-x2)":abs(x3-x1),
        "y1":y1, "y2":y2, "y3":y3,
    })
    
    def parabolic_minimum(x1, x2, x3, y1, y2, y3):
        return x2 - 0.5 * ((x2 - x1) ** 2 * (y2 - y3) - (x2 - x3) ** 2 * (y2 - y1)) / (
            (x2 - x1) * (y2 - y3) - (x2 - x3) * (y2 - y1))

    while x3 - x1 >= e:

        u = parabolic_minimum(x1, x2, x3, y1, y2, y3)
        yu = f(u)
        num_calcs += 1

        if u < x2:
            if yu < y2:
                x3, y3 = x2, y2
                x2, y2 = u, yu

            else:

                x1, y1 = u, yu
        else:

            if yu < y2:
                x1, y1 = x2, y2
                x2, y2 = u, yu

            else:
                x3, y3 = u, yu
                
        history.append({
            "x1":x1, "x2":x2, "x3":x3, 
            "abs(x3-x2)":abs(x3-x1),
            "y1":y1, "y2":y2, "y3":y3,
        })
        
    x_min = (x1 + x3) / 2
    f_min = f(x_min)
        
    return history, x_min, f_min, num_calcs 


def _brent_min(f, a, b, eps):    
    history = list()
    num_calcs = 1
    
    k = (3 - math.sqrt(5)) / 2
    x = (a + b) / 2
    w = (a + b) / 2
    v = (a + b) / 2
    f_x = f(x)
    f_w = f(x)
    f_v = f(x)
    
    d = b - a
    e = b - a
    
    u = None
    
    def parabolic_minimum(x1, x2, x3, y1, y2, y3):
        return x2 - 0.5 * ((x2 - x1) ** 2 * (y2 - y3) - (x2 - x3) ** 2 * (y2 - y1)) / (
            (x2 - x1) * (y2 - y3) - (x2 - x3) * (y2 - y1))
    
    def are_different(a, b, c):
        if a == b or b == c or a == c:
            return False
        return True
    
    history.append({
        "a": a, "c":b, "c-a":b-a, "x":x, "v":v, 
        "w":w, "f(x)":f_x, "f(v)":f_v, "f(w)":f_w,
    })
    
    # or d
    while b - a > eps:
        g = e
        e = d
        
        if are_different(x, w, v) and are_different(f_x, f_w, f_v):
            u = parabolic_minimum(w, x, v, f_w, f_x, f_v)
            
        if u is not None and u >= a + eps and u <= b - eps and abs(u - x) < g/2:
            d = abs(u - x)
            
        else:
            if x < (b + a) / 2:
                u = x + k * (b - x)
                d = b - x
            else:
                u = x - k * (x - a)
                d = x - a
                
            if abs(u - x) < eps:
                u = x + eps if (u - x) >= 0 else x - eps
            
            f_u = f(u)
            num_calcs += 1
            
            if f_u <= f_x:
                if u >= x:
                    a = x
                else:
                    b = x
                v = w
                w = x
                x = u
                f_v = f_w
                f_w = f_x
                f_x = f_u
                
            else:
                if u >= x:
                    b = x
                else:
                    a = x
                if f_u <= f_w or w == x:
                    v = w
                    w = u
                    f_v = f_w
                    f_w = f_u
                elif f_u <= f_v or v == x or v == w:
                    v = u
                    f_v = f_u
                    
        history.append({
            "a": a, "c":b, "c-a":b-a, "x":x, "v":v, 
            "w":w, "f(x)":f_x, "f(v)":f_v, "f(w)":f_w,
        })

    return history, x, f(x), num_calcs


_NAME_TO_FN = {
    'dichotomy':_dichotomy_min, 
    'golden-section':_golden_section_min, 
    'fibonacci':_fibonacci_min, 
    'parabolic':_parabolic_min, 
    'brent':_brent_min,
}


def _check_method_name(method_name: str):
    if method_name not in _AVAILABLE_METHODS:
        raise ValueError(
            f"{method_name} not supported. \n" +
            f"Choose one of {_AVAILABLE_METHODS}")
