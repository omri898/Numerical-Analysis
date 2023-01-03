"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random

def bezier3(P1, P2, P3, P4):
    M = np.array(
        [[-1, +3, -3, +1],
         [+3, -6, +3, 0],
         [-3, +3, 0, 0],
         [+1, 0, 0, 0]],
        dtype=np.float32
    )
    P = np.array([P1, P2, P3, P4], dtype=np.float32)

    def f(t):
        T = np.array([t ** 3, t ** 2, t, 1], dtype=np.float32)
        return T.dot(M).dot(P)

    return f

def Thomas(a, b, c, d):
    nf = len(d)
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """
        
        if n==1:
            f1 = lambda x: f((a+b)/2)
            return f1
        
        xs = np.linspace(a,b,n)
        ys = [f(x) for x in xs]
        points = np.array(list(zip(xs,ys)))
        nsplines = n-1
        B3 = {}
        x_values = {}

        W = 4 * np.identity(nsplines)
        np.fill_diagonal(W[1:], 1)
        np.fill_diagonal(W[:, 1:], 1)
        W[0, 0] = 2
        W[nsplines - 1, nsplines - 1] = 7
        W[nsplines - 1, nsplines - 2] = 2
        
        K = [2 * (2 * points[i] + points[i + 1]) for i in range(nsplines)]
        K[0] = points[0] + 2 * points[1]
        K[nsplines - 1] = 8 * points[nsplines - 1] + points[nsplines]

        a = np.diag(W, k=-1)
        b = np.diag(W, k=0)
        c = np.diag(W, k=1)
        dx = [x[0] for x in K]
        dy = [y[1] for y in K]
        X = Thomas(a,b,c,dx)
        Y = Thomas(a,b,c,dy)
        A = np.array(list(zip(X,Y)))
        B = [0] * nsplines
        for i in range(nsplines - 1):
            B[i] = 2 * points[i + 1] - A[i + 1]
        B[nsplines - 1] = (A[nsplines - 1] + points[nsplines]) / 2

        for i in range(len(points)-1):
            Ci = (points[i], A[i], B[i], points[i+1])
            B3[(Ci[0][0],Ci[3][0])] = bezier3(Ci[0],Ci[1],Ci[2],Ci[3])
            x_values[(Ci[0][0],Ci[3][0])] = (Ci[0][0],Ci[1][0],Ci[2][0],Ci[3][0])

        def interpolated_f(x):
            for i in B3:
                if i[0] < x < i[1]:
                    X = x_values[i]
                    t = (x-X[0])/(X[3]-X[0])
                    y = B3[i](t)[1]
                    return y

        return interpolated_f


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm
from commons import *

class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()
        
        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            # f = np.poly1d(a)
            f = lambda x: np.sin(x**2)
            # f = lambda x: np.sin(x**2)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(f'Time = {T}')
        print(f'Mean err = {mean_err}')

    # def test_with_poly_restrict(self):
    #     ass1 = Assignment1()
    #     a = np.random.randn(5)
    #     f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
    #     ff = ass1.interpolate(f, -10, 10, 10)
    #     xs = np.random.random(20)
    #     for x in xs:
    #         yy = ff(x)

if __name__ == "__main__":
    unittest.main()