"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import random

def inverse(A):
    n = len(A)
    AM = np.matrix.copy(A)
    I = np.identity(n)
    IM = np.matrix.copy(I)
    indices = list(range(n))
    for fd in range(n):
        fdScaler = 1.0 / AM[fd][fd]
        for j in range(n):
            AM[fd][j] *= fdScaler
            IM[fd][j] *= fdScaler
        for i in indices[0:fd] + indices[fd+1:]: 
            crScaler = AM[i][fd] 
            for j in range(n): 
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
                IM[i][j] = IM[i][j] - crScaler * IM[fd][j]
    return IM

class Assignment4A:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def fit(self, f: callable, a: float, b: float, d:int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        T = time.time()
        n = 50000
        if d >= 10:
            threshold = 2
        else:
            threshold = 1.5

        degree = d
        powers = np.empty(d+1)
        for i in range (d+1):
            powers[i] = degree
            degree -= 1

        while time.time() - T < maxtime - threshold:
            xs = np.linspace(a,b,n)
            ys = np.array([f(x) for x in xs])
            A = np.empty((n,d+1))
            for i in range (n):
                xi = np.full(d+1,xs[i])
                row = np.power(xi,powers)
                A[i] = row
            n += 10000
            # print(n)
        X = inverse(A.T.dot(A)).dot(A.T).dot(ys)
        # print(f'X = {X}')

        def f(x):
            y = np.poly1d(X)
            return y(x)
        return f

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):
    # def test_return(self):
    #     # print('Here 1')
    #     f = NOISY(0.01)(poly(2,1,1,1,2,8,1,9,4))
    #     ass4 = Assignment4A()
    #     T = time.time()
    #     shape = ass4.fit(f=f, a=0, b=10, d=8, maxtime=5)
    #     T = time.time() - T
    #     self.assertLessEqual(T, 5)

    # def test_delay(self):
    #     # print('Here 2')
    #     f = DELAYED(7)(NOISY(0.01)(poly(2,1,1,1,2,8,1,9,4)))
    #     ass4 = Assignment4A()
    #     T = time.time()
    #     shape = ass4.fit(f=f, a=0, b=10, d=8, maxtime=5)
    #     T = time.time() - T
    #     self.assertGreaterEqual(T, 5)

    def test_err(self):
        times = 5
        ass4 = Assignment4A()
        total_mse = 0
        # total_time = 0
        for i in range (times):
            # f = poly(2,1,1,1,2,8,1,9,4)
            # nf = DELAYED(7)(NOISY(0.01)(poly(1,2,3,4)))
            f = poly(1,2,3,4,5,6,7,8,1,2,3,7)
            nf = NOISY(1)(f)
            T = time.time()
            ff = ass4.fit(f=nf, a=0, b=1, d=11, maxtime=5)
            T = time.time() - T
            if T>5:
                raise AssertionError(f'T = {T} - Too much time')
            # total_time += T
            mse=0
            for x in np.linspace(-2,2,1000):            
                self.assertNotEquals(f(x), nf(x))
                mse+= (f(x)-ff(x))**2
            mse = mse/1000
            print(f'MSE = {mse}')
            total_mse += mse
        total_mse /= times
        print(f'Avarage MSE = {total_mse}')


if __name__ == "__main__":
    unittest.main()
