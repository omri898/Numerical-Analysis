"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable

derivative = lambda f,a,h=0.001: (f(a + h) - f(a))/h

# def derivative(f,a,h=0.001):
#     return (f(a + h) - f(a))/h

def bisection (f,a,b,e,N=10):
    while abs(b-a) > 2*e and N > 0:
        estimate = (a+b)/2
        if f(a)*f(estimate) < 0:
            b = estimate
        else:
          a = estimate
        N -= 1
    if f(a+b)/2 < 2*e:
        return (a+b)/2
    return None

def newton(f,x0,epsilon,a,b,max_iter=5):
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon: # Found solution after max_iter iterations.
            return xn 
        Dfxn = derivative(f,xn) # Zero derivative, No solution found. Try bisection.
        if Dfxn == 0: 
            return bisection(f,a,b)
        xn = xn - fxn/Dfxn # Exceeded maximum iterations, No solution found. Try bisection.
    return bisection(f,a,b,epsilon)

def add_check_duplicate (X,root,maxerr):
    '''
    Goes thorugh a list of roots to find if the given root is already inside.
    '''
    for i in X[::-1]:
        if abs(root - i) < maxerr*2:
            return None
    X.append(root)
    
class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        X = [] # List of roots
        f = lambda x: 1000*(f2(x) - f1(x)) # Need to find roots for this function. Multiply by a large number to get a bigger gradient.
        h = (b-a)/3000 # Divide into 3000 sections
        left = a
        right = a + h

        for i in range (3000):
            if i == 2999: # When reached the end
                right = b
                
            # CHECK IF LEFT OR RIGHT IS THE ROOT
            if abs(f(left)) <= maxerr:
                add_check_duplicate(X,left,maxerr)
                left = right
                right += h
            elif abs(f(right)) <= maxerr:
                add_check_duplicate(X,right,maxerr)
                left = right
                right += h

            # There's a root if the intermediate value theorem applies
            elif f(right)*f(left) < 0: 
                guess = (left+right)/2
                root = newton(f,guess,maxerr,left,right,5) # Trying to find the root using newton / bisection
                if root != None and left < root < right:
                    add_check_duplicate(X,root,maxerr)
                left = right
                right += h

            else:
                left = right
                right += h
        return X


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
from commons import *
import math

class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)
        f1 = lambda x: np.sin(x)
        f2 = lambda x: 1

        X = ass2.intersections(f1, f2, -10, 10, maxerr=0.001)
        print(f'Roots = {X}')
        print(len(X))

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
