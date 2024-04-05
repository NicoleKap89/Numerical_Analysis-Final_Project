"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


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

        def deriv_calc(func, value):
            """
            The function receives a function and a value and calculates the derivative of the function in this point
            using the derivative using the formal definition of the derivative as a limit.
            """
            h = 0.00000000001
            top = func(value + h) - func(value)
            bottom = h
            slope = top / bottom
            return slope

        def bisection_method(func, left, right):
            """
            the function receives function and two points and finds a root using the bisection method.
            """
            x0 = left
            x1 = right
            x2 = (x0 + x1) / 2
            while abs(func(x2) >= maxerr):
                if func(x0) * func(x2) < 0:
                    x1 = x2
                else:
                    x0 = x2
                x2 = (x0 + x1) / 2
            return x2

        def newton_raphson_method(func, deriv, left_lim, right_lim):
            """
            The function receives a function, function that calculates the derivative and two point and
            returns a point
            """
            if abs(func(left_lim)) <= maxerr:  # left limit function value is smaller then maxerr hence meets the condition
                return left_lim
            if abs(func(right_lim)) <= maxerr:  # right limit function value is smaller then maxerr hence meets the condition
                return right_lim
            flag = 0  # limit number of iteration so we want have inf loop
            x0 = (left_lim + right_lim) / 2  # first guess
            if abs(func(x0)) <= maxerr:  # x0 function value is smaller then maxerr hence meets the condition
                return x0
            f = func(x0)  # value of func for first guess
            d = deriv(func, x0)  # value of derivative for first guess
            x_n = x0 - f / d  # calculation of next value

            while abs(func(x_n) - f) >= maxerr and flag < 20:  # while we haven't found x that meets the condition
                f = func(x_n)  # func value for next x
                d = deriv(func, x_n)  # derivative value for next x
                x_n = x_n - f / d  # calculation of next value
                flag += 1
                if abs(func(x_n) - f) < maxerr:  # if current x meets the condition return it
                    return x_n
            return

        X = []  # intersection points
        limits = np.arange(a, b, maxerr*75)  # range of points to do the check on
        new_func = lambda x: f1(x) - f2(x)

        for i in range(len(limits) - 1):
            left_lim = limits[i]
            right_lim = limits[i+1]
            if new_func(left_lim) * new_func(right_lim) <= 0:  # f(x) for to points has different sigh hence there is a possible root
                root = newton_raphson_method(new_func, deriv_calc, left_lim, right_lim)  # calculate root with newton raphson method
                if root is not None:  # newton raphson method returned a value
                    X.append(root)
                else:  # newton raphson method didn't find a root
                    X.append(bisection_method(new_func, left_lim, right_lim))

        return X


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


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

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_final_stack(self):
        ass2 = Assignment2()

        def f_Q4_1(x):
            return 0

        def f_Q4_2(x):
            return 39 * np.power(x, 2)

        X1 = ass2.intersections(f_Q4_1, f_Q4_2, -1, 1, maxerr=0.005)
        print(X1, 'test_final_stack Q4')


if __name__ == "__main__":
    unittest.main()
