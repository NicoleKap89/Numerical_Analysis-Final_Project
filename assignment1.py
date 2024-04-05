"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random


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

        def thomas(y, z, c, d):
            """
            function that solves tridiagonal systems of equations
            """
            n = len(z)
            aa, bb, cc, dd = map(np.array, (y, z, c, d))
            x = [0] * n
            for i in range(1, n):
                bb[i] = bb[i] - aa[i - 1] * cc[i - 1] / bb[i - 1]
                dd[i] = dd[i] - aa[i - 1] * dd[i - 1] / bb[i - 1]
            x[-1] = dd[-1] / bb[-1]
            for i in range(n - 2, -1, -1):
                x[i] = (dd[i] - cc[i] * x[i + 1]) / bb[i]
            return x


        def get_bezier_coef(points):
            """
            the function finds the a&b needed to the Bezier curve
            """
            n = len(points) - 1

            C = 4 * np.identity(n)
            np.fill_diagonal(C[1:], 1)
            np.fill_diagonal(C[:, 1:], 1)
            C[0, 0] = 2
            C[n - 1, n - 1] = 7
            C[n - 1, n - 2] = 2

            a = C.diagonal(-1)
            b = C.diagonal()
            c = C.diagonal(1)

            P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
            P[0] = points[0] + 2 * points[1]
            P[n - 1] = 8 * points[n - 1] + points[n]

            A = thomas(a, b, c, P)
            B = [0] * n
            for i in range(n - 1):
                B[i] = 2 * points[i + 1] - A[i + 1]
            B[n - 1] = (A[n - 1] + points[n]) / 2

            return A, B

        def get_cubic(a, b, c, d):
            """
            the function returns Bezier cubic formula
            """
            return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t,2) * c + np.power(t, 3) * d

        def result(x):
            """
            the function returns the y value for given x that matches the Bezier curve
            """
            n = len(points) - 1
            if n < 2:
                return lambda x: x
            index = 0
            for i in range(n):
                if points[i][0] <= x <= points[i + 1][0]:
                    index = i
                    break
            A, B = get_bezier_coef(points)
            cubic = get_cubic(points[index], A[index], B[index], points[index + 1])
            t = (x - points[index][0]) / (points[index + 1][0] - points[index][0])
            return cubic(t)[1]

        x_values = np.linspace(a, b, n)
        points = np.array([(x, f(x)) for x in x_values])

        return lambda x: result(x)

##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

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
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

if __name__ == "__main__":
    unittest.main()
