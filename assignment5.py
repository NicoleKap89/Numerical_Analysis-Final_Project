"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""

from functionUtils import AbstractShape
import numpy as np


class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, area):
        self.shape_area = area

    def area(self):
        return self.shape_area


class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """



    def area(self, contour: callable, maxerr=0.001)->np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        def area_for_t_points_calc(t):
            area_points = contour(t)
            curr_area = 0
            for i in range(len(area_points)):
                curr_area += 0.5 * ((area_points[i][0]-area_points[i-1][0]) * (area_points[i][1]+area_points[i-1][1]))
            return curr_area

        n = 300
        prev_area = area_for_t_points_calc(n)
        cur_area = area_for_t_points_calc(int(n * 1.2))
        while n < 10000:
            if abs(abs(cur_area) - abs(prev_area)) / abs(cur_area) < maxerr:
                return np.float32(abs(cur_area))
            n = int(n * 1.2)
            prev_area = cur_area
            cur_area = area_for_t_points_calc(n)
        return np.float32(abs(cur_area))

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """
        def adjustment(x: list, y: list, z: list) -> int:
            adj = (y[1] - x[1]) * (z[0] - y[0]) - (y[0] - x[0]) * (z[1] - y[1])
            return 1 if adj > 0 else 2 if adj < 0 else 0

        def line_fitted(section: np.ndarray) -> list:
            len_seg = len(section)
            xs = section[:, 0]
            ys = section[:, 1]
            p = np.polyfit(xs, ys, 3)
            gap = ((section[-1][0] - section[0][0]) / (len_seg / 2))
            x_gap = np.arange(section[0][0], section[-1][0], gap)
            f = np.poly1d(p)
            y_gap = f(x_gap)
            return np.column_stack((x_gap, y_gap)).tolist()

        def points_sorted(points: list) -> np.ndarray:
            x = np.array(points)
            avg = np.mean(x, axis=0)
            xs = x - avg
            angles = np.angle(xs[:, 0] + 1j * xs[:, 1])
            sorted_indices = np.argsort(angles)
            x_sort = xs[sorted_indices, :] + avg
            return x_sort

        def convex(points: list) -> list:
            con = []
            len_points = len(points)
            if len_points <= 2:
                return con
            cur_point = min(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
            x = cur_point
            while True:
                con.append(x)
                y = (x + 1) % len_points
                for i in range(len_points):
                    if adjustment(points[x], points[i], points[y]) == 2:
                        y = i
                x = y
                if x != cur_point:
                    continue
                else:
                    break
            return con

        inf = [sample() for _ in range(500)]
        inf_sorted = points_sorted(inf).tolist()
        indexes_of_con = convex(inf_sorted)

        index_of_points_change = [indexes_of_con[0]]
        lt = False
        for i in range(len(indexes_of_con) - 1):
            current = inf_sorted[indexes_of_con[i]]
            next_point = inf_sorted[indexes_of_con[i + 1]]
            if next_point[0] <= current[0]:
                if not lt:
                    lt = True
                    index_of_points_change.append(indexes_of_con[i])
            elif next_point[0] > current[0]:
                if lt:
                    lt = False
                    index_of_points_change.append(indexes_of_con[i])

        res = []
        for i, first_index in enumerate(index_of_points_change[:-1]):
            last_index = index_of_points_change[i + 1]
            section = inf_sorted[first_index:last_index] if first_index < last_index else inf_sorted[first_index:] + inf_sorted[:last_index]
            res += line_fitted(np.array(section))

        first_index = index_of_points_change[-1]
        last_index = index_of_points_change[0]
        if first_index > last_index:
            res += line_fitted(np.array(inf_sorted[first_index:] + inf_sorted[:last_index]))
        else:
            res += line_fitted(np.array(inf_sorted[first_index:last_index]))

        field = 0
        field += sum((res[i + 1][0] - res[i][0]) * (((res[i + 1][1]) + (res[i][1])) / 2) for i in range(len(res) - 1))
        fin_res = MyShape(abs(field))
        return fin_res



##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
