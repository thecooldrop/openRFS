import numpy as np
import scipy.optimize
import scipy.spatial

def ospa(first, second, c, p):
    """
    Computes the OSPA distance between first and second set of inputs. The OSPA metric is computed as described in

    D. Schuhmacher, B. Vo and B. Vo, "A Consistent Metric for Performance Evaluation of Multi-Object Filters,"
    in IEEE Transactions on Signal Processing, vol. 56, no. 8, pp. 3447-3457, Aug. 2008, doi: 10.1109/TSP.2008.920469.

    The first two parameters are matrices where each row represents an input vector. The second and third parameters
    are the cutoff and order of the OSPA metric as described in the paper.

    :param first: Ndarray-like object representing the inputs
    :param second:Ndarray-like object representing the inputs
    :param c: A floating point number representing the cutoff distance
    :param p: An integer representing the order of OSPA metric
    :return: Three element tuple which contains OSPA, localization OSPA and cardinality OSPA as respective elements
    """
    m = first.shape[0]
    n = second.shape[0]
    if m == 0 and n == 0:
        return 0, 0, 0
    if m==0 or n == 0:
        return c, 0, c

    div = max(m,n)
    distances = scipy.spatial.distance.cdist(first, second)
    cost_matrix = np.minimum(distances, c)**p
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    cost = cost_matrix[row_ind, col_ind].sum()

    ospa = ((1/div) * (cost + c**p * abs(n-m)))**(1/p)
    ospa_loc = ((1/div) * cost)**(1/p)
    ospa_card = (c**p * abs(n-m) / div)**(1/p)
    return ospa, ospa_loc, ospa_card

if __name__=='__main__':
    a = np.random.random((10,4))
    b = np.random.random((0,4))
    print(ospa(a, b, 1000000,1))