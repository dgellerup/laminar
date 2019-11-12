import pytest
from scipy import spatial
import numpy as np
import itertools
from typing import Callable
import multiprocessing
from laminar import laminar

def get_pwdist_indices(sequences):
    """
    From a list of sequences get lower triangle indices tuples (i,j)
    for pairwise comparisons.

    Parameters
    ----------
    sequences : list of strings
        list of (likely amino acid) strings

    Returns
    -------
    ind_tuples : list
        ist of tuples (i,j)

    """
    ind_tuples = []
    L = len(sequences)
    for i, j in itertools.product(list(range(L)), list(range(L))):
        if i <= j:
            ind_tuples.append((i, j))
    return(ind_tuples)

def cosine_dist(u: np.ndarray, v:np.ndarray)->float:
    """Define the cosine distance metric between two numeric vectors u,v"""
    assert isinstance(u, np.ndarray)
    assert isinstance(v, np.ndarray)
    return(spatial.distance.cosine(u, v))

def tuple_to_dist(ij: tuple, matrix: np.ndarray, func: Callable)->float:
    """ given an ith row and jth row of a matrix apply a distance metric """
    i = ij[0] # i from tuple
    j = ij[1] # j from tuple
    u = matrix[i,] # u-vector is the ith row
    v = matrix[j,] # v-vector is the jth row
    return func(matrix[i,], matrix[j,])

def vectorize_tuple_to_dist(ijs:list, matrix:np.ndarray, func: Callable)->list:
    """vectorize the tuple to dist function."""
    return [tuple_to_dist(ij, matrix, func) for ij in ijs]

def collapse(iter_flow_result:dict)->list:
    """ Return iter_flow dict result as a single flat list"""
    nested_lists = [iter_flow_result[k] for k in iter_flow_result.keys()]
    flat_list    = [item for sublist in nested_lists for item in sublist]
    return(flat_list)

def test_in_series(n = 65):
    """ Test the process in series (no iter.flow)"""
    mat = np.random.rand(65,5)
    ijs = get_pwdist_indices(np.arange(65))
    print(len(ijs))
    distances = vectorize_tuple_to_dist(ijs, matrix = mat, func = cosine_dist)
    assert(len(distances) == len(ijs))

testspace = [(10, 2),
             (30, 2),
             (50, 2),
             (65, 2),
             (10, 4),
             (30, 4),
             (50, 4),
             (65, 4),
             (10, 8),
             (30, 8),
             (50, 8),
             (65, 8)]

@pytest.mark.parametrize("n, cores", testspace)
def test_using_laminar_iterflow(n, cores):
    """ Test the process in parrallel (no iter.flow)"""
    # Here is the process in series
    mat = np.random.rand(n,5)
    ijs = get_pwdist_indices(np.arange(n))
    distances = vectorize_tuple_to_dist(ijs, matrix = mat, func = cosine_dist)
    assert(len(distances) == len(ijs))

    r = laminar.iter_flow(function = vectorize_tuple_to_dist,
                      data = ijs,
                      matrix = mat,
                      func = cosine_dist,
                      cores= cores,
                      sort_results = True)
    [print(k) for k in r.keys()]
    assert np.allclose(collapse(r), distances, atol=1e-3)
    return(r)