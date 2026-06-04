"""Tests for sparse operations."""

import numpy as np

from topo.base import sparse


def test_arr_unique():
    arr = np.array([3, 1, 2, 1, 3])
    res = sparse.arr_unique(arr)
    np.testing.assert_array_equal(res, [1, 2, 3])


def test_arr_union():
    ar1 = np.array([1, 2])
    ar2 = np.array([2, 3])
    res = sparse.arr_union(ar1, ar2)
    np.testing.assert_array_equal(res, [1, 2, 3])


def test_arr_intersect():
    ar1 = np.array([1, 2])
    ar2 = np.array([2, 3])
    res = sparse.arr_intersect(ar1, ar2)
    np.testing.assert_array_equal(res, [2])


def test_sparse_sum():
    ind1 = np.array([0, 1])
    data1 = np.array([1.0, 2.0])
    ind2 = np.array([1, 2])
    data2 = np.array([3.0, 4.0])
    r_ind, r_data = sparse.sparse_sum(ind1, data1, ind2, data2)
    np.testing.assert_array_equal(r_ind, [0, 1, 2])
    np.testing.assert_allclose(r_data, [1.0, 5.0, 4.0])


def test_sparse_diff():
    ind1 = np.array([0, 1])
    data1 = np.array([1.0, 2.0])
    ind2 = np.array([1, 2])
    data2 = np.array([3.0, 4.0])
    r_ind, r_data = sparse.sparse_diff(ind1, data1, ind2, data2)
    np.testing.assert_array_equal(r_ind, [0, 1, 2])
    np.testing.assert_allclose(r_data, [1.0, -1.0, -4.0])


def test_sparse_euclidean():
    ind1 = np.array([0, 1])
    data1 = np.array([3.0, 0.0])
    ind2 = np.array([1, 2])
    data2 = np.array([0.0, 4.0])
    assert sparse.sparse_euclidean(ind1, data1, ind2, data2) == 5.0
