from numba import cuda


@cuda.jit(
    func_or_sig='float32[:, :],',
    device=True
)
def get_min_positive(array):
    """
    Function returns the index and the value of the smallest value in the
    array subject to it being positive
    """
    _min = -1
    _index = -1
    for index in range(array.shape[0]):
        if array[index][0] > 0:
            if array[index][0] < _min:
                _min = array[index][0]
                _index = index
            if _min == -1:
                _index = index
                _min = array[index][0]
    return _index, _min
