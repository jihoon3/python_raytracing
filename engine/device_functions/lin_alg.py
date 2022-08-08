# Linear algebra calculations
from numba import cuda


@cuda.jit(
    func_or_sig='float32[:], float32[:]',
    device=True,
)
def add(vec1, vec2):
    """
    Returns the element-wise addition of two vectors
    Immutable and referentially transparent
    Args:
        vec1:
            the first vector of shape (3,)
        vec2:
            the second vector of shape (3,)
    Returns:
        the shape (3,) after performing element wise addition
    """
    return vec1[0] + vec2[0], vec1[1] + vec2[1], vec1[2] + vec2[2]


@cuda.jit(
    func_or_sig='float32[:], float32[:]',
    device=True,
)
def mult(vec1, vec2):
    """
    Returns the element-wise multiplication of two vectors
    Immutable and referentially transparent
    Args:
        vec1:
            the first vector of shape (3,)
        vec2:
            the second vector of shape (3,)
    Returns:
        the shape (3,) after performing element wise multiplication
    """
    return vec1[0] * vec2[0], vec1[1] * vec2[1], vec1[2] * vec2[2]


@cuda.jit(
    func_or_sig='float32[:], float32',
    device=True,
)
def mult_fac(vec, fac) -> object:
    """
    Returns the product of a vector and factor (element wise)
    Immutable and referentially transparent
    Args:
        vec:
            a vector of shape (3,)
        fac:
            a scalar
    Returns:
        the shape (3,) after each coordinate of vec multiplied by fac
    """
    return vec[0] * fac, vec[1] * fac, vec[2] * fac


@cuda.jit(
    func_or_sig='float32[:], float32[:]',
    device=True
)
def dot(vec1, vec2):
    """
    Returns the dot product of two arrays (of any same size)
    Immutable and referentially transparent
    Args:
        vec1:
            the first vector of shape (3,)
        vec2:
            the second vector of shape (3,)
    Returns:
        the shape (3,) representing the dot product between vec1 and vec2
    """
    output = 0
    for value1, value2 in zip(vec1, vec2):
        output += value1 * value2
    return output


@cuda.jit(
    func_or_sig='float32[:],',
    device=True,
)
def magnitude(vec):
    """
    Returns the magnitude of vector
    Immutable and referentially transparent
    Args:
        vec:
            a vector of shape (3,)
    Returns:
        a scalar representing the absolute value of vec (i.e distance from origin)
    """
    return dot(vec, vec) ** 0.5


@cuda.jit(
    func_or_sig='float32[:],',
    device=True
)
def normalise(vec):
    """
    Returns the normalised the vector
    Immutable and referentially transparent
    Args:
        vec:
            a vector of shape (3,). Cannot be the zero vector (if so, the zero vector is returned)
    Returns:
        the shape (3,) representing the unit vector of vec
    """
    mag = magnitude(vec)
    if mag == 0:
        return 0, 0, 0
    return mult_fac(vec, 1/mag)


@cuda.jit(
    func_or_sig='float32[:], float32[:]',
    device=True
)
def direction(vec1, vec2):
    """
    Returns the unit vector direction from vec1 to vec2
    Args:
        vec1:
            the first vector of shape (3,)
        vec2:
            the second vector of shape (3,)
    Returns:
        the shape (3,) representing the vector from vec1 to vec2
    """
    negative_vec = cuda.local.array(shape=(3,), dtype='float32')
    negative_vec[:] = mult_fac(vec1, -1)
    direction_vector = add(
        vec2,
        negative_vec
    )
    return direction_vector


@cuda.jit(
    func_or_sig='float32[:], float32[:]',
    device=True
)
def normalised_direction(vec1, vec2):
    """
    Returns the unit vector representing the direction from vec1 to vec2
    Args:
        vec1:
            the first vector of shape (3,)
        vec2:
            the second vector of shape (3,)
    Returns:
        the shape (3,) representing the unit vector from vec1 to vec2
    """
    direction_vector = cuda.local.array(shape=(3,), dtype='float32')
    direction_vector[:] = direction(vec1, vec2)
    return normalise(direction_vector)


@cuda.jit(
    func_or_sig='float32[:], float32[:]',
    device=True
)
def cross(vec1, vec2):
    """
    Returns the cross of two 3-dimensional arrays
    Immutable and referentially transparent
    Args:
        vec1:
            the first vector of shape (3,)
        vec2:
            the second vector of shape (3,)
    Returns:
        the shape (3,) representing the cross between vec1 and vec2
    """
    x = vec1[1] * vec2[2] - vec1[2] * vec2[1]
    y = vec1[2] * vec2[0] - vec1[0] * vec2[2]
    z = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    return x, y, z


@cuda.jit(
    func_or_sig='float32[:], float32[:]',
    device=True
)
def reflection_flat(ray_unit_vector, normal_unit_vector):
    """
    Calculates the unit vector of the reflected ray (assumes surface is flat, thus works on flat surfaces).
    Function assumes that the ray_unit_vector is going to hit the flat surface (i.e the surface and the POSITIVE
    direction intersect)
    Immutable and referentially transparent
    Args:
        ray_unit_vector:
            the unit vector representing the direction of the ray. Shape = (3,)
        normal_unit_vector:
            the unit vector representing the normal at the surface of reflection. Shape = (3,)
    Returns:
        an array of shape (3,) representing the unit vector of the direction of the ray after being reflected
    """
    dotted = dot(ray_unit_vector, normal_unit_vector)
    second_term = cuda.local.array(shape=(3,), dtype='float32')
    second_term[:] = mult_fac(normal_unit_vector, dotted * - 2)
    # return ray_unit_vector - 2 * dotted * normal_unit
    return add(
        ray_unit_vector,
        second_term
    )
