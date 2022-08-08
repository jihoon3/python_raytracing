# calculations involving spheres
from numba import cuda
from engine.device_functions import lin_alg


@cuda.jit(
    func_or_sig='float32[:], float32[:], float32[:], float32',
    device=True
)
def sphere_intersection(ray_origin, ray_unit_vector, sphere_centre, sphere_radius):
    """
    For a given ray position, direction, and a spherical object, function calculates
    if the ray intersects the sphere (being tangential DOES NOT count as an intersection here).
    If it does, it returns the distance, as well as the coordinates of the surface of the sphere
    it intersects. Note that if the ray intersects the sphere in the negative direction, this also
    does not count as an intersection

    Immutable and referentially transparent
    Args:
        ray_origin:
            a vector representing the origin of ray. shape = (3,)
        ray_unit_vector:
            a vector representing the unit direction of the ray. shape = (3,)
        sphere_centre:
            a vector representing the location of centre of sphere. shape = (3,)
        sphere_radius:
            a scalar representing the radius of the sphere
    Returns:
        (distance, coordinates) where
        distance:
            the positive distance between the ray and the surface of the sphere. Distance is -1
            if intersection does not occur
        coordinates:
            the vector representing of shape (3,) the point on the surface of the sphere
            that intersects with the ray. coordinates is (-1, -1, -1) if intersection does not occur
        normal_reverse:
            An indicator for if the ray intersected the sphere externally (1) or internally (-1)
    """
    # Using quadratic formula
    b = 2 * (
            ray_unit_vector[0] * (ray_origin[0] - sphere_centre[0])
            + ray_unit_vector[1] * (ray_origin[1] - sphere_centre[1])
            + ray_unit_vector[2] * (ray_origin[2] - sphere_centre[2])
    )
    c = (
            (ray_origin[0] - sphere_centre[0]) ** 2
            + (ray_origin[1] - sphere_centre[1]) ** 2
            + (ray_origin[2] - sphere_centre[2]) ** 2
            - sphere_radius ** 2
    )
    discriminant = b ** 2 - 4 * c
    if discriminant <= 0:
        return -1, (-1, -1, -1), 1

    d_sqrt = discriminant ** 0.5
    t1 = (-b - d_sqrt) / 2
    t2 = (-b + d_sqrt) / 2
    t = t1 * (t1 > 0) + t2 * (t1 < 0)  # if t1 is negative, take t2
    distance = t * (t > 0.01) + -1 * (t <= 0.01)
    x = (ray_origin[0] + ray_unit_vector[0] * distance) * (distance > 0) + -1 * (distance == -1)
    y = (ray_origin[1] + ray_unit_vector[1] * distance) * (distance > 0) + -1 * (distance == -1)
    z = (ray_origin[2] + ray_unit_vector[2] * distance) * (distance > 0) + -1 * (distance == -1)

    distance_vector_x = ray_origin[0] + ray_unit_vector[0] * distance/2 - sphere_centre[0]
    distance_vector_y = ray_origin[1] + ray_unit_vector[1] * distance/2 - sphere_centre[1]
    distance_vector_z = ray_origin[2] + ray_unit_vector[2] * distance/2 - sphere_centre[2]
    halfway_distance2 = distance_vector_x ** 2 + distance_vector_y ** 2 + distance_vector_z ** 2
    radius2 = sphere_radius ** 2
    normal_reverse = -1 * (halfway_distance2 < radius2) + 1 * (halfway_distance2 >= radius2)

    return distance, (x, y, z), normal_reverse
