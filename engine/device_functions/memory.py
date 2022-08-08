# Do not import this. For some reason, cuda hates the fact that the shared memory arrays are being returned
from numba import cuda


@cuda.jit(
    func_or_sig='float32[:], float32[:], float32[:, :, :], float32[:, :], float32[:, :, :], float32[:]',
    device=True
)
def create_shared_memory(
        background_colour,
        camera_location,
        unit_rays,
        light_encoded,
        spheres_encoded,
        other_data
):
    """
    Function creates shared memory arrays required for rendering an image
    Returns:
        shared_spheres:
            The (512, 5, 3) array of spheres
        shared_sphere_intersections:
            The array of intersection data (shape (512, 4)). Each row will have distance,
            hit_x, hit_y and hit_z data (all -1 if not hit)
        shared_intersection_data:
            An array of shape (3,). It contains the index of sphere hit, the distance from the ray to the light, and
            the number of spheres that block the ray from seeing the light
        shared_screen_data:
            An array of shape (8, 3).
                - array[0] is the screen pixel value
                - array[1] is the camera location
                - array[2] is the unit ray
                - array[3] is the ray origin
                - array[4:8] is the encoded light data
        shared_other_data:
            An array of shape (2,), containing epsilon and number of reflections
        shared_calculation_data:
            An array of shape (3,3) keeping track of the unit normal of the sphere, the unit vector of
            ray to camera, and the unit vector of ray to light in that order
    """
    pixel_x = cuda.blockIdx.x
    pixel_y = cuda.blockIdx.y
    thread_pos = cuda.threadIdx.x

    # Load the spheres data into memory
    shared_spheres = cuda.shared.array(
        (512, 5, 3),
        dtype='float32'
    )
    for height in range(5):
        for width in range(3):
            shared_spheres[thread_pos][height][width] = spheres_encoded[thread_pos][height][width]

    # Load the screen_pixel, camera_location, unit_ray, and light data into shared
    # Schema:
    # array[0] is screen pixel
    # array[1] is camera location
    # array[2] is unit ray
    # array[3] is the ray origin
    # array[4:8] are the encoded light data
    shared_screen_data = cuda.shared.array(
        (8, 3),
        dtype='float32'
    )

    # Load the other data into shared memory
    shared_other_data = cuda.shared.array(
        (2,),
        dtype='float32'
    )

    if thread_pos == 0:
        # Populate shared_other_data and shared_screen_data
        for axis in range(3):
            shared_screen_data[0][axis] = background_colour[axis]
            shared_screen_data[1][axis] = camera_location[axis]
            shared_screen_data[2][axis] = unit_rays[pixel_x, pixel_y][axis]
            shared_screen_data[3][axis] = camera_location[axis]
            for k in range(4):
                shared_screen_data[4 + k][axis] = light_encoded[k][axis]

        shared_other_data[0] = other_data[0]
        shared_other_data[1] = other_data[1]

    # Initialise shared memory to aid calculations
    shared_intersection_data = cuda.shared.array(
        (3,),
        dtype='int32'
    )  # The first element is the index of the sphere that the ray hits. The second index is the distance
    # from ray to light.should be the number of. The third index is the number of spheres that block the light from
    # seeing the ray

    shared_calculation_data = cuda.shared.array(
        (3, 3),
        dtype='float32'
    )  # An array keeping track of sphere normal, vector to camera, vector to light (all units) in that order

    shared_sphere_intersections = cuda.shared.array(
        (512, 4),
        dtype='float32'
    )  # The array of intersection data. Each row will have distance, hit_x, hit_y and hit_z data (all -1 if not hit)

    return shared_spheres, \
        shared_sphere_intersections, \
        shared_intersection_data, \
        shared_screen_data, \
        shared_other_data, \
        shared_calculation_data
