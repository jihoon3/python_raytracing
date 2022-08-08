from numba import cuda
import engine.device_functions as device_functions


_render_image_signature = ', '.join([
    'float32[:]',  # background_colour
    'float32[:]',  # camera_location
    'float32[:, :, :]',  # unit_rays
    'float32[:, :]',  # light_encoded
    'float32[:, :, :]',  # spheres_encoded
    'float32[:]',  # other_data
    'float32[:, :, :]',  # output_frame
])


@cuda.jit(
    func_or_sig=_render_image_signature,
    device=False
)
def render_image(
        background_colour,
        camera_location,
        unit_rays,
        light_encoded,
        spheres_encoded,
        other_data,
        output_frame,
):
    """
    Main processing kernel. Intended to be used with h by w blocks (where h and w is the resolution)
    and up to 512 threads (n threads, 1 for each spherical object).

    Args:
        background_colour:
            An array of shape (3,) indicating the initial pixel value
        camera_location:
            an array of three coordinates x, y, z
        unit_rays:
            The unit vectors of the rays at start. Shape is (h, w, 3)
        light_encoded:
            the light value encoded. Shape is (4, 3)
        spheres_encoded:
            the spheres objects encoded. Shape is (512, 5, 3)
        other_data:
            the epsilon and number of iterations. Shape is (2,)
        output_frame:
            the output screen of size (height, width, 3) - to be written to
    """

    pixel_x = cuda.blockIdx.x
    pixel_y = cuda.blockIdx.y
    thread_pos = cuda.threadIdx.x
    """
    Creates shared memory. The following items are defined
        shared_spheres:
            The (512, 5, 3) array of spheres
        shared_sphere_intersections:
            The array of intersection data (shape (512, 4)). Each row will have distance,
            hit_x, hit_y and hit_z data (all -1 if not hit)
        shared_intersection_data:
            An array of shape (3,). It contains the index of sphere hit, the distance from the ray to the light, and
            the number of spheres that block the ray from seeing the light
        shared_scene_data:
            An array of shape (9, 3).
                - array[0] is the screen pixel value
                - array[1] is the camera location
                - array[2] is the unit ray
                - array[3] is the ray origin
                - array[4:8] is the encoded light data:
                    array[4] is the coordinate vector of light
                    array[5] is the ambient vector
                    array[6] is the diffuse vector
                    array[7] is the specular vector
                - array[8] contains 3 separate data points:
                    1. epsilon
                    2. number of reflections
                    3. the light intensity
                
        shared_calculation_data:
            An array of shape (3,3) keeping track of the unit normal of the sphere, the unit vector of
            ray to camera, and the unit vector of ray to light in that order
    """
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
    # array[4:8] is the encoded light data:
    #   array[4] is the coordinate vector of light
    #   array[5] is the ambient vector
    #   array[6] is the diffuse vector
    #   array[7] is the specular vector
    # array[8] contains 3 separate data points:
    #   1. epsilon
    #   2. number of reflections
    #   3. the light intensity
    shared_scene_data = cuda.shared.array(
        (9, 3),
        dtype='float32'
    )

    if thread_pos == 0:
        # Populate the shared_scene_data
        for axis in range(3):
            shared_scene_data[0][axis] = background_colour[axis]
            shared_scene_data[1][axis] = camera_location[axis]
            shared_scene_data[2][axis] = unit_rays[pixel_x, pixel_y][axis]
            shared_scene_data[3][axis] = camera_location[axis]
            for k in range(4):
                shared_scene_data[4 + k][axis] = light_encoded[k][axis]

        shared_scene_data[8][0] = other_data[0]
        shared_scene_data[8][1] = other_data[1]
        shared_scene_data[8][2] = light_encoded[4][0]

    # Initialise shared memory to aid calculations
    shared_intersection_data = cuda.shared.array(
        (3,),
        dtype='float32'
    )  # The first element is the index of the sphere that the ray hits. The second index is the distance
    # from ray to light.should be the number of. The third index is the number of spheres that block the light from
    # seeing the ray

    shared_calculation_data = cuda.shared.array(
        (3, 3),
        dtype='float32'
    )  # An array keeping track of sphere normal, vector to camera, vector to light (all units) in that order

    shared_sphere_intersections = cuda.shared.array(
        (512, 5),
        dtype='float32'
    )
    # The array of intersection data. Each row will have:
    # 0: the distance
    # 1: normal inverse indicator (1 means intersection occurred externally, -1 means internally)
    # 2, 3, 4: the coordinates of the point on surface of intersection

    current_reflectivity = 1.

    for i in range(int(shared_scene_data[8][1])):
        cuda.syncthreads()

        # Determine the distances to each sphere
        # Adjust origin by eps * direction
        if thread_pos == 0:
            for axis in range(3):
                shared_scene_data[3][axis] = shared_scene_data[3][axis] +\
                                              shared_scene_data[2][axis] * shared_scene_data[8][0]
        cuda.syncthreads()
        distance, hit_coordinates, normal_multiplier = device_functions.spherical.sphere_intersection(
            shared_scene_data[3],  # ray_origin
            shared_scene_data[2],  # ray_unit_vector
            shared_spheres[thread_pos][0],  # sphere_centre
            shared_spheres[thread_pos][4][-1],  # sphere_radius
        )

        shared_sphere_intersections[thread_pos][0] = distance
        shared_sphere_intersections[thread_pos][1] = normal_multiplier
        shared_sphere_intersections[thread_pos][2:] = hit_coordinates

        # Given distance, we reduce the distance by epsilon as a percentage divided by 10
        for axis in range(3):  # adjust by epsilon
            shared_sphere_intersections[thread_pos][axis + 2] -= shared_scene_data[2][axis] *\
                                                                 shared_sphere_intersections[thread_pos][0] *\
                                                                 (shared_scene_data[8][0] / 10)
        shared_sphere_intersections[thread_pos][0] *= (1 - shared_scene_data[8][0] / 10)

        cuda.syncthreads()  # Wait for all threads to finish in the block to calculate which sphere was hit

        if thread_pos == 0:
            # Only one thread finds the minimum distance
            index, min_dist = device_functions.numerical_utils.get_min_positive(shared_sphere_intersections)
            shared_intersection_data[0] = index
            if int(shared_intersection_data[0]) != -1:
                # Calculate unit normal, unit vector to light, unit vector to camera, distance to light,
                # and unit vector of reflected rays
                shared_calculation_data[0] = device_functions.lin_alg.normalised_direction(  # unit normal
                    shared_spheres[index][0],  # centre of sphere
                    shared_sphere_intersections[index][2:5]  # point on surface of sphere
                )
                shared_calculation_data[0] = device_functions.lin_alg.mult_fac(
                    shared_calculation_data[0],
                    shared_sphere_intersections[index][1]
                )
                shared_calculation_data[1] = device_functions.lin_alg.normalised_direction(  # unit vector of ray to cam
                    shared_sphere_intersections[index][2:5],  # point on surface of sphere
                    shared_scene_data[1]  # camera location
                )

                shared_calculation_data[2] = device_functions.lin_alg.direction(  # vector from ray to light
                    shared_sphere_intersections[index][2:5],  # point on surface of sphere
                    shared_scene_data[4]  # light location
                )  # note we save the output of this "lin_alg.direction" function into shared_calculation_data[2]
                # since we need it to be converted to an array.

                shared_intersection_data[1] = device_functions.lin_alg.magnitude(shared_calculation_data[2])  # the
                # distance to light

                shared_calculation_data[2] = device_functions.lin_alg.normalise(shared_calculation_data[2])  # unit
                # direction to light

                shared_scene_data[2] = device_functions.lin_alg.reflection_flat(  # Reflected ray
                    shared_scene_data[2],  # original ray vector
                    shared_calculation_data[0]  # the unit normal of surface
                )
                for axis in range(3):
                    shared_scene_data[3][axis] = shared_sphere_intersections[index][axis + 2] +\
                                                  shared_calculation_data[0][axis] * shared_scene_data[8][0]
                    # this is the new origin plus an epsilon amount * surface normal

            # Set number of obstructions to 0
            shared_intersection_data[2] = 0

        cuda.syncthreads()
        index = int(shared_intersection_data[0])

        if index == -1:
            # No sphere got intersected, no more interaction available
            break

        # Determine how many spheres are in the way between the ray and the light
        distance, intersection_coordinates, normal_multiplier = device_functions.spherical.sphere_intersection(
            shared_scene_data[3],  # The new ray origin (keyword is ray_origin)
            shared_calculation_data[2],  # The unit direction of ray to light (keyword is ray_unit_vector)
            shared_spheres[thread_pos][0],  # sphere_centre
            shared_spheres[thread_pos][4][-1],  # sphere_radius
        )
        if 0 < distance < shared_intersection_data[1]:
            # Distance to object is shorter than distance to light.
            cuda.atomic.add(shared_intersection_data, 2, 1)
        cuda.syncthreads()

        if thread_pos == 0:
            for axis in range(3):
                # Reset new origin to true origin (state before we added an epsilon * surface normal)
                shared_scene_data[3][axis] = shared_scene_data[3][axis] - \
                                              shared_calculation_data[0][axis] * shared_scene_data[8][0]
            if shared_intersection_data[2] == 0:
                # Ray sees light
                pixel_delta = device_functions.blinn_phong.blinn_phong_sphere(
                    current_reflectivity,  # current_reflectivity
                    shared_scene_data[8][2],  # light_intensity
                    shared_intersection_data[1],  # distance_to_light
                    shared_spheres[index][1],  # sphere_ambient
                    shared_spheres[index][2],  # sphere_diffuse
                    shared_spheres[index][3],  # sphere_specular
                    shared_spheres[index][4][0],  # sphere_shine
                    shared_scene_data[5],  # light_ambient
                    shared_scene_data[6],  # light_diffuse
                    shared_scene_data[7],  # light_specular
                    shared_calculation_data[2],  # light_unit_vector
                    shared_calculation_data[1],  # camera_unit_vector
                    shared_calculation_data[0]  # surface_normal_vec
                )
                for axis in range(3):
                    shared_calculation_data[0][axis] = pixel_delta[axis]
                # Note we save the results to shared_calculation_data as we must convert output to array
                x, y, z = device_functions.lin_alg.add(
                    shared_scene_data[0],
                    shared_calculation_data[0]
                )
                shared_scene_data[0][0] = min(max(0, x), 1)
                shared_scene_data[0][1] = min(max(0, y), 1)
                shared_scene_data[0][2] = min(max(0, z), 1)

            current_reflectivity *= shared_spheres[index][4][1]

    # Write results to a new pixel
    if thread_pos == 0:
        for axis in range(3):
            output_frame[pixel_x, pixel_y][axis] = shared_scene_data[0][axis]
