from numba import cuda
from .lin_alg import dot, add, mult, mult_fac, normalise


_blinn_phong_sphere_signature = ', '.join([
    'float32',  # current_reflectivity
    'float32',  # light_intensity
    'float32',  # distance_to_light
    'float32[:]',  # sphere_ambient
    'float32[:]',  # sphere_diffuse
    'float32[:]',  # sphere_specular
    'float32',  # sphere_shine
    'float32[:]',  # light_ambient
    'float32[:]',  # light_diffuse
    'float32[:]',  # light_specular
    'float32[:]',  # light_unit_vector
    'float32[:]',  # camera_unit_vector
    'float32[:]'  # camera_unit_vector
])


@cuda.jit(
    func_or_sig=_blinn_phong_sphere_signature,
    device=True
)
def blinn_phong_sphere(
        current_reflectivity,
        light_intensity,
        distance_to_light,
        sphere_ambient,
        sphere_diffuse,
        sphere_specular,
        sphere_shine,
        light_ambient,
        light_diffuse,
        light_specular,
        light_unit_vector,
        camera_unit_vector,
        surface_normal_vec,
):
    """
    Returns the new updated pixel value and new multiplicity.
    Follows the shading theory described in https://en.wikipedia.org/wiki/Phong_reflection_model and
    https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_reflection_model (both accessed on 25/06/2022).
    In addition to the Blinn Phong model, this shader implements an intensity of light calculation based off
    the distance to the light
    Referentially transparent and immutable
    Args:
        current_reflectivity:
            a scalar representing the  current reflectivity rate of the pixel.
            This starts at 1, but loses reflectivity the more surfaces
            the ray reflects off (i.e after 3 bounces the surface should
            contribute less toward the final pixel calculation)
        light_intensity:
            The intensity factor of light. Think of this as the distance up to which the light asserts
            full intensity, after which intensity decays with the square of distance
        distance_to_light:
            the distance to the light source
        sphere_ambient:
            a vector of shape (3,) representing the ambient of the sphere object (blinn-phong)
        sphere_specular:
            a vector of shape (3,) representing the specular of the sphere object (blinn-phong)
        sphere_shine:
            a scalar representing the shine of the sphere object (blinn-phong)
        sphere_diffuse:
            a vector of shape (3,) representing the diffuse of the sphere object (blinn-phong)
        light_ambient:
            a vector of shape (3,) representing the ambient of the light object (blinn-phong)
        light_specular:
            a vector of shape (3,) representing the specular of the light object (blinn-phong)
        light_diffuse:
            a vector of shape (3,) representing the diffuse of the light object (blinn-phong)
        light_unit_vector:
            the unit vector of shape (3,) representing the direction from the point
            of intersection on the sphere to the light source
        camera_unit_vector:
            the unit vector of shape (3,) representing the direction from
            the point of intersection on the sphere to the camera
        surface_normal_vec:
            the unit vector of shape (3,) representing the normal at the
            point of intersection on the surface of the sphere
    Returns:
        a vector of shape (3,) representing the delta of pixel (i.e how much does
        the current pixel need to change by)
    """

    normal_dot_light = dot(surface_normal_vec,  light_unit_vector)

    # buffer variables help convert output of device functions as arrays
    buffer_variables = cuda.local.array(shape=(2, 3), dtype='float32')
    buffer_variables[0] = add(light_unit_vector, camera_unit_vector)
    buffer_variables[0] = normalise(buffer_variables[0][:])
    normal_dot_light_plus_cam = dot(surface_normal_vec, buffer_variables[0])
    sign_correction = -1 * (normal_dot_light_plus_cam < 0) + 1 * (normal_dot_light_plus_cam >= 0)
    normal_dot_light_plus_cam_shined = sign_correction * abs(normal_dot_light_plus_cam) ** (sphere_shine / 4)

    # Define diffuse_light * diffuse_sphere * normal_dot_light
    buffer_variables[0][:] = mult(sphere_diffuse, light_diffuse)
    buffer_variables[0][:] = mult_fac(buffer_variables[0], normal_dot_light)

    # Define ambient_light * ambient_sphere, and sum to above term
    buffer_variables[1][:] = mult(sphere_ambient, light_ambient)
    buffer_variables[0][:] = add(buffer_variables[0], buffer_variables[1])

    # Define diffuse_light * diffuse_sphere * normal_dot_light_plus_cam_shined, and add to first term
    buffer_variables[1][:] = mult(sphere_specular, light_specular)
    buffer_variables[1][:] = mult_fac(buffer_variables[1], normal_dot_light_plus_cam_shined)
    buffer_variables[0][:] = add(buffer_variables[0], buffer_variables[1])

    distance_sq = distance_to_light ** 2

    intensity_factor = min(distance_sq, light_intensity) / distance_sq

    return mult_fac(buffer_variables[0], current_reflectivity * intensity_factor)
