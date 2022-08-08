import numpy as np
from typing import Tuple, List, TYPE_CHECKING, ItemsView, KeysView, ValuesView, Any, Union
from copy import deepcopy
from numba import cuda
from .Excs import SceneError
from ExcThreading import ExcThreading
from engine import render_image


if TYPE_CHECKING:
    from Objects import BaseObject, MetaObjects, SolidObjects
    from Objects._AutoNumpyUpdate import _AutoNumpyUpdate

_FORBIDDEN: dict = {
    '_object_directory': {},
    '_camera_updated': True,
    '_light_updated': True,
    '_spheres_updated': True,
    '_eps_reflect_updated': True,
    '_frames': None,
    '_gpu_initialised': False,
    '_device_background_colour': None,
    '_device_camera': None,
    '_device_rays': None,
    '_device_spheres': None,
    '_device_light': None,
    '_device_other_data': None,
    '_device_output_frame': None
}


class _SceneInterface:
    """
    Class handles all objects involved. Ensures there is only 1 viewer, 1 light source, etc.
    """
    __slots__ = tuple(_FORBIDDEN)
    _EPS: float = 0.02
    _MAX_REFLECTIONS: float = 3.
    _RESOLUTION: Tuple[int, int] = None
    _SPECIAL_NAMES: set = {
        '_light',
        '_camera'
    }

    def __init__(self):
        for name, value in _FORBIDDEN.items():
            super().__setattr__(name, deepcopy(value))

    def __repr__(self):
        """
        Summary of scene
        """
        directory: dict = super().__getattribute__('_object_directory')
        number_of_spheres: int = len([i for i, j in directory.items() if i not in {'_light', '_camera'}])
        camera: bool = '_camera' in directory
        light: bool = '_light' in directory
        camera_updated: bool = super().__getattribute__('_camera_updated')
        light_updated: bool = super().__getattribute__('_light_updated')
        spheres_updated: bool = super().__getattribute__('_spheres_updated')
        gpu_initialised: bool = super().__getattribute__('_gpu_initialised')
        camera_state: str = "up to date" if not camera_updated\
            else "device copy required" if camera \
            else "no camera defined"
        light_state: str = "up to date" if not light_updated \
            else "device copy required" if light \
            else "no light defined"
        sphere_state: str = "up to date" if not spheres_updated \
            else "device copy required" if number_of_spheres \
            else "no spheres defined"

        output = f"""Scene Interface Object:
        \tCamera defined: {camera},
        \tLight defined: {light},
        \tNumber of spheres: {number_of_spheres},
        \tGPU camera: {camera_state},
        \tGPU light: {light_state},
        \tGPU spheres: {sphere_state},
        \tGPU initialised: {gpu_initialised},
        \tEpsilon value: {self._EPS},
        \tMax reflections: {self._MAX_REFLECTIONS}
        """
        return output

    def __getitem__(self, item) -> 'BaseObject':
        """
        Retrieves item from object_directory
        """
        directory = super().__getattribute__('_object_directory')
        return directory[item]

    def set_eps(self, eps: float):
        """
        Sets the epsilon value. This is to handle floating point precision errors
        """
        if not 0 < eps <= 0.1:
            raise ValueError(f'eps must be between 0 (excl.) and 0.1 (incl.)')
        self.__class__._EPS = eps
        super().__setattr__('_eps_reflect_updated', True)

    def set_reflect(self, reflect: int):
        """
        Sets the max reflections value
        """
        if not 0 <= reflect <= 10:
            raise ValueError(f'max reflections must be between 0 (incl.) and 10 (incl.)')
        self.__class__._MAX_REFLECTIONS = reflect
        super().__setattr__('_eps_reflect_updated', True)

    def items(self) -> ItemsView[str, 'BaseObject']:
        """
        Iterate over _object_directory.items()
        """
        return super().__getattribute__('_object_directory').items()

    def keys(self) -> KeysView[str]:
        """
        Iterate over _object_directory.keys()
        """
        return super().__getattribute__('_object_directory').keys()

    def values(self) -> ValuesView['BaseObject']:
        """
        Iterate over _object_directory.keys()
        """
        return super().__getattribute__('_object_directory').values()

    def __contains__(self, item: str) -> bool:
        """
        Checks if item in _object_directory
        """
        return item in super().__getattribute__('_object_directory')

    def __getattribute__(self, key: str) -> Any:
        if key in _FORBIDDEN:
            raise NotImplementedError(f'You are not allowed to directly access the following {_FORBIDDEN}')
        return super().__getattribute__(key)

    def __setattr__(self, key: str, value: Any) -> None:
        if key in _FORBIDDEN:
            raise NotImplementedError(f'You are not allowed to directly write to the following {_FORBIDDEN}')
        super().__setattr__(key, value)

    def _encoded_camera(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the camera location, pixels array and rays unit vector (in that order)
        """
        camera: 'MetaObjects.Camera' = self['_camera']
        rays = camera._construct_rays()
        return camera.coordinates.astype('float32'),\
            camera.background_colour.astype('float32'),\
            rays.astype('float32')

    def _encoded_light(self) -> np.ndarray:
        """
        Returns the encoded light data (to be transferred to cuda).
        shape=(4, 3)
        where:
            array[0] is the coordinate vector of light
            array[1] is the ambient vector
            array[2] is the diffuse vector
            array[3] is the specular vector
        """
        light: 'MetaObjects.Light' = self['_light']
        encoded = np.vstack([
            light.coordinates,
            light.ambient,
            light.diffuse,
            light.specular,
            np.full(shape=(3,), fill_value=light.intensity ** 2, dtype='float32')
        ])
        return encoded.astype('float32')

    def _encode_spheres(self) -> np.ndarray:
        """
        Returns an encoded sphere array
        shape=(512,5,3)
        where:
            array[i] is the ith sphere (will fill the array with zeros if not enough spheres)
            array[i][0] is the coordinate vector of the centre of the ith sphere
            array[i][1] is the ambient vector of the ith sphere
            array[i][2] is the diffuse vector of the ith sphere
            array[i][3] is the specular vector of the ith sphere
            array[i][4] is the vector representing [shine, reflect, radius] of the ith sphere
        Note in the cuda kernel, you can identify if a sphere is a placeholder or an actual sphere by checking if
        radius == 0
        """
        from Objects import SolidObjects
        output = np.zeros(
            shape=(512, 5, 3)
        )
        sphere_data = np.stack([
            np.stack([
                sphere.coordinates,
                sphere.ambient,
                sphere.diffuse,
                sphere.specular,
                np.array([sphere.shine, sphere.reflect, sphere.radius], dtype='float32')
            ])
            for sphere in self.values() if isinstance(sphere, SolidObjects.Sphere)
        ])
        output[:len(sphere_data), :, :] = sphere_data
        return output.astype('float32')

    def _first_time_initialise(self) -> None:
        """
        Initialises the gpu memories for the first time.
        Keeps a reference to the cuda DeviceNDArray object
        """
        super().__setattr__('_gpu_initialised', True)
        self.__class__._RESOLUTION = self['_camera'].resolution
        camera_location, background_colour, rays = self._encoded_camera()
        light_encoded = self._encoded_light()
        spheres_encoded = self._encode_spheres()
        threads = [
            ExcThreading(
                target=lambda target, array: super(self.__class__, self).__setattr__(target, cuda.to_device(array)),
                args=(name, value)
            )
            for name, value in [
                ('_device_background_colour', background_colour),
                ('_device_camera', camera_location),
                ('_device_rays', rays),
                ('_device_light', light_encoded),
                ('_device_spheres', spheres_encoded),
                ('_device_other_data', np.array([self._EPS, self._MAX_REFLECTIONS], dtype='float32')),
                ('_device_output_frame', np.zeros(shape=rays.shape, dtype='float32')),
            ]
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def _transfer_to_gpu(self):
        """
        Method transfers data to gpu to prepare for processing
        """
        gpu_initialised: bool = super().__getattribute__('_gpu_initialised')
        if not gpu_initialised:
            self._first_time_initialise()
        else:
            if super().__getattribute__('_camera_updated'):
                camera_location, background_colour, rays = self._encoded_camera()
                threads = [
                    ExcThreading(
                        target=lambda target, array: super(
                            self.__class__,
                            self
                        ).__getattribute__(target).copy_to_device(array),
                        args=(name, value)
                    )
                    for name, value in [
                        ('_device_background_colour', background_colour),
                        ('_device_camera', camera_location),
                        ('_device_rays', rays),
                    ]
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            if super().__getattribute__('_light_updated'):
                light_encoded = self._encoded_light()
                super().__getattribute__('_device_light').copy_to_device(light_encoded)
            if super().__getattribute__('_spheres_updated'):
                spheres_encoded = self._encode_spheres()
                super().__getattribute__('_device_spheres').copy_to_device(spheres_encoded)
            if super().__getattribute__('_eps_reflect_updated'):
                super().__getattribute__('_device_other_data').copy_to_device(
                    np.array([self._EPS, self._MAX_REFLECTIONS], dtype='float32')
                )
        super().__setattr__('_camera_updated', False)
        super().__setattr__('_light_updated', False)
        super().__setattr__('_spheres_updated', False)
        super().__setattr__('_eps_reflect_updated', False)

    def _check_identical_frame(self) -> Union[np.ndarray, None]:
        """
        Checks if identical frame is being captured (in which case we just
        duplicate the last frame).
        Appends the last frame onto frames and returns the frame if identical frame is
        captured, else returns None
        """
        frames = super().__getattribute__('_frames')
        if frames is None or frames.shape[0] == 0:
            return None
        conditions = [
            super().__getattribute__('_camera_updated'),
            super().__getattribute__('_light_updated'),
            super().__getattribute__('_spheres_updated'),
            super().__getattribute__('_eps_reflect_updated'),
        ]
        if any(conditions):
            return None
        frame = deepcopy(frames[-1])
        self._add_frame_to_frames(frame)
        return frame

    def _add_frame_to_frames(self, frame: np.ndarray):
        """
        Method appends the given frame to the _frames attribute
        """
        frames = super().__getattribute__('_frames')
        shape = frame.shape
        if frames is None:
            new_frames = frame.reshape((1,) + shape)
        else:
            new_frames = np.append(frames, frame.reshape((1,) + shape), axis=0)
        super().__setattr__('_frames', new_frames)

    def capture_frame(self) -> np.ndarray:
        """
        Captures the frame and appends it to the frames array.
        Also returns the newly created frame
        Raises a SceneError if there is something wrong with the arrangement of objects
        """
        frame = self._check_identical_frame()
        if frame is not None:
            return frame
        self._check_scene()
        self._transfer_to_gpu()
        blocks_per_grid = self['_camera'].resolution
        threads_per_block = len([i for i in self.keys() if i not in self._SPECIAL_NAMES])
        device_output_frame = super().__getattribute__('_device_output_frame')
        render_image[blocks_per_grid, threads_per_block](
            super().__getattribute__('_device_background_colour'),
            super().__getattribute__('_device_camera'),
            super().__getattribute__('_device_rays'),
            super().__getattribute__('_device_light'),
            super().__getattribute__('_device_spheres'),
            super().__getattribute__('_device_other_data'),
            device_output_frame
        )
        frame = device_output_frame.copy_to_host()
        self._add_frame_to_frames(frame)
        return frame

    def _check_scene(self):
        """
        Checks for scene errors
        Raises a SceneError if there is something wrong with the arrangement of objects
        """
        errors = []
        camera: bool = '_camera' in self
        light: bool = '_light' in self
        number_of_spheres: int = len([i for i, j in self.items() if i not in self._SPECIAL_NAMES])

        if not camera:
            errors.append(f'Camera is not defined')
        if not light:
            errors.append(f'Light is not defined')
        if not number_of_spheres:
            errors.append(f'No objects to render')
        elif number_of_spheres > 512:
            errors.append(f'The maximum number of spheres is 512 (current = {number_of_spheres})')

        if errors:
            raise SceneError('\n'.join(errors))

    def de_register_object(self, object_name: str):
        """
        De-registers an object given its name
        """
        directory: dict = super().__getattribute__('_object_directory')
        if object_name not in directory:
            raise KeyError(f'Given name "{object_name}" is not registered')
        self._assign_updated(directory[object_name])
        del directory[object_name]

    def de_register_objects(self, object_names: List[str]):
        """
        De-registers multiple objects to the global_directory
        """
        object_names = set(object_names)
        unknown = [i for i in object_names if i not in self]
        if unknown:
            raise KeyError(f'The following names are unrecognised: {unknown}')
        rollback_objects = []
        try:
            for name in object_names:
                object_item = self[name]
                self.de_register_object(name)
                rollback_objects.append(object_item)
        except (ValueError, KeyError):
            for object_item in rollback_objects:
                self.register_object(object_item)
            raise

    def register_objects(self, object_items: List['BaseObject']):
        """
        Registers multiple objects to the global_directory
        """
        rollback_names = []
        try:
            for object_item in object_items:
                self.register_object(object_item)
                rollback_names.append(object_item.name)
        except (ValueError, KeyError):
            for name in rollback_names:
                self.de_register_object(name)
            raise

    def register_object(self, object_item: 'BaseObject'):
        """
        Registers an object to the global directory
        """
        from Objects import BaseObject, MetaObjects
        if not isinstance(object_item, BaseObject):
            raise TypeError(f'Objects being registered must be an instance of a child of Objects.BaseObject')
        if object_item.__class__ is MetaObjects.Camera:
            self._check_resolution(object_item.resolution)
        self._check_name(object_item.name, object_class=object_item.__class__)
        directory: dict = super().__getattribute__('_object_directory')
        directory[object_item.name] = object_item
        self._assign_updated(object_item)

    def _check_resolution(self, resolution):
        """
        Checks that the resolution of the given camera is correct
        """
        res = self.__class__._RESOLUTION
        if res is not None:
            if res != resolution:
                raise ValueError(f'The camera resolution is invalid (received {resolution}, expecting {res})')

    def _check_name(self, name, object_class):
        """
        Checks that the name is unique. Special names "_camera" and "_light" for the camera and light objects
        """
        from Objects.MetaObjects import Light, Camera
        directory: dict = super().__getattribute__('_object_directory')
        if name in directory:
            raise ValueError(f'Given name "{name}" already exists')
        if object_class is Camera and name != '_camera':
            raise ValueError(f'Camera name must be "_camera"')
        elif object_class is Light and name != '_light':
            raise ValueError(f'Light name must be "_light"')

    def _assign_updated(self, object_item: Union['BaseObject', '_AutoNumpyUpdate']):
        """
        Assign updated when required
        """
        from Objects.MetaObjects import Light, Camera
        from Objects.SolidObjects import Sphere
        if object_item.name in self:
            if object_item.__class__ is Camera:
                super().__setattr__('_camera_updated', True)
            elif object_item.__class__ is Light:
                super().__setattr__('_light_updated', True)
            elif object_item.__class__ is Sphere:
                super().__setattr__('_spheres_updated', True)

    @property
    def frames(self) -> np.ndarray:
        _frames: np.ndarray = super().__getattribute__('_frames')
        return _frames

    @property
    def eps(self) -> float:
        return self._EPS

    @property
    def reflect(self) -> float:
        return self._MAX_REFLECTIONS


scene = _SceneInterface()
