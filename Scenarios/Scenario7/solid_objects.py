import numpy as np
from Objects.SolidObjects import Sphere
from .loop_colours import loop_colours
# Contains all the data for the spheres involved in scenario 7

sphere_settings = [
    {
        "coordinates": np.array([-2.0, 4.400000095367432, -0.699999988079071], dtype="float32"),
        "ambient": np.array([0.30000001192092896, 0.20000000298023224, 0.10000000149011612], dtype="float32"),
        "diffuse": np.array([0.925000011920929, 0.6779999732971191, 0.4000000059604645], dtype="float32"),
        "specular": np.array([1.0, 1.0, 1.0], dtype="float32"),
        "reflect": 0.2,
        "shine": 80,
        "radius": 0.3,
    },
    {
        "coordinates": np.array([-3.4000000953674316, 6.0, -0.5], dtype="float32"),
        "ambient": np.array([0.10000000149011612, 0.0, 0.0], dtype="float32"),
        "diffuse": np.array([0.699999988079071, 0.0, 0.0], dtype="float32"),
        "specular": np.array([1.0, 1.0, 1.0], dtype="float32"),
        "reflect": 0.2,
        "shine": 20,
        "radius": 0.5,
    },
    {
        "coordinates": np.array([5.0, 8.0, -0.30000001192092896], dtype="float32"),
        "ambient": np.array([0.10000000149011612, 0.10000000149011612, 0.0], dtype="float32"),
        "diffuse": np.array([0.699999988079071, 0.699999988079071, 0.0], dtype="float32"),
        "specular": np.array([1.0, 1.0, 1.0], dtype="float32"),
        "reflect": 0.2,
        "shine": 20,
        "radius": 0.7,
    },
    {
        "coordinates": np.array([3.0, 12.0, 0.10000000149011612], dtype="float32"),
        "ambient": np.array([0.01600000075995922, 0.07599999755620956, 0.08399999886751175], dtype="float32"),
        "diffuse": np.array([0.1599999964237213, 0.7599999904632568, 0.8399999737739563], dtype="float32"),
        "specular": np.array([0.8619999885559082, 0.9639999866485596, 0.9800000190734863], dtype="float32"),
        "reflect": 0.3,
        "shine": 20,
        "radius": 1.1,
    },
    {
        "coordinates": np.array([-2.5, 17.0, 0.5], dtype="float32"),
        "ambient": np.array([0.027499999850988388, 0.08240000158548355, 0.42750000953674316], dtype="float32"),
        "diffuse": np.array([0.3215999901294708, 0.5647000074386597, 0.9685999751091003], dtype="float32"),
        "specular": np.array([0.694100022315979, 0.8039000034332275, 0.9882000088691711], dtype="float32"),
        "reflect": 1,
        "shine": 30,
        "radius": 1.5,
    },
    {
        "coordinates": np.array([1.7999999523162842, 20.0, -0.6499999761581421], dtype="float32"),
        "ambient": np.array([0.12999999523162842, 0.08240000158548355, 0.20000000298023224], dtype="float32"),
        "diffuse": np.array([0.6269999742507935, 0.3068999946117401, 0.8744999766349792], dtype="float32"),
        "specular": np.array([0.694100022315979, 0.8039000034332275, 0.9882000088691711], dtype="float32"),
        "reflect": 0.3,
        "shine": 30,
        "radius": 0.35,
    },
    {
        "coordinates": np.array([5.0, 30.0, 5.0], dtype="float32"),
        "ambient": np.array([0.20000000298023224, 0.04690000042319298, 0.07649999856948853], dtype="float32"),
        "diffuse": np.array([0.7803999781608582, 0.21570000052452087, 0.6118000149726868], dtype="float32"),
        "specular": np.array([0.94100022315979, 0.5039000034332275, 0.9882000088691711], dtype="float32"),
        "reflect": 0.85,
        "shine": 30,
        "radius": 6,
    },
    {
        "coordinates": np.array([-7.900000095367432, 25.0, -0.30000001192092896], dtype="float32"),
        "ambient": np.array([0.06369999796152115, 0.19509999454021454, 0.05389999970793724], dtype="float32"),
        "diffuse": np.array([0.2549000084400177, 0.7803999781608582, 0.21570000052452087], dtype="float32"),
        "specular": np.array([0.694100022315979, 0.8039000034332275, 0.9882000088691711], dtype="float32"),
        "reflect": 0.5,
        "shine": 30,
        "radius": 0.7,
    },
    {
        "coordinates": np.array([-9.5, 29.0, 0.0], dtype="float32"),
        "ambient": np.array([0.2353000044822693, 0.016200000420212746, 0.11270000040531158], dtype="float32"),
        "diffuse": np.array([0.2353000044822693, 0.12939999997615814, 0.9020000100135803], dtype="float32"),
        "specular": np.array([1.0, 1.0, 1.0], dtype="float32"),
        "reflect": 0.5,
        "shine": 30,
        "radius": 1,
    },
    {
        "coordinates": np.array([-5.300000190734863, 10.0, 0.20000000298023224], dtype="float32"),
        "ambient": np.array([0.1137000024318695, 0.09709999710321426, 0.030400000512599945], dtype="float32"),
        "diffuse": np.array([0.9097999930381775, 0.7764999866485596, 0.24310000240802765], dtype="float32"),
        "specular": np.array([1.0, 1.0, 1.0], dtype="float32"),
        "reflect": 0.5,
        "shine": 30,
        "radius": 1.2,
    },
    {
        "coordinates": np.array([0.0, 5.0, -0.550000011920929], dtype="float32"),
        "ambient": np.array([0.10100000351667404, 0.020099999383091927, 0.03629999980330467], dtype="float32"),
        "diffuse": np.array([0.8077999949455261, 0.1607999950647354, 0.29019999504089355], dtype="float32"),
        "specular": np.array([1.0, 1.0, 1.0], dtype="float32"),
        "reflect": 0.5,
        "shine": 30,
        "radius": 0.45,
    },
    {
        "coordinates": np.array([-10.0, 17.0, -0.11999999731779099], dtype="float32"),
        "ambient": np.array([0.02500000037252903, 0.02500000037252903, 0.125], dtype="float32"),
        "diffuse": np.array([0.20000000298023224, 0.20000000298023224, 1.0], dtype="float32"),
        "specular": np.array([1.0, 1.0, 1.0], dtype="float32"),
        "reflect": 0.5,
        "shine": 30,
        "radius": 0.88,
    },
    {
        "coordinates": np.array([2.0, 4.0, -0.800000011920929], dtype="float32"),
        "ambient": np.array([0.125, 0.05000000074505806, 0.125], dtype="float32"),
        "diffuse": np.array([1.0, 0.4000000059604645, 1.0], dtype="float32"),
        "specular": np.array([1.0, 1.0, 1.0], dtype="float32"),
        "reflect": 0.5,
        "shine": 30,
        "radius": 0.2,
    },
    {
        "coordinates": np.array([-8.0, -50.0, 0.0], dtype="float32"),
        "ambient": np.array([0.125, 0.05000000074505806, 0.125], dtype="float32"),
        "diffuse": np.array([0.699999988079071, 0.5, 1.0], dtype="float32"),
        "specular": np.array([1.0, 1.0, 1.0], dtype="float32"),
        "reflect": 0.85,
        "shine": 25,
        "radius": 1,
        "name": "pretty_mirror_sphere"
    },
] + [
        {
            "coordinates": np.array([
                4.5 - 0.0 * i if i % 2 == 0 else -4 + 0.00 * i,
                -i - 10,
                -0.8,
            ], dtype='float32'),
            'ambient': loop_colours[i % 5]['ambient'],
            'diffuse': loop_colours[i % 5]['diffuse'],
            'specular': loop_colours[i % 5]['specular'],
            'shine': 30,
            'reflect': 0.5,
            'radius': 0.21,

        }
        for i in range(60)
]


plane_settings = [
    {
        'name': 'floor',
        'north': np.array([0, 1, 0], dtype='float32'),
        'east': np.array([1, 0, 0], dtype='float32'),
        'point_on_surface': np.array([0, 0, -1], dtype='float32'),
        'ambient': np.array([0.0184313, 0.0580392, 0.0184313], dtype='float32'),
        'diffuse': np.array([0.184313, 0.580392, 0.184313], dtype='float32'),
        'specular': np.array([0.274509, 0.745098, 0.274509], dtype='float32'),
        'shine': 60,
        'radius': 100000,
        'reflect': 0.5
    },
    {
        'name': 'sky',
        'north': np.array([0, 1, 0], dtype='float32'),
        'east': np.array([-1, 0, 0], dtype='float32'),
        'radius': 100000,
        'point_on_surface': np.array([0, 0, 25], dtype='float32'),
        'ambient': np.array([0.016, 0.086, 0.094], dtype='float32'),
        'diffuse': np.array([0.16, 0.86, 0.94], dtype='float32'),
        'specular': np.array([0.862, 0.932, 0.98], dtype='float32'),
    }
]

ind1 = -1
for ind1, setting in enumerate(sphere_settings):
    new_setting = {**{'name': f'sphere{ind1}'}, **setting}
    Sphere(**new_setting)

for ind2, setting in enumerate(plane_settings):
    new_setting = {**{'name': f'sphere{ind2 + ind1 + 1}'}, **setting}
    Sphere.create_flat_surface(**setting)
