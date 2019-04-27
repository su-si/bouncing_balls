import pymunk
import pygame
from pygame.color import THECOLORS
import math
import os
import sys
import numpy as np
from enum import Enum

from src.utils import utils
import pygame_code.pymunk_utils as pymunk_utils
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


shape_types = ['circle', 'rectangle', 'triangle']
shape_type_string_dtype = '|S10'  # increase the number (here 10) if you ever add longer bouncer type names


class ObjectTypes(Enum):
        circle = 0, 'circle'
        rectangle = 1, 'rectangle'
        triangle = 2, 'triangle'
        outer_wall = 10, 'outer_wall'
        inner_wall = 11, 'inner_wall'
        unknown = 12, 'unknown'

        def __new__(cls, value, name):
            '''https://stackoverflow.com/questions/6060635/convert-enum-to-int-in-python'''
            member = object.__new__(cls)
            member._value_ = value
            member.fullname = name
            return member

        def __int__(self):
            return self.value

        @classmethod
        def fromstr(cls, typestr):
            if type(typestr) in [str, np.str_]:
                return getattr(cls, typestr)
            elif type(typestr) == np.bytes_:
                return cls.frombytestr(typestr)

        @classmethod
        def frombytestr(cls, typestr):
            typestr = typestr.decode('UTF-8')
            return getattr(cls, typestr)

        @classmethod
        def fromstrarr(cls, typestrarr):
            if typestrarr.dtype.type == np.bytes_:
                keyfun = np.vectorize(lambda s: cls.frombytestr(s).value)
            elif typestrarr.dtype.type ==  np.str_:
                keyfun = np.vectorize(lambda s: cls.fromstr(s).value)
            elif typestrarr.dtype.type == np.object_:
                typestrarr = typestrarr.astype(shape_type_string_dtype)
                    # => dtype.type == numpy.bytes_
                assert typestrarr.dtype.type == np.bytes_
                keyfun = np.vectorize(lambda s: cls.frombytestr(s).value)
            else:
                print("Error: Unsupported label type. Label was: "+str(typestrarr[:5]))
                raise TypeError("Unrecognized type was: "+str(typestrarr.dtype.type)+". Expected np.str_ or np.bytes_")
            return keyfun(typestrarr)


class AbstractBouncer:
    ''' Encapsulates one pygame body (or even multiple connected ones?)'''
    def __init__(self, mass, vel):
        ''':param mass: positive float
           :param vel: vec2d'''
        self.bodies = []
        self.body = None # shortcut
        self.shapes = {} # a dict containing a list for each body, indexed by the bodies themselves
        self.bouncer_type = "Not set yet"


    @property
    def mass(self):
        return self.body.mass

    @property
    def velocity(self):
        return self.body.velocity

    @property
    def diameter(self):
        raise NotImplementedError
        #main_shape = self.body.shapes[0]
        #if hasattr(main_shape, 'radius'):
        #    return main_shape.radius
        #elif hasattr(main_shape, 'height'):
        #    if hasattr(main_shape, 'width'):
        #        return max([main_shape.height, main_shape.width])
        #    else:
        #        return main_shape.height
        #elif hasattr(main_shape, 'width'):
        #    return main_shape.width
        #else:
        #    raise NotImplementedError

    def draw(self, screen):
        ''' Use if additional drawing needed, e.g. for filling boxes'''
        pass


class CircleBouncer(AbstractBouncer):

    def __init__(self, mass, vel, radius=10., mass_propto_area=False, mass_scale_factor=0.1): #, rot=0., w=0., radius=10.):
        ''':param mass: positive float
           :param radius: positive float
           :param vel: vec2d
           :param mass_propto_area: if True, calculate mass as shape_area * mass_scale_factor'''
        if mass_propto_area:
            assert mass is None, "Mass will be generated automatically as == shape-area"
            mass = math.pi * radius**2 * mass_scale_factor
        else:
            assert mass >= 0
        super().__init__(mass, vel)
        moment = pymunk.moment_for_circle(mass, 0, radius, vel)
        body = pymunk.Body(mass, moment)
        body.velocity = vel
        bouncer_shape = pymunk.Circle(body, radius=radius)
        bouncer_shape.color = THECOLORS["white"]
        self.shapes[body] = [bouncer_shape]
        self.bodies = [body]
        self.body = body
        self.radius = radius
        self.bouncer_type = ObjectTypes.circle

    @property
    def diameter(self):
        return self.radius


class RectangleBouncer(AbstractBouncer):

    def __init__(self, mass, vel, height=10, width=10, mass_propto_area=False, mass_scale_factor=0.1):
        if mass_propto_area:
            assert mass is None, "Mass will be generated automatically as == shape-area"
            mass = height*width * mass_scale_factor
        else:
            assert mass >= 0
        super().__init__(mass, vel)
        moment = pymunk.moment_for_box(mass, size=(width, height))
        body = pymunk.Body(mass, moment)
        body.velocity = vel
        self.height = height
        self.width = width
        # counterclockwise rectangle
        # Need a Poly(), because pymunks segments apparently can't collide with each other
        vertices = [(-(width - 1) / 2., -(height - 1) / 2.),
                    ((width - 1) / 2., -(height - 1) / 2.),
                    ((width - 1) / 2., (height - 1) / 2.),
                    (-(width - 1) / 2., (height - 1) / 2.)]

        vertices = list(map(pymunk.Vec2d, vertices))
        shape = pymunk.Poly(body, vertices)#, radius=1)#, radius=math.sqrt(height**2 / 4. + width**2 / 4.))
        #shape = pymunk.Poly(body, [(0, -(height - 1) / 2.), (0, (height-1) / 2.)], radius = width)
        shape.color = THECOLORS["white"]
        self.shapes[body] = [shape]
        self.bodies = [body]
        self.body = body
        self.bouncer_type = ObjectTypes.rectangle

    @property
    def diameter(self):
        return max([self.height, self.width])

class TriangleBouncer(AbstractBouncer):
    ''' triangle with two same-length sides'''
    def __init__(self, mass, vel, height=10, width=None, mass_propto_area=False, mass_scale_factor=0.1):
        if width is None:
            # equal-sided triangle
            # h**2 + w**2/4 = side**2   & w=side --> h**2 = 3/4*side**2 -->  2/sqrt(3)*h = side
            width = 2./np.sqrt(3) * height
        self.height = height
        self.width = width
        if mass_propto_area:
            assert mass is None, "Mass will be generated automatically as == shape-area"
            mass = self.height * self.width / 2. * mass_scale_factor
        else:
            assert mass >= 0
        super().__init__(mass, vel)
        # Need a Poly(), because pymunks segments apparently can't collide with each other
        vertices = pymunk_utils.same_sided_triangle_verts(height, width)
        assert (pymunk.Vec2d((0,0)) - pymunk_utils.compute2DPolygonCentroid_so(vertices))\
                   .length < 0.00001, "error in centroid calculation / vertex positioning"

        moment = pymunk.moment_for_poly(mass, vertices, offset=(0, 0), radius=0)
        body = pymunk.Body(mass, moment)
        body.velocity = vel
        shape = pymunk.Poly(body, vertices)#, radius=1)#, radius=math.sqrt(height**2 / 4. + width**2 / 4.))
        shape.color = THECOLORS["white"]
        self.shapes[body] = [shape]
        self.bodies = [body]
        self.body = body
        self.bouncer_type = ObjectTypes.triangle

    @property
    def diameter(self):
        return max([self.width, self.height])




#                                           Bouncer classes, end
# ---------------------------------------------------------------
# Functions, start

def random_bouncer(shape_types=['circle'], sz_min=10., sz_max=20., same_sided=False, m_min=1., m_max=10., v0_max=0):

        shape_type = shape_types[np.random.randint(0, len(shape_types))]
        assert shape_type in shape_types
        mass = utils.rand_float(m_min, m_max, shape=())
        vel = tuple(utils.rand_float(v0_max, v0_max, shape=(2,)))
        radius = utils.rand_float(sz_min / 2., sz_max / 2., shape=())

        if shape_type is 'circle':
            bouncer = CircleBouncer(mass, vel, radius=radius)
        elif shape_type is 'rectangle':
            height = radius * 2
            if not same_sided:
                width = utils.rand_float(sz_min, sz_max, shape=())
            else:
                width = height
            bouncer = RectangleBouncer(mass, vel, height=height, width=width)
        elif shape_type is 'triangle':
            height = radius * 2
            width = None if same_sided else utils.rand_float(sz_min, sz_max, shape=())
            bouncer = TriangleBouncer(mass, vel, height=height, width=width)
        else:
            raise NotImplementedError("Shape type " + shape_type + " has not been implemented so far.")
        return bouncer


def draw_random_bouncer(surface, pymunk_draw_options, shape_types=['circle'], sz_min=10., sz_max=20., same_sided=False,
                        colors=[THECOLORS['gray100'], THECOLORS['gray17'], THECOLORS['gray12'], THECOLORS['gray39'], THECOLORS['gray85']]):
    surf_w, surf_h = surface.get_size()
    assert sz_max < min([surf_w, surf_h])

    color = colors[np.random.randint(len(colors))]
    position = [utils.rand_float(0, surf_w), utils.rand_float(0, surf_h)]
    #angle = utils.rand_float(0, 2*math.pi)

    bouncer = random_bouncer(shape_types=shape_types, sz_min=sz_min, sz_max=sz_max, same_sided=same_sided)
    #bouncer.body.angle = angle # does not work, would need an environment step to move the angle update to the bouncers shapes

    for key, shape_list in bouncer.shapes.items():
        for shape in shape_list:
            shape.color = color
            pymunk_utils.draw_shape(surface, shape, position, pymunk_draw_options)
