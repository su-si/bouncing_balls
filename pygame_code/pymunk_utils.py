import numpy as np
import matplotlib.path as mplPath
import copy
import pymunk
import pymunk.pygame_util as ppu
import pygame
from pygame.color import THECOLORS

from src.utils import utils
from src.utils.decorators import *




def shapes_abs_position(shape):
    '''
    :param shape: a pymunk shape
    :return: for Circles: the circle center point in abs coords
             for Segments: [seg.a, seg.b] in abs coords
             for Polys: a list of transformed vec2ds
    '''
    assert shape.body is not None, "shape needs to be attached to a body"
    if type(shape) == pymunk.Circle:
        return abs_position(shape.offset, shape.body)
    elif type(shape) ==pymunk.Segment:
        A = abs_position(shape.a, shape.body)
        B = abs_position(shape.b, shape.body)
        return [A, B]
    elif type(shape) == pymunk.Poly:
        verts = []
        for v in shape.get_vertices():
            verts.append(abs_position(v, shape.body))
    else:
        raise NotImplementedError("shapes_abs_position() does not yet work for shape type "+str(type(shape)))

def abs_position(avec2d, body):
    ''' vec2d: a vector in body's coordinates
        returns the vector in absolute coordinates
        from https://stackoverflow.com/questions/23592312/how-to-get-circle-coordinates-in-pymunk/23604420#23604420'''
    if not type(avec2d) == pymunk.Vec2d:
        avec2d = pymunk.Vec2d(avec2d)
    return body.position + avec2d.rotated(body.angle)



#@accepts(pymunk.Body, pymunk.Space, pymunk.BB, int, int)
#@returns(bool)
def assign_valid_location(body, space, bb, maxiter=100, radius=0):
    assert isinstance(body, pymunk.Body) and isinstance(space, pymunk.Space) and isinstance(bb, pymunk.BB)
    loc_valid = False
    counter = 0
    while (not loc_valid) and counter < maxiter:
        loc_valid = True
        pos_y = np.random.randint(bb.bottom + radius, bb.top - radius)
        pos_x = np.random.randint(bb.left + radius, bb.right - radius)
        body.position = pos_x, pos_y
        for cshp in body.shapes:
            sqinfo = space.shape_query(cshp)
            if len(sqinfo) > 0:
                loc_valid = False
    return loc_valid



# ----- For measuring distances, etc -------- #

def line_object_distance(segment, object):
    ''' Calculate the distance between a line that extends into the directon of a segment shape,
        and a circle shape or a polygon shape.
        Make sure to give a radius to a polygon object so that the true distance, from the
        object outline to the segment, can be calculated.
        Returns: a vector from the closest point on the segment to the object boundary. And a distance.'''
    v_s = segment.a - segment.b
    n_s = [-v_s[1], v_s[0]] # normal to v_s
    n_s = n_s / np.linalg.norm(n_s)
    p_o = object.body.position
    d = (segment.a - p_o).dot(n_s)
    dist_vec = n_s * d
    d = abs(d) - segment.radius - object.radius
    return dist_vec, d


def line_object_collision_time(segment, object):
    direct_dist_vec, d = line_object_distance(segment, object)
    if d < 0 :
        delta_t = 0
    else:
        n = direct_dist_vec / np.linalg.norm(direct_dist_vec)
        v_orth_o = object.body.velocity.dot(n)
        v_orth_s = segment.body.velocity.dot(n)
        if (v_orth_o == v_orth_s):
            delta_t = pymunk.inf # relative velocity = 0
        else:
            delta_t = d / (v_orth_o - v_orth_s)
            if delta_t < 0:
                delta_t = pymunk.inf

    return direct_dist_vec, d, delta_t




# ---------- Geometry operations -------------- #

def reflect_vector_at(vector, segment):
    '''like light reflection'''
    dx = segment.a.x - segment.b.x
    dy = segment.a.y - segment.b.y
    n = pymunk.Vec2d(-dy, dx).normalized()
    s = pymunk.Vec2d(dx, dy).normalized()

    reflected = vector.dot(s) * s - vector.dot(n) * n  # reverse part orthogonal to s (parallel to n), keep part parallel to s
    return reflected

def compute2DPolygonCentroid_so(vertex_list):
    '''
    from https://stackoverflow.com/questions/2792443/finding-the-centroid-of-a-polygon
    use as check only (for now)
    '''
    vertexCount = len(vertex_list)
    centroid = pymunk.Vec2d(0, 0);
    signedArea = 0.0;
    x0 = 0.0; # Current vertex X
    y0 = 0.0; # Current vertex Y
    x1 = 0.0; # Next vertex X
    y1 = 0.0; # Next vertex Y
    a = 0.0;  # Partial signed area

    # For all vertices
    for i in range(len(vertex_list)):
        x0 = vertex_list[i].x;
        y0 = vertex_list[i].y;
        x1 = vertex_list[(i+1) % vertexCount].x;
        y1 = vertex_list[(i+1) % vertexCount].y;
        a = x0*y1 - x1*y0;
        signedArea += a;
        centroid.x += (x0 + x1)*a;
        centroid.y += (y0 + y1)*a;

    signedArea *= 0.5;
    centroid.x /= (6.0*signedArea);
    centroid.y /= (6.0*signedArea);

    return centroid

def centroid_same_sided_triangle(height, width):
    '''centroid relative to lower left corner, assuming "width" is the length of the horizontal bottom side'''
    return pymunk.Vec2d( float(width) / 2. , float(height) / 3.)

def same_sided_triangle_verts(height, width):
        '''return list of verts relative to centroid'''
        verts = [(-width/2., -height/3.),  (0, 2./3.*height),  (width/2., -height/3.)]
        return [pymunk.Vec2d(v) for v in verts]


# --------------------------
# ------ pre / post solvers:

def begin_check_conservation_laws(arbiter, space, data):
        ''' We don't want rotation to take away or add momentum. Add as a collision handlers' begin()
            and the corresponding separate() below to ensure this.
        '''
        #return True
        for sp in arbiter.shapes:
            if sp.body.torque != 0 or sp.surface_velocity.length != 0 or arbiter.surface_velocity.length != 0:
                print("found something!")
        if arbiter.restitution != 1.:
            print("found another thing!")
            raise AttributeError("Energy was lost in collision!")
        data['momentum_beginning'] = [sp.body.mass * sp.body.velocity  for sp in arbiter.shapes]
        data['kin_energy_beginning'] = [sp.body.kinetic_energy for sp in arbiter.shapes]
        print("velocities: " + str([sp.body.velocity.length for sp in arbiter.shapes]))
        return True
        #set_ = arbiter.contact_point_set
        #if len(set_.points) > 0:
        #    assert len(set_.points) == 1, "Strange things happening - more contact points for this collision? Inspect and maybe remove assertion."
        #    segment_shape = arbiter.shapes[0]
            # calculate the collision normal:
            #bouncer_shape = arbiter.shapes[1]
            #seg_dir = segment_shape.a - segment_shape.b
            #normal = seg_dir.perpendicular_normal()
            #set_.normal = normal
            # set distance to 0, hopefully prevents bugs/further changes of movement direction
            #set_.points[0].distance = 0
        #arbiter._contacts = set_
        #return True

def separate_check_conservation_laws(arbiter, space, data):

    momentum_end = [sp.body.mass * sp.body.velocity  for sp in arbiter.shapes]
    if  (sum(momentum_end) - sum(data['momentum_beginning'])).length > 0.000001:
            print("not good")
            print("momentum diff: "+str(sum(momentum_end) - sum(data['momentum_beginning'])))
            print("velocities: "+str([sp.body.velocity.length  for sp in arbiter.shapes]))
            #raise AssertionError
    kin_en_end = [sp.body.kinetic_energy for sp in arbiter.shapes]
    if  abs((sum(kin_en_end) - sum(data['kin_energy_beginning']))) > 0.000001:
        print("wah, kin energy diff: "+str(sum(kin_en_end) - sum(data['kin_energy_beginning'])))
        #raise AssertionError
    assert arbiter.restitution == 1.
    return

#  ----  partially copied from  https://github.com/rszeto/moving-symbols/blob/master/moving_symbols/moving_symbols.py
#  ----  and partially from ..
#  ---- :)

def _bouncer_bouncer_pre_handler_remove_angular(arbiter, space, data):
        """ -- This is useful only for circles!! All others gain / lose angular momentum anyways.
        Remove angular velocity of the symbol.
        This handler sets the angular velocity of the symbol to zero, which prevents the physics
        simulation from adding kinetic energy due to rotation.
        """
        data['angular_velocities'] = {shp.body: shp.body.angular_velocity for shp in arbiter.shapes}
        data['old_bodies'] = {i: shp.body for i, shp in enumerate(arbiter.shapes)}
        data['old_shapes'] = arbiter.shapes
        #set_ = arbiter.contact_point_set
        if len(arbiter.contact_point_set.points) > 0:
            for shp in arbiter.shapes:
                body = shp.body
                body.angular_velocity = 0
           # set_.points[0].distance = 0 # No!
        #arbiter.contact_point_set = set_
        return True


def _bouncer_bouncer_post_handler_remove_angular(arbiter, space, data):
        """Restore angular velocity of the symbols.
        This handler restores the angular velocity after the collision has been solved. It looks
        up the fixed angular velocity from the Symbol instance associated with the body in the
        collision.
        """
        if len(arbiter.contact_point_set.points) > 0:
            for shp in arbiter.shapes:
                body = shp.body
                try:
                    body.angular_velocity = data['angular_velocities'][body]
                except:
                    print("whats going on")
        if not arbiter.shapes == data['old_shapes']:
            print("still didnt find the error")
        new_bodies = {i: shp.body for i, shp in enumerate(arbiter.shapes)}
        if not new_bodies == data['old_bodies']:
            print("nononono")
        return True



def _bouncer_bouncer_pre_handler_no_torque(arbiter, space, data):
        """ Modified, from: https://github.com/AI-ON/TheConsciousnessPrior/blob/master/src/environments/billiards.py
            Custom collision, where conversion between momentum and angular momentum does not happen.
        """
        if not arbiter.is_first_contact:
            return True #_bouncer_bouncer_pre_handler_remove_angular(arbiter, space, data)
        set_ = arbiter.contact_point_set
        if len(set_.points) > 0:
            # assert len(set_.points) == 1
            # n: The collision normal is the direction of compression in a collision.
            # For two circles, it's the difference vector between centers (draw it to see). For
            #  a corner and a segment (e.g., polygon side), it's the orthogonal to the segment.
            # For a corner and a circle, or a corner and a corner, it might be undefined. But let's
            #  hope pymunk takes care of this :)
            n = set_.normal
            b_1, b_2 = [shp.body for shp in arbiter.shapes]
            Ki = (b_1.velocity.dot(b_1.velocity) * b_1.mass + b_2.velocity.dot(b_2.velocity) * b_2.mass) / 2.
            Pi = (b_1.velocity * b_1.mass + b_2.velocity *b_2.mass)
            v_1 = n.dot(b_1.velocity)
            v_2 = n.dot(b_2.velocity)

            new_v_1, new_v_2 = new_speeds(
                b_1.mass, b_2.mass, v_1, v_2)
            b_1.velocity += n * (new_v_1 - v_1)
            b_2.velocity += n * (new_v_2 - v_2)
            Ke = (b_1.velocity.dot(b_1.velocity) * b_1.mass + b_2.velocity.dot(b_2.velocity) * b_2.mass) / 2.
            Pe = (b_1.velocity * b_1.mass + b_2.velocity * b_2.mass)
            assert abs(Ki - Ke) < 0.00001
            assert abs(Pi - Pe).length < 0.00001
            set_.points[0].distance = 0 # No?
        arbiter.contact_point_set = set_

        return False



def _wall_bouncer_pre_handler_no_torque(arbiter, space, data):
    ''' ! First shape: wall, second shape: bouncer.'''
    if not arbiter.is_first_contact:
        return True
    set_ = arbiter.contact_point_set
    if len(set_.points) > 0:
        bouncer = arbiter.shapes[1].body
        segment_shape = arbiter.shapes[0]
        bouncer.velocity = reflect_vector_at(bouncer.velocity, segment_shape)
        set_.points[0].distance = 0
    arbiter.contact_point_set = set_
    return False


# -------------------- pre / post solvers end
# -------------------------------------------


def is_within_walls(body, wall_shapes):
    ''' Uses the radius of each of body.shapes to calculate when it hit a wall'''
    assert type(wall_shapes) not in [set, dict], "Need an ordered list of wall segments for function is_within_walls()"

    # first, is it outside of walls?
    within_walls = True
    for shape in body.shapes:
        if hits_walls(shape, wall_shapes):
            continue # collisions will be handled by pymunk
        elif not is_inside_polygon(shape, wall_shapes):
            within_walls = False
            break
    return within_walls


def is_inside_polygon(shape, segment_shapes):
    ''':param shape: a pymunk shape object
       :param segment_shapes: a *list* of shapes
    Assumes segment_shapes are already ordered to give a connected polygon'''
    assert len(segment_shapes) > 0, "Not a valid set of segments for function is_inside_polygon()"
    assert type(segment_shapes)  not in [set, dict], "Need an ordered list or array for function is_inside_polygon()"
    #if type(segment_shapes) == set:
    #    segment_shapes = list(segment_shapes)   # try to convert to list; will throw error if this list does not have correct segment order (likely)
    # build a matplotlib Path object from segment shapes
    corners = []
    former = abs_position(segment_shapes[0].a, segment_shapes[0].body)
    former = (former.x, former.y)
    last = abs_position(segment_shapes[-1].b, segment_shapes[-1].body)
    last = (last.x, last.y)
    assert former == last, "Polygon not closed"
    for seg in segment_shapes:
        seg_glob = shapes_abs_position(seg)   # ~-> glob = [(a.x, a.y),(b.x, b.y)]
        assert (seg_glob[0].x, seg_glob[0].y) == former, "Segments are not connected"
        corners.append((seg_glob[0].x, seg_glob[0].y))
        former = (seg_glob[1].x, seg_glob[1].y)
    corners.append(last)

    codes = [mplPath.Path.MOVETO]
    for i in range(len(corners) - 2):
        codes.append(mplPath.Path.LINETO)
    codes.append(mplPath.Path.CLOSEPOLY)

    bbPath = mplPath.Path(corners, codes)

    if type(shape) == pymunk.Circle:
        point = abs_position(shape.offset, shape.body)#shape.body.position + shape.offset.rotated(shape.body.angle)
    else:
        point = shape.body.position
    result = bbPath.contains_point((point.x, point.y))
    if not result:
        print("ohnooo")
    return bbPath.contains_point((point.x, point.y))


def hits_walls(shape, wall_shapes):
    hits = False
    for ws in wall_shapes:
        if shape.segment_query(ws.a, ws.b).shape is not None:
            hits = True
            break
    return hits


#def limit_velocity(list_of_bodies, limit):
#    for body in list_of_bodies:
#        if body.velocity.length > limit:
#            body.velocity = body.velocity / body.velocity.length
#    return



# ----- physics:

def new_speeds(m1, m2, v1, v2):
    ''' copied from https://github.com/AI-ON/TheConsciousnessPrior/blob/master/src/environments/billiards.py
        (See https://en.wikipedia.org/wiki/Elastic_collision#One-dimensional_Newtonian)'''
    new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2




# ----- drawing:


class CustomDrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface):
        """Draw a pymunk.Space on a pygame.Surface object.
        Don't draw collision points, don't draw circle orientation.
        See DrawOptions class in pymunk.

        :Parameters:
                surface : pygame.Surface
                    Surface that the objects will be drawn on
        """
        from pymunk.space_debug_draw_options import SpaceDebugDrawOptions as sddo
        self.surface = surface
        super(CustomDrawOptions, self).__init__()
        self.flags = sddo.DRAW_SHAPES | \
                     sddo.DRAW_CONSTRAINTS
                     #sddo.DRAW_COLLISION_POINTS

    def draw_circle(self, pos, angle, radius, outline_color, fill_color):
        p = ppu.to_pygame(pos, self.surface)

        pygame.draw.circle(self.surface, fill_color, p, int(radius), 0)

        #circle_edge = pos + pymunk.Vec2d(radius, 0).rotated(angle)
        #p2 = ppu.to_pygame(circle_edge, self.surface)
        #line_r = 2 if radius > 20 else 1
        #pygame.draw.lines(self.surface, outline_color, False, [p, p2], line_r)

    def draw_segment(self, a, b, color):
        p1 = ppu.to_pygame(a, self.surface)
        p2 = ppu.to_pygame(b, self.surface)

        pygame.draw.aalines(self.surface, color, False, [p1, p2])

    def draw_fat_segment(self, a, b, radius, outline_color, fill_color):
        p1 = ppu.to_pygame(a, self.surface)
        p2 = ppu.to_pygame(b, self.surface)

        r = int(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color, False, [p1, p2], r)

    def draw_polygon(self, verts, radius, outline_color, fill_color):
        ps = [ppu.to_pygame(v, self.surface) for v in verts]
        ps += [ps[0]]

        pygame.draw.polygon(self.surface, fill_color, ps)

        if radius < 1 and False:
            pygame.draw.lines(self.surface, outline_color, False, ps)
        else:
            pygame.draw.lines(self.surface, outline_color, False, ps, int(radius * 2))

    def draw_dot(self, size, pos, color):
        p = ppu.to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color, p, int(size), 0)


# -------------------------

def draw_shape(surface, shape, position, pymunk_draw_options):
    assert isinstance(shape, pymunk.Shape)
    pymunk_draw_options = copy.copy(pymunk_draw_options)
    pymunk_draw_options.surface = surface
    assert pymunk_draw_options.surface == surface
    tp = type(shape)
    if tp == pymunk.Circle:
        assert shape.radius > 0
        pymunk_draw_options.draw_circle(position, 0., shape.radius, shape.color, shape.color) # do these shapes have different outline_color and inner color?
    elif tp == pymunk.Poly:
        verts = [vert + position for vert in shape.get_vertices()]
        pymunk_draw_options.draw_polygon(verts, shape.radius, shape.color, shape.color)  #
    elif tp == pymunk.Segment:
        if shape.radius > 0:
            pymunk_draw_options.draw_fat_segment(shape.a + position, shape.b + position, shape.radius, shape.color, shape.color)
        else:
            pymunk_draw_options.draw_segment(shape.a + position, shape.b + position, shape.color)
    else:
        raise NotImplementedError("draw_shape() not yet implemented for shape type "+str(tp)+".")
