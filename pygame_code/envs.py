import numpy as np
from gym import spaces
import pygame
import pygame.locals as ploc
from pygame.color import THECOLORS
import pymunk, pymunk.pygame_util
assert pymunk.version >= '5.0.0', "This is the version for pymunk v. 5.3.x, run the code in folder 'pygame_code_v4_0_0' with pymunk 4.0.0."
from enum import Enum
from collections import OrderedDict

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pygame_code.pymunk_utils as pymunk_utils
import src.utils.utils as utils
import pygame_code.bouncers as bc
from src.utils.decorators import *
import matplotlib.pyplot as plt

class AbstractEnv:

    def __init__(self):
        #self.input_space = None
        self.state_space = None         # gym.spaces.Dict
        self.observation_space = None
        self.action_space = None

        self.bouncer_objects = []
        self.player_body = None
        self.space = None
        self.limit_velocity_func = None  # fun of (body, gravity, sthelse, dt) that can set the body's velocity

        # TODO: Reinsert
        #for sp in [self.state_space, self.observation_space, self.action_space]:
        #    assert isinstance(sp, spaces.Box) or isinstance(sp, spaces.Discrete) or isinstance(sp, spaces.Dict)

    def start(self):
        raise NotImplementedError

    def step(self, action=None):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_observation(self):
        '''return a np array of what's currently visible in the env'''
        raise NotImplementedError

    @property
    def outer_wall_shapes(self):
        raise NotImplementedError

    @property
    def outer_walls(self):
        raise NotImplementedError

class CollisionTypes(Enum):
        STATIC = 0
        MOVING_WALLS = 1
        BOUNCERS = 2


class BouncingBallEnv(AbstractEnv):
    '''
    Keeping track of collision types:
    0: the static outer walls
    1: the movable inner walls
    2: bouncing objects
    '''

    def __init__(self, env_cfg, moving_walls=True):
        super().__init__()

        self.moving_walls = moving_walls

        self.space = pymunk.Space()
        self.space.iterations = 10 #30

        # -- copy configuration for shortness later -- #
        self.cfg = env_cfg
        self.n_bouncers = env_cfg['num_bouncers']
        if self.moving_walls:
            self.box_decel  = env_cfg['box_deceleration']
            self.box_accel  = env_cfg['box_acceleration']
        self.mass_propto_area = env_cfg['mass_propto_area']
        if not self.mass_propto_area:
            self.m_min,  self.m_max  = env_cfg['mass_range']
        self.sz_min, self.sz_max = env_cfg['size_range']
        self.min_wall_width = 2 * env_cfg['min_wall_width']
        self.v0_max = env_cfg['max_velocity']
        self.w0_max = env_cfg['max_init_angular_velocity']
        self.max_velocity = env_cfg['max_velocity']
        self.max_angular_velocity = env_cfg['max_angular_velocity']
        self.screen_size = [env_cfg['outer_box_width'], env_cfg['outer_box_height']]
        self.allow_rotation = env_cfg['allow_rotation']
        self.ball_hits_ball = env_cfg['ball_hits_ball']
        self.velocities_in_angles = env_cfg['velocities_in_angles'] # unused
        self.clutter_background = env_cfg['clutter_background']
        self.clutter_background_white = env_cfg['clutter_background_white']
        self.clutter_sz_min = env_cfg['clutter_sz_min']
        self.clutter_sz_max = env_cfg['clutter_sz_max']
        self.clutter_num = env_cfg['clutter_num']
        self.show_screen = env_cfg['show_screen']
        self.show_screen_step = env_cfg['show_screen_step']

        # --- derived quantities --- #
        if self.moving_walls:
            self.n_objects = self.n_bouncers + 1 # inner walls' position: lower left corner
            self.n_iwalls = 4 # inner walls
        else:
            self.n_objects = self.n_bouncers
            self.n_iwalls = 0
        self.n_owalls = 4 # outer walls
        self.n_shapes = self.n_iwalls + self.n_owalls + self.n_bouncers

        def limit_velocity(body, gravity, damping, dt):
            v = body.velocity
            if v.length != 0:
                v.length = min(self.max_velocity, v.length)
            body.velocity = v
            if not self.allow_rotation:
                body.angular_velocity = 0
            else:
                body.angular_velocity = min(self.max_angular_velocity, body.angular_velocity)
                body.angular_velocity = max(- self.max_angular_velocity, body.angular_velocity)
        self.limit_velocity_func = limit_velocity

        # RGB frames
        self.observation_space = spaces.Box(low=0, high=255, shape=(env_cfg['outer_box_width'], env_cfg['outer_box_height'], 3), dtype=np.uint8)
        # All "measurements" combined
        self.create_state_space()
        if self.moving_walls:
            self.action_space = spaces.MultiBinary(4)  # for each arrow key, whether it's pressed or not

        # Building the environment
        self.space.gravity = pymunk.Vec2d(0., 0.)
        # outer walls
        create_walls(self, env_cfg['outer_box_height'], env_cfg['outer_box_width'], kinematic=False,
                     color=THECOLORS['burlywood1'], center=False) # ... burlywood?

        self.player_body = None
        self.bouncer_objects = []

        #h = self.space.add_collision_handler(CollisionTypes.MOVING_WALLS.value, CollisionTypes.BOUNCERS.value)
        if not self.allow_rotation:
            h = self.space.add_collision_handler(CollisionTypes.BOUNCERS.value, CollisionTypes.BOUNCERS.value)
            hiw = self.space.add_collision_handler(CollisionTypes.MOVING_WALLS.value, CollisionTypes.BOUNCERS.value)
            how = self.space.add_collision_handler(CollisionTypes.STATIC.value,       CollisionTypes.BOUNCERS.value)
                # for debugging:
            #h.begin = pymunk_utils.begin_check_conservation_laws
            #h.separate = pymunk_utils.separate_check_conservation_laws
                # for balls only, can switch off rotation changes at collisions
                #   by setting angular velocity to zero during collision:
            #if self.cfg['bouncer_shapes'] == ['circle']:
            #    h.pre_solve    = pymunk_utils._bouncer_bouncer_pre_handler_remove_angular
            #    h.post_solve   = pymunk_utils._bouncer_bouncer_post_handler_remove_angular
            #    hiw.pre_solve  = pymunk_utils._bouncer_bouncer_pre_handler_remove_angular
            #    hiw.post_solve = pymunk_utils._bouncer_bouncer_post_handler_remove_angular
            #    how.pre_solve  = pymunk_utils._bouncer_bouncer_pre_handler_remove_angular
            #    how.post_solve = pymunk_utils._bouncer_bouncer_post_handler_remove_angular
            # otherwise, need makeshift solution:
            #else:
            h.pre_solve   = pymunk_utils._bouncer_bouncer_pre_handler_no_torque
            hiw.pre_solve = pymunk_utils._bouncer_bouncer_pre_handler_no_torque
            how.pre_solve = pymunk_utils._wall_bouncer_pre_handler_no_torque


        if not self.ball_hits_ball:
            h = self.space.add_collision_handler(CollisionTypes.BOUNCERS.value, CollisionTypes.BOUNCERS.value)
            h.pre_solve = lambda arbiter, space, data: False

        # Final setup
        self.fps = env_cfg['fps']
        self.dt = env_cfg['step_dt']
        self.screen = None
        self.clock = None
        #self.keysdown = {ploc.K_DOWN: False, ploc.K_UP: False, ploc.K_LEFT: False, ploc.K_RIGHT: False}

        self.n_steps = 0
        self.running = False
        self.observation = None
        if self.clutter_background:
            self.background_surf = None
        # ,                       'observation': self.observation})


    def empty(self):
        ''' Removes all bodies except the static body from the environment
            Resets self.player_body, self.bouncer_objects'''
        self.player_body = None
        self.bouncer_objects = []
        bodies = self.space.bodies
        # check for duplicates
        assert len([x for x in bodies if bodies.count(x) > 1]) == 0, "space.bodies contained duplicates - is that bad? " \
                                                                     "Will space.remove() remove all duplicates also?"
        bodies = set(bodies)
        bodies -= set([self.space.static_body])  # looks like the static body is not contained in .bodies anyway
        for b in bodies:
            self.space.remove(b.shapes)
            self.space.remove(b.constraints)
            self.space.remove(b)
        for shp in self.space.shapes:
            assert shp in self.space.static_body.shapes, "Error: Found a shape in the environment after call to empty()"
        for cons in self.space.constraints:
            assert cons in self.space.static_body.constraints, "Wah, random constraint appeared"


    def fill(self):
        ''' fill space with objects'''
        assert self.player_body is None and len(self.bouncer_objects) == 0, "Do you really want to call fill() on already filled environment?"
        if self.moving_walls:
            # create moving walls
            self.player_body = create_walls(self, self.cfg['inner_box_height'], self.cfg['inner_box_width'],
                                            kinematic=True, color=THECOLORS['gray10'], center=True)
        # --  Set random bouncers
        self.bouncer_objects = []
        # First, get bounding box to place bouncers in
        walls = self.enclosing_wall_shapes
        assert len(walls)==4, "Placing objects currently only works for inside a box. Need to add stuff if you want non-box outer walls."
        corners = np.array([pymunk_utils.shapes_abs_position(w) for w in walls])
        bb = pymunk.BB(np.min(corners[:,:,0]), np.min(corners[:,:,1]),
                       np.max(corners[:,:,0]), np.max(corners[:,:,1]))
        #if moving_walls:
        #    bb = pymunk.BB(self.player_body.position[0], self.player_body.position[1], self.player_body.position[0] + self.cfg['inner_box_width'],
        #               self.player_body.position[1] + self.cfg['inner_box_height'])

        for _ in range(self.n_bouncers):
            bouncer = create_bouncer(self, shape_types=self.cfg['bouncer_shapes'], bb=bb)
            self.bouncer_objects.append(bouncer)
        if not self.allow_rotation:
            for bd in self.space.bodies:
                bd.angular_velocity = 0.
                bd.angular_velocity_limit = 0.
        self.count_shapes()  # contains assertion
        if self.clutter_background and self.clutter_background_white:
            self.background_surf = generate_cluttered_background(['circle', 'rectangle', 'triangle'], self.screen_size[0],
                                                self.screen_size[1],self.draw_options, sz_min=self.clutter_sz_min, sz_max=self.clutter_sz_max,
                                                            same_sided=False, n_objects=self.clutter_num, colors=[THECOLORS['white']])
        elif self.clutter_background:
            self.background_surf = generate_cluttered_background(['circle', 'rectangle', 'triangle'], self.screen_size[0],
                            self.screen_size[1], self.draw_options, sz_min=self.clutter_sz_min, sz_max=self.clutter_sz_max,
                                                                 same_sided=False, n_objects=self.clutter_num)


    def reset(self):
        '''call either reset or start'''
        self.empty()
        self.fill()
        self.clock = pygame.time.Clock()
        self.running = True
        self.get_state()
        self.n_steps = 0
        return self.state, self.observation

    def start(self):
        ''' call either reset() or start() - start(), the first time, later reset()'''
        # mysterious "pygame module initialization" function
        pygame.init()
        if self.cfg['show_screen']:
            self.screen = pygame.display.set_mode((self.cfg['outer_box_width'], self.cfg['outer_box_height']))
        else:
            self.screen = pygame.Surface((self.cfg['outer_box_width'], self.cfg['outer_box_height']))
        #self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options = pymunk_utils.CustomDrawOptions(self.screen)

        state, observation = self.reset()
        return state, observation

    def apply_action(self, action=None):
        assert self.moving_walls
        up, down, left, right = [0, 1, 2, 3]
        if action is None:
            action = [0, 0, 0, 0]  # no keys currently pressed
        elif (action == [-1, -1, -1, -1]).all():
            state, observation = self.get_state()
            self.running = False
            return state, observation
        assert self.running and len(action) == 4
        # Update velocity
        self.player_body.velocity *= self.box_decel
        if 1 in action:
            dv_vert =  - self.box_accel * (action[down]) + self.box_accel * (action[up])
            dv_hor = - self.box_accel * (action[left]) + self.box_accel * (action[right])
            self.player_body.velocity += (dv_hor, dv_vert)


    # --- step --- #
    def step(self, action=None):
        ''' actions: [key_up: 1 / 0, key_down: 1/0, key_left: 1/0, key_right: 1/0]
                     all '-1's means stop '''
        if self.moving_walls:
            self.apply_action(action)

        self.space.step(self.dt)
        self.clock.tick(self.fps)

        # stop env if a bouncer escaped
        try:
            for bouncer in self.bouncer_objects:
                b = bouncer.body
                assert pymunk_utils.is_within_walls(b, self.enclosing_wall_shapes)
        except AssertionError:
            print("Error, bouncer escaped the box. Set number of iterations of solver up, or max velocity down.")
            self.running = False
            return

        ### Clear screen
        self.screen.fill(THECOLORS["black"])
        if self.clutter_background:
            self.screen.blit(source=self.background_surf, dest=(0, 0))
        ### Draw stuff
        self.space.debug_draw(self.draw_options)
        #pymunk.pygame_util.draw(self.screen, self.space)

        if self.cfg['show_screen'] and self.n_steps % self.show_screen_step == 0:
            pygame.display.flip()
        state, observation = self.get_state()
## switch on for showing locations and velocities
#        obs_dbg = observation.copy()
#        plt.imshow(obs_dbg, cmap='gray')
#        pos, vel = (state['positions'], state['velocities'])
#        dt = 0.2
#        for v, p in zip(vel, pos):
#                    plt.scatter(x=[p[1]], y=[p[0]])#, c=cols[i], s=40)
#                    plt.arrow(x=p[1], y=p[0], dx=v[1]*dt, dy=v[0]*dt)
#                    plt.pause(0.001)

        self.n_steps += 1
        return state, observation
        # --- step-end --- #


    def get_observation(self):
        '''return a np array of what's currently visible in the env'''
        raise NotImplementedError

    @property
    def outer_wall_shapes(self):
        assert self.space is not None, "space not initialized"
        return self.space.static_body.shapes_ordered
        #return self.space.static_body.shapes
    @property
    def outer_walls(self):
        assert self.space is not None, "space not initialized"
        return self.space.static_body

    @property
    def enclosing_wall_shapes(self):
        ''' return inner wall shapes if inner wall exists (if moving_walls), else
            outer wall shapes'''
        if self.moving_walls:
            return self.player_body.shapes_ordered
        else:
            return self.outer_wall_shapes

    def count_shapes(self):
        assert self.n_shapes == len(self.space.shapes)
        return len(self.space.shapes)


    def get_state(self):
        ''' '''
        body_types = np.zeros(self.state_space.spaces['body_types'].shape, dtype='|S10')
        shape_body_ids = np.zeros(self.state_space.spaces['shape_body_ids'].shape, dtype=int)
        bouncer_sizes = np.zeros(self.state_space.spaces['bouncer_sizes'].shape)
        bouncer_weight = np.zeros(self.state_space.spaces['bouncer_weight'].shape)
        vels = np.zeros(self.state_space.spaces['velocities'].shape)
        poss = np.zeros(self.state_space.spaces['positions'].shape)
        angles = np.zeros(self.state_space.spaces['angles'].shape)
        ang_vels = np.zeros(self.state_space.spaces['angular_velocities'].shape)
        iw_col_id = np.zeros(self.state_space.spaces['next_wall_to_hit_id'].shape, dtype=int)
        iw_col_point = np.zeros(self.state_space.spaces['next_wall_to_hit_where'].shape)
        iw_col_dt = np.zeros(self.state_space.spaces['next_wall_to_hit_when'].shape)
        rel_dists = np.zeros(self.state_space.spaces['relative_position_matrix'].shape)
        # Can extend this with more features
        for i, x in enumerate(self.space.shapes):
            vels[i, 1] = - x.body.velocity.int_tuple[1]  # [x.body.velocity.x, x.body.velocity.y]
            vels[i, 0] = x.body.velocity.int_tuple[0]  # [x.body.velocity.x, x.body.velocity.y]
            poss[i, 1] = self.screen_size[1] - x.body.position.int_tuple[1]  # [x.body.position.x, x.body.position.y]
            poss[i, 0] = x.body.position.int_tuple[0]  # [x.body.position.x, x.body.position.y]
            angles[i] = x.body.angle
            ang_vels[i, :] = x.body.angular_velocity
            for j, y in enumerate(self.space.shapes):  # is it a list?
                if i == j:
                    continue
                if y.body in [self.player_body, self.space.static_body]:
                    if x.body in [self.player_body, self.space.static_body]:
                        dist_vec = (y.b + y.a) / 2. - (x.b + x.a) / 2.
                        rel_dists[i, j, :2] = dist_vec.int_tuple
                        rel_dists[i, j, 2:] = dist_vec.length
                    else:
                        # line-object distance; in this case it doesn't matter that dist to the line, not the segment is calculated - they will always be the same quantity here
                        dist, d = pymunk_utils.line_object_distance(y, x)
                        rel_dists[i, j, :2] = dist
                        rel_dists[i, j, 2:] = d
                else:
                    if x.body in [self.player_body, self.space.static_body]:
                        dist, d = pymunk_utils.line_object_distance(x, y)
                        rel_dists[i, j, :2] = dist
                        rel_dists[i, j, 2:] = d
                    else:
                        rel_dists[i, j, :2] = (y.body.position - x.body.position).int_tuple
                        rel_dists[i, j, 2:] = (y.body.position - x.body.position).length - y.radius - x.radius
                if i > j:
                    if isinstance(x, pymunk.Segment) != isinstance(y, pymunk.Segment):
                        assert (rel_dists[j, i, :2] == rel_dists[i, j, :2]).all()
                    else:
                        assert (rel_dists[j, i, :2] == -rel_dists[i, j, :2]).all()
        # now for the distance in movement direction
        for i, b in enumerate(self.bouncer_objects):
            x = b.body
            assert len(x.shapes) == 1
            # get times to hit any of the walls)
            wall_dists = np.zeros((4, 3))
            # walls to calc difference to:
            #   the closest ones (-> inner walls if they exist)
            wall_shapes = self.player_body.shapes_ordered if self.moving_walls else self.space.static_body.shapes_ordered
            for j, y in enumerate(wall_shapes):
                dist_v, d, delta_t = pymunk_utils.line_object_collision_time(y, list(x.shapes)[0])
                wall_dists[j, :] = [dist_v[0], dist_v[1], delta_t]
            # get argmin
            iw_col_id[i] = np.argmin(wall_dists[:, 2])
            iw_col_dt[i] = wall_dists[iw_col_id[i], 2]
            if iw_col_dt[i] == np.inf:  # ball moves at same speed as the walls, right now
                iw_col_id[i] = -1
                iw_col_dt[i] = -1 # invalid
                iw_col_point[i] = [-1,-1]
            else:
                # get collision point
                direct_dist_vec = wall_dists[iw_col_id[i], :2]
                if np.linalg.norm(direct_dist_vec) == 0:
                    iw_col_point[i] = x.position.int_tuple
                else:
                    iw_col_point[i] = x.position.int_tuple + x.velocity.int_tuple * iw_col_dt[i] + direct_dist_vec / np.linalg.norm(
                    direct_dist_vec) * list(x.shapes)[0].radius

        for j, y in enumerate(self.space.shapes):
            shape_body_ids[j] = utils.where(y.body, self.space.bodies)
        for j, b in enumerate(self.space.bodies):
            body_types[j] = get_obj_type(self, b).name
        # sort bouncer objects into same order as their bodies appears in self.space.bodies
        # (later: order of their main bodies -- then, todo: need to add another field to the state saying which body belongs to which bouncer)
        bouncers = [(utils.where(bo.body, self.space.bodies), bo) for bo in self.bouncer_objects]
        bouncers = sorted(bouncers)
        for i, (j, bo) in enumerate(bouncers):
            bouncer_sizes[i] = get_bouncer_size(bo)
        for i, (j, bo) in enumerate(bouncers):
            bouncer_weight[i] = bo.mass

        self.state['shape_body_ids'] = shape_body_ids
        self.state['body_types'] = body_types
        self.state['bouncer_sizes'] = bouncer_sizes
        self.state['bouncer_weight'] = bouncer_weight
        self.state['velocities'] = vels
        self.state['positions'] = poss
        self.state['angles'] = angles
        self.state['angular_velocities'] = ang_vels
        self.state['next_wall_to_hit_id'] = iw_col_id
        self.state['next_wall_to_hit_where'] = iw_col_point
        self.state['next_wall_to_hit_when'] = iw_col_dt
        self.state['relative_position_matrix'] = rel_dists
        for key in self.state.keys():
            try:
                utils.check_validity(self.state[key], key)
            except:
                raise
        self.observation = pygame.surfarray.array3d(
            self.screen)  # https://www.pygame.org/docs/ref/surfarray.html#pygame.surfarray.array2d
        # self.state['observation'] = self.observation

        return self.state.copy(), self.observation





    def create_state_space(self):
        ''' needs:
                self.n_shapes -- self.n_bouncers '''
        if self.moving_walls:
            n_objects = self.n_bouncers + 1 # plus walls
        else:
            n_objects = self.n_bouncers
        state_space = spaces.Dict({
            # a vector containing the index of every shapes body in self.space.bodies; -1 for static
            "shape_body_ids": spaces.MultiDiscrete([self.n_shapes] * self.n_shapes),
        # sort of header, the index of every shapes' body in the list
            # a list of ? strings giving the type of each of the bodies in the list (excluding the static body)
            "body_types": spaces.MultiDiscrete([len(bc.ObjectTypes)] * (n_objects)),
            "bouncer_sizes": spaces.Box(low=-pymunk.inf, high=pymunk.inf, shape=(self.n_bouncers, 2), dtype=np.float32),
            "bouncer_weight": spaces.Box(low=-pymunk.inf, high=pymunk.inf, shape=(self.n_bouncers,), dtype=np.float32),
            "positions": spaces.Box(low=-pymunk.inf, high=pymunk.inf, shape=(self.n_shapes, 2), dtype=np.float32),
            "velocities": spaces.Box(low=-pymunk.inf, high=pymunk.inf, shape=(self.n_shapes, 2), dtype=np.float32),
            "angles": spaces.Box(low=0., high=2. * np.pi, shape=(self.n_shapes, 1), dtype=np.float32),
            "angular_velocities": spaces.Box(low=-pymunk.inf, high=pymunk.inf, shape=(self.n_shapes, 1), dtype=np.float32),
            # 3 coordinates: dx, dy - the distance between the centers of the objects, and the
            #  third coordinate contains the absolute distance ( ==  norm((dx,dy)) - o1.radius - o2.radius)
            "relative_position_matrix": spaces.Box(low=-pymunk.inf, high=pymunk.inf,
                                                   shape=(self.n_shapes, self.n_shapes, 3), dtype=np.float32),
            # estimate; maybe a non-wall collision comes first
            "next_wall_to_hit_id": spaces.MultiDiscrete([4] * self.n_bouncers),
            "next_wall_to_hit_where": spaces.Box(low=-pymunk.inf, high=pymunk.inf, shape=(self.n_bouncers, 2),
                                                 dtype=np.float32),
            "next_wall_to_hit_when": spaces.Box(low=-pymunk.inf, high=pymunk.inf, shape=(self.n_bouncers, 1),
                                                dtype=np.float32)})
            # 1's for every pair that just started to collide this round
            # "collision_matrix":         spaces.Discrete(2, shape=(self.n_bouncers + self.n_iwalls, self.n_bouncers + self.n_iwalls)),
            # "observation":              self.observation_space})
        self.state_space = state_space
        self.state =  OrderedDict({'shape_body_ids': None, 'body_types': None, 'bouncer_sizes': None,
                                   'bouncer_weight':None, 'positions': None, 'velocities': None, 'angles': None,
                                   'angular_velocities': None, 'relative_position_matrix': None,
                                   'next_wall_to_hit_id': None, 'next_wall_to_hit_where': None,
                                   'next_wall_to_hit_when': None })

        return self.state_space, self.state



def get_obj_type(env, body):
    ''' return an element of enum bouncers.ObjectTypes'''
    assert isinstance(body, pymunk.Body)
    assert isinstance(env, AbstractEnv)
    all_bouncer_bodies = [b for bo in env.bouncer_objects for b in bo.bodies]
    all_bouncers = [bo for bo in env.bouncer_objects for _ in bo.bodies]
    if body == env.player_body:
        assert env.player_body is not None
        return bc.ObjectTypes.inner_wall
    elif body == env.space.static_body:
        return bc.ObjectTypes.outer_wall
    elif body in all_bouncer_bodies:
        i = utils.where(body, all_bouncer_bodies)
        bouncer = all_bouncers[i]
        return bouncer.bouncer_type
    else:
        raise ValueError("Body type not recognized. Body: "+str(body))

def get_bouncer_size(bouncer):
    ''' Always return a list of two floats, "width" and "height". They
        are the same (== its radius) for circles.'''
    assert isinstance(bouncer, bc.AbstractBouncer)
    if type(bouncer) == bc.CircleBouncer:
        return [bouncer.radius * 2.] * 2
    elif type(bouncer) == bc.RectangleBouncer:
        return [bouncer.width, bouncer.height]
    elif type(bouncer) == bc.TriangleBouncer:
        return [bouncer.width, bouncer.height]
    else:
        raise ValueError("get_bouncer_size() not implemented for bouncer type "+str(type(bouncer)))

def attached_bouncer(bouncer_list, body):
    ''' return the bouncer in bouncer_list that this body is attached to.
        Throws if body doesnt belong to any of the bouncer objects'''
    assert isinstance(body, pymunk.Body)
    all_bouncer_bodies = [b for bo in bouncer_list for b in bo.bodies]
    all_bouncers = [bo for bo in bouncer_list for _ in bo.bodies]
    assert body in all_bouncer_bodies
    i = utils.where(body, all_bouncer_bodies)
    return all_bouncers[i]



def create_walls(env, height, width, kinematic=False, color=THECOLORS['gray10'], center=False):

        if kinematic:
            walls_body = pymunk.Body(mass=1000, moment=pymunk.inf) # if not both mass and moment == 0, automatically create ..either a dynamic or a kinematic body
        else:                                                     # seems they are the same ...?
            walls_body = env.space.static_body
        minw = env.min_wall_width
        if center:
            distx, disty = (np.array(env.screen_size) - np.array([width, height])) / 2.
            walls_body.position = pymunk.Vec2d((distx, disty))
            #walls = [
            #        pymunk.Segment( walls_body, (distx, disty),            (distx, disty + height),    minw),
            #        pymunk.Segment( walls_body, (distx, height + disty),       (width + disty, height + disty),minw),
            #        pymunk.Segment( walls_body, (width + disty, height + disty),   (width + disty, disty),     minw),
            #        pymunk.Segment( walls_body, (width + disty, disty),        (distx, disty),         minw)
            #    ]
        #else:
        walls = [
                pymunk.Segment(walls_body, (0, 0), (0, height), minw),
                pymunk.Segment(walls_body, (0, height), (width, height), minw),
                pymunk.Segment(walls_body, (width, height), (width, 0), minw),
                pymunk.Segment(walls_body, (width, 0), (0, 0), minw)
            ]
        for wall in walls:
            wall.friction = 1.
            wall.group = 1 if not kinematic else 2
            wall.collision_type = CollisionTypes.STATIC.value if not kinematic else CollisionTypes.MOVING_WALLS.value
            wall.color = color
            wall.elasticity = 1.
        walls_body.shapes_ordered = walls
        env.space.add(walls)
        if kinematic:
            env.space.add(walls_body)
            walls_body.velocity_func = env.limit_velocity_func
        return walls_body



def create_bouncer(env, shape_types=['circle'], bb=None, same_sided=False):
        assert isinstance(env, AbstractEnv)
        if bb is None:
            bb = pymunk.BB(0, 0, env.screen_size[0], env.screen_size[1])

        shape_type = shape_types[np.random.randint(0, len(shape_types))]
        vel     = tuple(utils.rand_float(-env.v0_max, env.v0_max, shape=(2,)))
        radius  = utils.rand_float(env.sz_min / 2., env.sz_max / 2., shape=())
        if env.mass_propto_area:
            mass = None
        else:
            mass = utils.rand_float(env.m_min, env.m_max, shape=())# np.random.random(size=1) * (env.m_max - env.m_min) + env.m_min
            assert mass <= env.m_max and mass >= env.m_min
        assert shape_type in bc.shape_types
        assert radius * 2 < min([bb.top - bb.bottom, bb.right - bb.left])

        if shape_type is 'circle':
            bouncer = bc.CircleBouncer(mass, vel, radius=radius, mass_propto_area=env.mass_propto_area)
        elif shape_type is 'rectangle':
            height = radius * 2
            if not same_sided:
                width = utils.rand_float(env.sz_min, env.sz_max, shape=())
            else:
                width = height
            assert width < min([bb.top - bb.bottom, bb.right - bb.left])
            bouncer = bc.RectangleBouncer(mass, vel, height=height, width=width, mass_propto_area=env.mass_propto_area)
        elif shape_type is 'triangle':
            height = radius * 2
            width = None if same_sided else utils.rand_float(env.sz_min, env.sz_max, shape=())
            bouncer = bc.TriangleBouncer(mass, vel, height=height, width=width, mass_propto_area=env.mass_propto_area)
        else:
            raise NotImplementedError("Shape type " + shape_type + " has not been implemented so far.")

        bouncer_body = bouncer.bodies[0]
        bouncer_shape = bouncer.shapes[bouncer_body][0]
        bouncer_shape.elasticity = 1.
        bouncer_shape.collision_type = CollisionTypes.BOUNCERS.value
        bouncer_body.velocity = vel
        bouncer_body.angle = utils.rand_float(0, 2. * np.pi)
        bouncer_body.angular_velocity = utils.rand_float(-env.w0_max, env.w0_max) if env.allow_rotation else 0.
        bouncer_body.velocity_func = env.limit_velocity_func
        # direction = pymunk.Vec2d(1, 0).rotated(bouncer_body.angle)
        ok = pymunk_utils.assign_valid_location(bouncer_body, env.space, bb, maxiter=100,
                                                radius=(bouncer.diameter/2. + env.min_wall_width/2.))
        if not ok:
            raise AssertionError("Could not create new body; space too crowded")

        env.space.add(bouncer_body, bouncer_shape)
        return bouncer


def generate_cluttered_background(shape_types, width, height, pymunk_draw_options, sz_min=10., sz_max=20., same_sided=False, n_objects=10,
                                  colors=[THECOLORS['gray100'], THECOLORS['gray17'], THECOLORS['gray12'], THECOLORS['gray39'], THECOLORS['gray85']]):
    #if "pygame_code.bouncers" not in sys.modules:
    #    import pygame_code.bouncers as pyb
    #surf = np.zeros((width, height), dtype=np.uint8)
    surf = pygame.Surface((width, height))
    surf.fill(THECOLORS["black"])
    for i in range(n_objects):
        bc.draw_random_bouncer(surf, pymunk_draw_options, shape_types=shape_types, sz_min=sz_min, sz_max=sz_max, same_sided=same_sided, colors=colors)

    #surface = pygame.surfarray.array2d(surf)
    return surf


def test_generate_cluttered_background():
    space = pymunk.Space()
    space.gravity = pymunk.Vec2d(0., 0.)
    pygame.init()
    screen = pygame.display.set_mode((300, 200))
    draw_options = pymunk_utils.CustomDrawOptions(screen)
    obs = pygame.surfarray.array3d(screen)

    for i in range(10):
        space.step(0.1)
        ### Clear screen
        screen.fill(THECOLORS["black"])
        colors = [THECOLORS['gray100'], THECOLORS['gray17'], THECOLORS['gray12'], THECOLORS['gray39'], THECOLORS['gray85']]
        cluttered_surf = generate_cluttered_background(['circle', 'rectangle', 'triangle'], 300, 200, draw_options, sz_min=10., sz_max=20.,
                                                       same_sided=False, n_objects=100, colors=colors)
        screen.blit(source=cluttered_surf, dest=(0,0))
        ### Draw stuff
        space.debug_draw(draw_options)
        pygame.display.flip()




# -----------------------------------------------------
# -----------------------------------------------------

def main_dbg(cfg, env_cfg):
    env = BouncingBallEnv(env_cfg=env_cfg, moving_walls=env_cfg['environment']=='moving_walls')
    env.start()
    while True:
        if not env.running:
            break
        state, obs = env.step()
        #print("stepped")



if __name__ == '__main__':
    from pygame_code.env_config import cfg, env_cfg
    main_dbg(cfg, env_cfg)

    #test_generate_cluttered_background()