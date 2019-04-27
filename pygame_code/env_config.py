from enum import Enum
import pygame_code.bouncers as bc
import numpy as np

#class Shape(Enum):
#     CIRCLE = 1
#     SQUARE = 2
#     HEXAGON = 3



cfg = {}
env_cfg = {}
cfg['mode'] = 'static' # one of ['manual', 'random', 'static']
env_cfg['environment'] = 'simple' # one of ['moving_walls', 'simple']
env_cfg['num_bouncers'] = 1
env_cfg['bouncer_shapes'] = bc.shape_types # ['circle'] # bc.shape_types  #['circle', ...]
env_cfg['mass_propto_area'] = True
if not env_cfg['mass_propto_area']:
    env_cfg['mass_range'] = [1., 10.]
env_cfg['size_range'] = [12., 20.]
env_cfg['max_velocity'] = 100. # for all objects and the player
env_cfg['fps'] = 30.
env_cfg['step_dt'] = 1. / env_cfg['fps']
env_cfg['min_wall_width'] = env_cfg['step_dt'] * env_cfg['max_velocity'] * 1.01
env_cfg['max_init_angular_velocity'] = 6. # in radians per second
env_cfg['max_angular_velocity'] = 20. # in radians per second
env_cfg['allow_rotation'] = False
# Forbidding rotation does not work well with rectangular shapes (when hitting a wall, probably,
#  most of the momentum is transfered to rotational velocity first. So if this is canceled out,
#  the boxes lose most of their momentum.
env_cfg['ball_hits_ball'] = True
env_cfg['velocities_in_angles'] = False  # can't change this so far
env_cfg['clutter_background'] = True # if False, don't clutter at all
env_cfg['clutter_background_white'] = True # white: most confusing (the balls are white, too)
if env_cfg['clutter_background']:
    env_cfg['clutter_sz_min'] = 10.
    env_cfg['clutter_sz_max'] = 25. if env_cfg['clutter_background_white'] else 30.
    env_cfg['clutter_num']    = 10  if env_cfg['clutter_background_white'] else 20  # the number of clutter-objects to draw on background
        # white is much more confusing; with lots ofgray objects, can easily cover the whole screen. Not so with white clutter.
env_cfg['show_screen'] = False
env_cfg['show_screen_step'] = 1 # might be changed below
if env_cfg['environment'] == 'moving_walls':
    env_cfg['box_deceleration'] = 0.7 # a factor
    env_cfg['box_acceleration'] = 40. # absolute value
    env_cfg['outer_box_height'] = 160
    env_cfg['outer_box_width'] = 160
    env_cfg['inner_box_height'] = 80
    env_cfg['inner_box_width'] = 80
    assert env_cfg['outer_box_width'] >= env_cfg['inner_box_width']
    assert env_cfg['outer_box_height'] >= env_cfg['inner_box_height']
elif env_cfg['environment'] == 'simple':
    # moving MNIST: 64 x 64 (MNIST: 28x28) -- add a few pixels because we have a wide wall
    # CIFAR:  32x32 colour images
    env_cfg['outer_box_height'] = 80
    env_cfg['outer_box_width'] = 80
#env_cfg['']
#env_cfg['']


cfg['seq_length'] = 20 # how many steps until reset
cfg['store_frames'] = True # False is for debugging
if cfg['store_frames']:
    cfg['sampling_step'] = 1  # don't store every frame, only every nth
    env_cfg['show_screen_step'] = cfg['sampling_step']
    cfg['num_sequences_train'] =  1600
    cfg['num_sequences_valid'] = 800
    cfg['num_sequences_test'] = 800
    cfg['train_first_file_number'] = 1
    cfg['valid_first_file_number'] = 1
    cfg['test_first_file_number'] = 1
    if env_cfg['bouncer_shapes'] == ['circle']:
        cfg['output_dir'] = "../../data/gen/bouncing_circles/short_sequences/" # where frames are stored, if any - ?store as images, or as num_seqs x seq_length x (h x w x d)?
    else: # objects, not just circles
        cfg['output_dir'] = "../../data/gen/bouncing_objects/short_sequences/"
    cfg['store_as_rgb'] = False
cfg['random_seed'] = 31 #np.random.randint(0,10000)
if cfg['mode'] == 'random':
    cfg['action_change_interval_min'] = 1
    cfg['action_change_interval_max'] = 3
cfg['output_dir'] += cfg['mode'] + "_" +  env_cfg['environment']
if env_cfg['clutter_background']:
    cfg['output_dir'] += '_cluttered'
    if env_cfg['clutter_background_white']:
        cfg['output_dir'] += '-white'
cfg['output_dir'] +=  "_"+str(env_cfg['num_bouncers'])+"_bcs/"

if env_cfg['environment'] == 'simple':
    assert cfg['mode'] == 'static', "It's a simple environment, what'cha expect"