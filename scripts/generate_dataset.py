import argparse
import importlib
import logging
import time
import numpy as np
import os
import sys
import gym
import pygame
import pygame.locals as ploc
from collections import OrderedDict


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.utils import export_config_txt, ensure_dir, set_logger, check_validity, RGB2gray_nd, RGB2bw_nd
from pygame_code.envs import BouncingBallEnv

# TODO -- update: mostly done: Write tests? Especially of state generation. Visualize?

MAX_NUM_SEQUENCES_PER_FILE = 400


def filepath(path, identifier_str, file_num):
    return os.path.join(path, identifier_str + "_" + str(file_num))


class AbstractActionGenerator():
    def __init__(self):
        self.state = None
    def action(self):
        raise NotImplementedError

class NoActionGenerator(AbstractActionGenerator):
    def __init__(self):
        self.state=None
    def action(self):
        return None

class RandomActionGenerator(AbstractActionGenerator):
    ''' throw a dice -> n, for n next steps, return randomly generated vector {0,1}^4'''
    def __init__(self, n_min=1, n_max=10 ):
        self.state = 0
        self.n_min = n_min
        self.n_max = n_max
        self.current_action = None
        assert n_min >0 and n_max >= n_min
    def action(self):
        if self.state == 0:
            self.state = np.random.random_integers(self.n_min, self.n_max+1, size=())
            self.current_action = np.random.random_integers(0,1, size=4)
        else:
            self.state -= 1
        assert len(self.current_action) == 4
        return self.current_action


class ManualActionGenerator(AbstractActionGenerator):
    ''' monitors arrow and esc, etc keys via pygame and returns the status of all arrow
        keys as a list [key_down, key_up, key_left, key_right], with 1 for "is pressed", 0 otherwise'''
    def __init__(self):
        self.state = None
        self.keysdown = OrderedDict({ploc.K_UP: False, ploc.K_DOWN: False, ploc.K_LEFT: False, ploc.K_RIGHT: False})
    def action(self):
        for event in pygame.event.get():
            if event.type == ploc.QUIT or \
                                    event.type == ploc.KEYDOWN and (event.key in [ploc.K_ESCAPE, ploc.K_q]):
                self.state = 0
                return np.array([-1,-1,-1,-1])
            if event.type == ploc.KEYDOWN and event.key in [ploc.K_DOWN, ploc.K_UP, ploc.K_LEFT, ploc.K_RIGHT]:
                #self.player_body.velocity += (0, +100.)
                self.keysdown[event.key] = True
            if event.type == ploc.KEYUP and event.key in [ploc.K_DOWN, ploc.K_UP, ploc.K_LEFT, ploc.K_RIGHT]:
                self.keysdown[event.key] = False
        return np.array([int(keydown) for keydown in self.keysdown.values()])

def get_action_generator(mode='static', n_min=None, n_max=None):
    if mode=='static':
        return NoActionGenerator()
    elif mode =='random':
        return RandomActionGenerator(n_min=n_min, n_max=n_max)
    elif mode == 'manual':
        return ManualActionGenerator()


def generate(env, num_seqs, seq_len, path, identifier_str, rgb=True, mode='static', n_min=None, n_max=None, sampling_step=1, first_file_num=1):
    '''
    Storage format: ?
        Need: a copy of the configuration.
        An array with all the ground truth positions / the "state". Label this! (Include a header)
        An array with the frames (same index as other array)

        store as .npz, with header information (each field of self.state is an array, store it with title)

        mode: one of 'static', 'manual', 'random'
        n_min, n_max: use with mode 'random'; see RandomActionGenerator for more
        first_file_num: added on 8th of Jan; starting file number for the files to be created

        So far, just discards the first (state, frame) pair. So, the action array is as long as all
        the other arrays.
    '''
    # pre-allocate a matrix to store the frames
    #total_num_frames = num_seqs * seq_len
    if MAX_NUM_SEQUENCES_PER_FILE < num_seqs:
        n_per_file =  MAX_NUM_SEQUENCES_PER_FILE
    else:
        n_per_file = num_seqs
    state_arrs = OrderedDict({})
    for key, space in env.state_space.spaces.items():
        #tp = np.float32 if type(space) == gym.spaces.Box else np.int32
        shp = (n_per_file, seq_len,) + space.shape
        if key=="body_types":
            state_arrs[key] = np.zeros(shp, dtype='|S10')
        else:
            state_arrs[key] = np.zeros(shp, dtype=space.dtype)
    # one array contains strings
    if mode != 'static':
        state_arrs['action'] = np.zeros((n_per_file, seq_len, 4), dtype=np.int8)
    if rgb:
        frame_arr = np.zeros((n_per_file, seq_len, env.screen_size[0], env.screen_size[1], 3), dtype=np.uint8)
    else:
        frame_arr = np.zeros((n_per_file, seq_len, env.screen_size[0], env.screen_size[1]), dtype=np.uint8)

    file_seq_idx = 0
    file_num = first_file_num
    gen = get_action_generator(mode, n_min=n_min, n_max=n_max)
    env.start()
    t0 = time.time()
    for si in range(num_seqs):

        for t in range(seq_len):
            # step and store frames and states in arrays
            action = gen.action()
            for step in range(sampling_step):
                state, obs = env.step(action=action)
            if not rgb:
                obs = RGB2gray_nd(obs)
            frame_arr[file_seq_idx, t, ...] = obs
            if mode != 'static':
                assert 'action' not in state.keys()
                state['action'] = action # will then be transfered to array two lines below
            for key in state_arrs.keys():
                state_arrs[key][file_seq_idx, t, ...] = state[key]
                check_validity(state[key], key)

        # reset the environment, recreate bouncers
        file_seq_idx += 1
        env.reset()
        print("\rCreated sequence "+str(si+1)+". Time: "+str(time.time() - t0), end='')

        # if we'd exceed the sequence limit in the next step, store it:
        if (file_seq_idx + 1)  > MAX_NUM_SEQUENCES_PER_FILE:
            # store arrays to disc
            np.save(filepath(path, identifier_str+"_frames", file_num), frame_arr)
            np.savez(filepath(path, identifier_str+"_labels", file_num), **state_arrs)
            logging.info("\tStored data to "+filepath(path, identifier_str+"_*", file_num))
            logging.info("\t Time taken: "+str(time.time() - t0))
            file_num += 1
            file_seq_idx = 0

    if file_seq_idx > 0:
        # store remaining frames to disc
        # state arrays:
        for key in state_arrs.keys():
            state_arrs[key] = state_arrs[key][ :file_seq_idx]
        np.savez(filepath(path, identifier_str+"_labels", file_num), **state_arrs)
        # frame array
        frame_arr = frame_arr[ :file_seq_idx]
        np.save(filepath(path, identifier_str+"_frames", file_num), frame_arr)
        logging.info("Stored data to "+filepath(path, identifier_str+"_*", file_num))
        logging.info("\t Time taken: "+str(time.time() - t0))





def generate_train_test(cfg, env_cfg):
    ''' Generate a train and a test set as specified in the configuration dicts'''
    assert cfg['store_frames']
    ensure_dir(cfg['output_dir'])
    # if this is an extension of another dataset, choose different config-file names to not overwrite
    add_string = "" if cfg['train_first_file_number'] == 1 else "_from_"+str(cfg['train_first_file_number'])
    export_config_txt(cfg, os.path.join(cfg['output_dir'], "config"+add_string+".txt"))
    export_config_txt(env_cfg, os.path.join(cfg['output_dir'], "env_config"+add_string+".txt"))
    set_logger(os.path.join(cfg['output_dir'], "log"+add_string+".txt"))
    logging.info("Copied config to output path: "+cfg['output_dir'])


    # set up environment
    if env_cfg['environment'] == 'moving_walls':
        env = BouncingBallEnv(env_cfg, moving_walls=True)
    elif env_cfg['environment'] == 'simple':
        env = BouncingBallEnv(env_cfg, moving_walls=False)
    else:
        raise ValueError("Environment "+env_cfg['environment']+" not recognized")

    file_base_name = "bouncing"
    file_base_name_train = file_base_name + "_train"
    file_base_name_val = file_base_name + "_val"
    file_base_name_test = file_base_name + "_test"

    # fixed seed
    np.random.seed(cfg['random_seed'])
    # generate

    t0 = time.time()
    if 'train_first_file_number' not in cfg.keys():
        cfg['train_first_file_number'] = 1
        cfg['valid_first_file_number'] = 1
        cfg['test_first_file_number'] = 1
    generate(env, cfg['num_sequences_train'], cfg['seq_length'], cfg['output_dir'],  file_base_name_train,
             mode=cfg['mode'], n_min=cfg.get('action_change_interval_min'),
             n_max=cfg.get('action_change_interval_max'), sampling_step=cfg['sampling_step'], rgb=cfg['store_as_rgb'],
             first_file_num=cfg['train_first_file_number'])
    logging.info("Finished train data creation. Time: "+str((time.time() - t0)/60.) +" min")
    generate(env, cfg['num_sequences_valid'], cfg['seq_length'], cfg['output_dir'],  file_base_name_val,
             mode=cfg['mode'], n_min=cfg.get('action_change_interval_min'),
             n_max=cfg.get('action_change_interval_max'), sampling_step=cfg['sampling_step'], rgb=cfg['store_as_rgb'],
             first_file_num=cfg['valid_first_file_number'] )
    logging.info("Finished validation data creation. Time: "+str((time.time() - t0)/60.) +" min")
    generate(env, cfg['num_sequences_test'], cfg['seq_length'], cfg['output_dir'],  file_base_name_test,
             mode=cfg['mode'], n_min=cfg.get('action_change_interval_min'),
             n_max=cfg.get('action_change_interval_max'), sampling_step=cfg['sampling_step'], rgb=cfg['store_as_rgb'],
             first_file_num=cfg['test_first_file_number'])
    logging.info("Finished test data creation. Time: "+str((time.time() - t0)/60.) +" min")

    logging.info("Generated "+str(cfg['num_sequences_train']) + \
                " training sequences and " + str(cfg['num_sequences_valid']) + \
                " validation sequences and "+str(cfg['num_sequences_test']) + \
                " testing sequences of "+str(cfg['seq_length'])+" frames each.")
    logging.info("Training data is at: "+ os.path.join(cfg['output_dir'], file_base_name_train))
    logging.info("Testing data is at: "+ os.path.join(cfg['output_dir'], file_base_name_test))








if __name__ == '__main__':

    default_config = "pygame_code.env_config"

    parser = argparse.ArgumentParser(description='Generate frame sequences of bouncing objects.')
    parser.add_argument('--config', dest='config_source', type=str, default=default_config, action='store',
                        help='the filename without .py, prepended by any paths by dots, to a file containing '
                             'the configuration as a python dictionary.')


    args = parser.parse_args()
    #print(args.config_source)

    i = importlib.import_module(args.config_source)

    generate_train_test(i.cfg, i.env_cfg)