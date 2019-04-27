import gym

# def gym_space_shape(space, return_type=True):
#     ''' Return a shape for storing the contents of a gym.Space to a numpy array.
#         (Not needed (I think), all gym.Spaces have an attribute 'shape'.)'''
#
#     assert isinstance(space, gym.Space)
#
#     tp = type(space)
#     spaceshape = None
#     spacetype = None
#     if tp == gym.spaces.Box:
#         spaceshape = space.shape
#     elif tp == gym.spaces.Dict:
#         raise ValueError("A dict of spaces does not have one single shape. Call this function on the elements of that dict instead.")
#     elif tp == gym.spaces.Discrete:
#         spaceshape = space.shape
#     elif tp == gym.spaces.MultiBinary:
#
#     elif tp == gym.spaces.MultiDiscrete:
#
#     else:
#         raise NotImplementedError("gym_space_shape is not yet implemented for space type "+str(tp))
#



