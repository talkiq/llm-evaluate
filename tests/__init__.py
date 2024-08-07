import os

# Don't use GPUs for tests
os.environ['CUDA_VISIBLE_DEVICES'] = ''
