# python script to run GAN algorithms.

import os, sys
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--type', default='dcgan', help='Select GAN type (dcgan/context-encoder/ali/discogan).')
args = parser.parse_args()
params = vars(args)
print json.dumps(params, indent = 4)


gan_type = params['type']

if gan_type == 'dcgan': os.system('th __0_dcgan/script/server.lua')
elif gan_type == 'context-encoder' : os.system('th __1_context-encoder/script/server.lua')
elif gan_type == 'ali' : os.system('th __2_ali/script/server.lua')
elif gan_type == 'discogan' : os.system('th __3_discogan/script/server.lua')
else:
    print('Error: wrong type arguments!')
    os.exit()


