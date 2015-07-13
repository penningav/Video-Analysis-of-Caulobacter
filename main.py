#!/usr/bin/env python

import os
import re
import sys
import time
import yaml

import parallel


def loadinputs(file_name):
    """
    Load analysis input parameters from .txt file.
    
    args:
        file_name (str): name of .txt file (YAML-formatted)
    
    """
    d = yaml.load(open(file_name, 'r'))
    
    # Expand all directories
    d['paths'] = {k : os.path.expanduser(v) for k, v in d['paths'].iteritems()}
    
    return d


def setaction(inputs, index):
    """
    Set parameters for the requested action.
    
    args:
        inputs (dict): dict of inputs read from file
        index (int): index of current action
    
    """
    
    # Set parameters for the current action
    d = dict(inputs['parameters'].items() + inputs['actions'][index].items())
    
    # Inputs are raw data for image processing, otherwise analyzed files    
    if d['function'] == 'segment':
        d['input_dir'] = inputs['paths']['raw_data']
    else:
        d['input_dir'] = inputs['paths']['analyses']
    d['output_dir'] = inputs['paths']['analyses']
    
    # Convert list of regex position patterns to full list of positions
    positions = set()
    for v in os.listdir(d['input_dir']):
        [positions.add(v) for p in inputs['positions'] if re.match(p, v) 
         and os.path.isdir(os.path.join(d['input_dir'], v))]
    d['positions'] = sorted(positions)
    
    return d


def main(file_name):
    """
    Analyze experiment specified by .txt file.
    
    args:
        file_name (str): name of .txt file (YAML-formatted)
    
    """
    inputs = loadinputs(file_name)
    
    # Loop over each action in order
    for s in inputs['sequence']:
        action = setaction(inputs, s)
        for p in action['positions']:
            d = dict(action)
            d['input_dir'] = os.path.join(action['input_dir'], p)
            d['output_dir'] = os.path.join(action['output_dir'], p)
            d['position'] = p
            d['experiment'] = inputs['experiment']
            
            # Evaluate the requested function in parallel
            eval('parallel.%s(%s)' % (action['function'], d))
            
            # Save an entry in a log file
            log_file = os.path.join(os.path.dirname(file_name), 'log.txt')
            with open(log_file, 'a') as f:
                s1 = str(d['experiment'])
                s2 = str(d['position'])
                s3 = '%s.%s' % (action['module'], action['function'])
                s4 = time.asctime()
                f.write('{:12s}{:8s}{:16s}{:s}'.format(s1, s2, s3, s4))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        raise IndexError('list index out of range')
        #file_name = '~/Dropbox/Caulobacter/Code/example/parameters.txt'
    main(os.path.expanduser(file_name))
    
    