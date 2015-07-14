#!/usr/bin/env python

import os
import sys

from time import asctime
from phase.workflow import posteditblock

__author__ = 'Charlie Wright'
__email__ = 'charles.s.wright@gmail.com'


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_dir = os.path.expanduser(sys.argv[1])
    else:
        raise IndexError('list index out of range')
    posteditblock(input_dir, {})
    print '========================'
    print asctime(), '\t', input_dir
    print '========================'
