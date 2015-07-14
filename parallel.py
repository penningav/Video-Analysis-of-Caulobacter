#!/usr/bin/env python

"""
This module is used to parallelize analysis code over multiple CPU cores.

"""

import multiprocessing
import sys

import numpy as np

__author__ = 'Charlie Wright'
__email__ = 'charles.s.wright@gmail.com'

NUM_PROCS = 8


def worker(func, args):
    """
    The worker function invoked in a process.

    args:
        func: the function performing the calculations
        args: the input arguments (must be provided as a list of lists)

    """
    [func(*v) for v in args]


def main(func, args, num_procs=NUM_PROCS, timeout=None):
    """
    The main multi-processing function. Processes each function by splitting
    the arguments into chunks based on the number of process required and
    assigning them to each target worker using the multiprocessing module.
    Specify a finite value of timeout for the function to exit without
    waiting on each spawned process to complete before exiting.

    args:
        func: the function performing the calculations
        args: the input arguments (must be provided as a list of lists)

    kwargs:
        num_procs (int): number of processes
        timeout (float): time to wait before exiting

    """
    
    # Each process will get 'chunk_size' calculations
    num_args = len(args)
    chunk_size = int(np.ceil(num_args / float(num_procs)))
    
    # Start processes running
    procs = []
    for i in range(num_procs):
        args_chunk = args[chunk_size * i:chunk_size * (i + 1)]
        p = multiprocessing.Process(target=worker, args=(func, args_chunk))
        procs.append(p)
        p.start()
    
    # Wait for all worker processes to finish
    [p.join(timeout) for p in procs]
    [p.terminate() for p in procs if p.is_alive()]
    [p.join() for p in procs]


if __name__ == '__main__':
    func, args = sys.argv[1:3]
    num_procs = int(sys.argv[3]) if len(sys.argv) > 3 else NUM_PROCS
    main(func, args, num_procs)
