#!/usr/bin/env python

"""
This module contains miscellaneous functions useful in file I/O operations.

"""

import os
import re
import shutil

import cPickle as pickle

from pandas import concat, DataFrame, read_pickle
from time import asctime

__author__ = 'Charlie Wright'
__email__ = 'charles.s.wright@gmail.com'

PATTERN = {'phasetif' : r'^(phase)([\d]*).tif$',
           'fluortif' : r'^(fluor)([\d]*).tif$',
           'blockdir' : r'^(frame)([\d]*)-([\d]*)$',
           'posndir' : r'^(pos)([\d]*)$'}

def getframenum(file_name, pattern=PATTERN['phasetif']):
    """
    Return the frame number by extracting it from the file name, which
    must follow the expected format for either 'phase' or 'fluor'.

    args:
        file_name (str): name of file to search (absolute or relative name)

    kwargs:
        pattern (str): regex pattern used to match file name

    returns:
        int: frame number corresponding to input file name

    """
    m = re.match(pattern, os.path.basename(file_name))
    try:
        frame_num = int(m.group(2))
    except:
        frame_num = -1
    return frame_num

def updatelog(expt, posn, func, input_dir=None):
    """
    Save an entry in a log file.

    args:
        expt (str): name of experiment
        posn (str): name of position
        func (str): name of function

    kwargs:
        input_dir (path): location of log file (if None, print to screen)

    """
    line = '{:12s}{:8s}{:16s}{:s}'.format(expt, posn, func, asctime())
    if input_dir is None:
        print line
    else:
        with open(os.path.join(input_dir, 'analysis.log'), 'a') as f:
            f.write(line + '\n')

def loadlogdata(expt_dir, posn=None):
    """
    Load the log data for the given experiment and position. The position can be specified explicitly by name or by index; if not given, the log data for the first position encountered is imported.

    args:
        input_dir (path): input directory

    kwargs:
        posn (int/str): current position, as a number or string

    returns:
        dict: contents of pickled log file

    """
    if isinstance(posn, str):
        posn_dir = posn
    else:
        for v in listdirs(expt_dir, PATTERN.posndir):
            if posn is None or int(re.match(PATTERN.posndir, v).group(2)) == posn:
                posn_dir = v
                break
    try:
        log_file = os.path.join(expt_dir, posn_dir, 'log.pickle')
        log_data = pickle.load(open(log_file, 'rb'))
    except IOError:
        log_data = {}
    return log_data

def listfiles(input_dir, pattern=r''):
    """
    List files in the specified directory that match the specified regular
    expression pattern. If the keep_dirs or keep_files parameters are False,
    do not return directories or files, respectively. The default is to
    return all files in the specified directory. A generator is returned.

    args:
        input_dir (path): input directory

    kwargs:
        pattern (str): regex pattern

    returns:
        generator: files in the directory matching the input pattern

    """
    for v in os.listdir(input_dir):
        if re.match(pattern, v):
            if os.path.isfile(os.path.join(input_dir, v)):
                yield v

def listdirs(input_dir, pattern=r''):
    """
    List files in the specified directory that match the specified regular
    expression pattern. If the keep_dirs or keep_files parameters are False,
    do not return directories or files, respectively. The default is to
    return all files in the specified directory. A generator is returned.

    args:
        input_dir (path): input directory

    kwargs:
        pattern (str): regex pattern

    returns:
        generator: sub-directories matching the input pattern

    """
    for v in os.listdir(input_dir):
        if re.match(pattern, v):
            if os.path.isdir(os.path.join(input_dir, v)):
                yield v

def makedataframe(input_dir, parameters):
    """
    Make DataFrame from Series data.

    args:
        input_dir (path): input directory

    kwargs:
        parameters (list): parameters to import

    returns:
        DataFrame: has columns corresponding to input parameters

    """
    vs = None
    for p in parameters:
        try:
            v = DataFrame(read_pickle(os.path.join(input_dir, p + '.pickle')))
        except IOError:
            continue
        vs = v if vs is None else vs.join(v)
    return vs

def stitchdataframes(df1, df2, n=None):
    """
    Stitch data from two separate blocks together, including only the first and last few frames of each block, resulting in a DataFrame with just the frames near the overlapping region. The default is to include all frames.

    args:
        df1 (DataFrame): DataFrame from 1st block
        df2 (DataFrame): DataFrame from 2nd block

    kwargs:
        num_frames (list): amount of overlap

    returns:
        DataFrame: overlaps between inputs data frames

    """
    f1 = df1.index.levels[0][-n:]
    f2 = df2.index.levels[0][:+n]

    j1 = [i for i in df1.index if i[0] in f1]
    j2 = [i for i in df2.index if i[0] in f2]
    return concat((df1.ix[j1], df2.ix[j2]))

def rmkdir(input_dir):
    """
    Recursively make directory (make new directory, filling in all required directories above it).

    args:
        input_dir (path): new directory tree to make

    """
    files = os.path.abspath(input_dir).split(os.sep)
    for i in range(1, len(files)):
        v = os.sep.join(files[:i+1])
        if not os.path.isdir(v):
            os.mkdir(v)

def cmkdir(input_dir):
    """
    Make clean copy of directory. Makes any top directories as needed.

    args:
        input_dir (path): new directory tree to (re)make

    """
    if os.path.isdir(input_dir):
        shutil.rmtree(input_dir)
    rmkdir(input_dir)

def walklevel(basepath, level=1):
    """
    Walk through directory levels to specified depth. Based on stackoverflow forum answer by nosklo at: stackoverflow.com/questions/229186/os-walk-without-digging-into-directories-below

    args:
        basepath (path): starting directory

    kwargs:
        level (int): number of levels

    returns:
        generator: root directory, sub-directories, sub-files

    """
    basepath = basepath.rstrip(os.sep)
    assert os.path.isdir(basepath)
    num_sep = basepath.count(os.sep)
    for root, dirs, files in os.walk(basepath):
        yield root, dirs, files
        if num_sep + level <= root.count(os.sep):
            del dirs[:]
