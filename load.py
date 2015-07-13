#!/usr/bin/env python

import codecs
import os
import re
import shutil
import sys
import traceback

import cPickle as pickle
import numpy as np

from distutils import archive_util
from pandas import concat, read_pickle, Series, DataFrame, MultiIndex
from numpy.linalg import norm
from scipy.interpolate import splev

from matplotlib import pyplot as plt
from matplotlib import cm, rc

import read

__author__ = "Charlie Wright"
__email__ = "charles.s.wright@gmail.com"

# Set plotting options
rc('font', family='serif')
rc('text', usetex=True)
num_colors = 16
cmap = 'Set1'
color_cycle = [cm.get_cmap(cmap)(float(i) / num_colors, 1)
               for i in range(num_colors)]
rc('axes', color_cycle=color_cycle)

# Set relevant constants
UPLOADS_DIR = os.path.join(os.path.expanduser('~'), 'Google Drive',
                           'Scherer Lab', 'Analysis (Spline)')

class TraceData():
    """
    This class loads and formats data by trace.

    """

    PX2UM = 13. / 250. # pixel size on chip in microns divided by magnification
    COLOR_CYCLE = color_cycle
    TXT_KEYS = ('Area',
                'AreaStalked',
#                 'AreaSwarmer',
                'Centroid',
                'DivisionTime',
#                 'EdgeSpline',
                'Event',
#                 'FourierCoef',
                'Generation',
                'Length',
                'LengthMax',
                'LengthStalked',
                'LengthStalkedMax',
                'LengthStalkedMaxCenter',
                'LengthStalkedMaxPole',
                'LengthStalkedMin',
#                 'LengthSwarmer',
                'LengthSwarmerMax',
                'LengthSwarmerMaxCenter',
                'LengthSwarmerMaxPole',
                'LengthSwarmerMin',
#                 'MidSpline',
                'Perimeter',
                'Radius',
                'RadiusStalked',
                'RadiusSwarmer',
                'StalkedPole',
                'SwarmerPole',
                'Time',
                'WidthMax',
                'WidthMean',
                'WidthMin',
#                 'Widths',
                'WidthsSmoothed',
                'WidthStalkedMax',
                'WidthStalkedMaxCenter',
                'WidthStalkedMaxPole',
                'WidthStalkedMin',
                'WidthSwarmerMax',
                'WidthSwarmerMaxCenter',
                'WidthSwarmerMaxPole',
                'WidthSwarmerMin'
                )

    PLT_KEYS = ('Area',
                'DivisionTime',
                'Generation',
                'Length',
                'StalkedPole', 'SwarmerPole')

    def __init__(self, analysis_dir, posns):
        """
        Load all saved data for the experiment and specified positions.

        args:
            analysis_dir (path): experiment analysis folder
            posns (list like): names of positions to load

        """
        self.analysis_dir = analysis_dir
        self.expt_name = os.path.basename(self.analysis_dir)
        top_dir = os.path.sep.join(analysis_dir.split(os.path.sep)[:-2])
        self.tables_dir = os.path.join(top_dir, 'Tables', self.expt_name)
        self.figures_dir = os.path.join(top_dir, 'Figures', self.expt_name)
        self.uploads_dir = os.path.join(UPLOADS_DIR, self.expt_name + '.zip')
        self.posns = []
        self.expt_data = None
        self.posn_data = []     # indexed only by position
        self.trace_data = []    # indexed by position and trace
        self.frame_data = []    # indexed by position, trace and frame
        for i, p in enumerate(posns):
            posn_dir = os.path.join(self.analysis_dir, p)

            # Try to load the saved data dictionary from file
            n = os.path.join(posn_dir, 'saved.pickle')
            try:
                d = read_pickle(n)
            except IOError:
                d = {}
                print IOError('No such file: %s' % n)
            self.posns.append(p)

            self.posn_data.append({})
            self.trace_data.append({})
            self.frame_data.append({})
            for k, v in d.items():
                if k in ('TimeP', 'TimeF'):
                    # Convert from seconds to minutes
                    self.posn_data[i][k] = v / 60
                elif k in ('Label', ):
                    self.frame_data[i][k] = v
                elif k in ('Taus', ):
                    # Convert from seconds to minutes
                    self.trace_data[i][k] = [u / 60 for u in v]
                else:
                    self.trace_data[i][k] = v

            # Load the log data dictionary from file
            with open(os.path.join(posn_dir, 'log.pickle'), 'rb') as f:
                d = pickle.load(f)
                for k in ('stage', ):
                    self.posn_data[i][k.capitalize()] = d[k]

                if self.expt_data is None:
                    self.expt_data = {}
                    for k in ('log', 'image', 'pumps'):
                        if k in ('pumps', ):
                            for j, u in enumerate(d[k]):
                                d[k][j]['Time'] = np.asarray(u['Time']) / 60
                        self.expt_data[k.capitalize()] = d[k]

        self.num_posns = len(self.posns)
        self.num_traces = sum([len(d['Trace']) for d in self.trace_data])
        self.num_frames = min([len(d['Label'].index.levels[1])
                               for d in self.frame_data])
        self.num_gens = sum([sum([len(v) for v in d['Divns']])
                             for d in self.trace_data])

    def __getitem__(self, idx):
        """
        Retrieve data for specified trace.

        args:
            idx (int): corresponds to the trace index

        returns:
            dict: data values for a single cell trace

        """
        j = None
        if isinstance(idx, int):
            i = idx
        else:
            i = idx[0]
            if len(idx) == 2:
                j = idx[1]
        a = self.posn_data[i]
        b = self.trace_data[i]
        c = self.frame_data[i]
        if j is None:
            d = dict(a.items() + b.items() + c.items())
        else:
            d = dict(a)
            for k, v in b.items():
                d[k] = v[j]
            for k, v in c.items():
                d[k] = v.ix[j]
            for k in ('TimeF', 'TimeP'):
                if d.has_key(k):
                    d[k] = d[k][d['Label'].index]
        return d

    def __len__(self):
        """
        Print the number of traces in the dataset.

        """
        return self.num_traces

    def loadvar(self, key, convert=True):
        """
        Load additional variables from file. These should be values that have been analyzed in blocks, but not sorted according to trace. This function will parse the requested variables for all saved traces.

        args:
            key (str): name of variable to load from file

        """
        name = key + '.pickle'
        for i, p in enumerate(self.posns):
            b = os.path.join(self.analysis_dir, p, 'blocks')

            # Load the original Series from file
            s = None
            for v in read.listdirs(b, read.PATTERN['blockdir']):
                n = os.path.join(b, v, name)
                try:
                    s = concat((s, read_pickle(n)))
                except IOError:
                    raise IOError('No such file: %s' % n)
            if s is None:
                continue

            # Reformat according to the saved traces
            j = self.frame_data[i]['Label']
            v = Series(index=j.index, dtype=s.dtype)
            for (trace, frame), label in j.iteritems():
                v[trace, frame] = s[frame, label]
            self.frame_data[i][key] = v

        # Convert to the correct units
        if convert:
            if 'Area' in key:
                for i in range(self.num_posns):
                    self.frame_data[i][key] *= self.PX2UM ** 2
            elif ('Centroid' in key or 'Length' in key or 'Perimeter' in key or
                  'Pole' in key or 'Radius' in key or 'Width' in key):
                for i in range(self.num_posns):
                    self.frame_data[i][key] *= self.PX2UM
            elif 'Spline' in key:
                for i in range(self.num_posns):
                    for k, v in self.frame_data[i][key].iteritems():
                        if len(v) == 3:
                            v[1][0] *= self.PX2UM
                            v[1][1] *= self.PX2UM
                        self.frame_data[i][key][k] = v

    def delvar(self, key):
        """
        Delete specified variable from memory. These function will only delete variables that have been loaded from file.

        args:
            key (str): name of variable to delete

        """
        if key not in ('Label', ):
            for i in range(self.num_posns):
                if self.frame_data[i].has_key(key):
                    del self.frame_data[i][key]

    def uploadall(self):
        """
        Upload all data to shared Google Drive folder.

        """
        old_dir = os.path.abspath(os.curdir)

        # Zip the tables directory
        os.chdir(os.path.dirname(self.tables_dir))
        base_name = os.path.basename(self.tables_dir)
        zip_name = archive_util.make_zipfile(base_name, base_name)

        # Upload the zipped file
        shutil.move(zip_name, os.path.join(UPLOADS_DIR, zip_name))
        os.chdir(old_dir)

    '''
    def upload(self, key):
        """
        Upload data to shared Google Drive folder.

        args:
            key (str): name of variable to upload

        """
        read.rmkdir(self.uploads_dir)

        for v in os.listdir(self.tables_dir):
            if key in v and os.path.splitext(v)[1] == '.txt':
                src = os.path.join(self.tables_dir, v)
                dst = os.path.join(self.uploads_dir, v)
                if os.path.isfile(src):
                    # Just copy the file over
                    shutil.copy2(src, dst)
                elif os.path.isdir(src):
                    for root, dirs, files in os.walk(src):
                        for v in (os.path.join(root, d) for d in dirs):
                            # Make all subdirectories from scratch
                            read.cmkdir(v.replace(src, dst))
                        for v in (os.path.join(root, f) for f in files):
                            if os.path.splitext(v)[1] == '.txt':
                                # Then copy the files over
                                shutil.copy2(v, v.replace(src, dst))
    '''

    def exportall(self, upload=False):
        """
        Export all data to text files.

        kwargs:
            upload (bool): upload to shared folder at time of exporting

        """
        [self.export(k, upload=upload) for k in self.TXT_KEYS]

    def export(self, key, upload=False):
        """
        Export data to a CSV file.

        args:
            key (str): name of variable to export

        kwargs:
            upload (bool): upload to shared folder at time of exporting

        """
        if key not in self.TXT_KEYS:
            raise ValueError("'%s' is not in TXT_KEYS" % key)

        has_var = self[0].has_key(key)
        if not has_var:
            try:
                self.loadvar(key)
            except IOError:
                pass

        # The file name is auto-generated, with 3 decimals and tab delimiters
        read.rmkdir(self.tables_dir)
        file_name = os.path.join(self.tables_dir,
                                 self.expt_name + '_' + key + '.txt')
        fmt = '%.3f'
        dlm = '\t'

        # Recast the DataFrame into a matrix, padded with NaN values
        nans = np.ones((self.num_traces, self.num_frames)) * np.nan
        if key in ('EdgeSpline', 'MidSpline'):
            # Load the StalkedPole and SwarmerPole values
            has_stalked = self[0].has_key('StalkedPole')
            if not has_stalked:
                self.loadvar('StalkedPole')
            has_swarmer = self[0].has_key('SwarmerPole')
            if not has_swarmer:
                self.loadvar('SwarmerPole')

            # Export data to a new Spline folder
            top_dir = os.path.join(self.tables_dir, self.expt_name + '_' + key)
            read.cmkdir(top_dir)

            # Number of digits determined by the number of cells traces
            num_digits = int(np.ceil(np.log10(self.num_traces + 1)))

            # Loop over each position then cell separately
            k = 0
            u = np.linspace(0., 1., 1e3)
            for i in range(self.num_posns):
                for t in range(len(self[i]['Trace'])):
                    j = k + t + 1
                    data = [[] for _ in range(5)]
                    for f, tck in self[i, t][key].iteritems():
                        if tck:
                            # Start indexing the frames at 1
                            data[0].append(f + 1)

                            # Find the indexes of the breaks
                            xs = np.asarray(zip(*splev(u, tck)))
                            p1 = self[i, t]['StalkedPole'].ix[f]
                            p2 = self[i, t]['SwarmerPole'].ix[f]
                            k1 = np.argmin([norm(v) for v in p1 - xs])
                            k2 = np.argmin([norm(v) for v in p2 - xs])
                            data[1].append((u[k1], u[k2]))

                            # Save the spline values
                            data[2].append(tck[0])
                            data[3].append(tck[1][0])
                            data[4].append(tck[1][1])

                    # Export data to a subfolder for each trace
                    sub_dir = os.path.join(top_dir, 'trace' +
                                           str(j).zfill(num_digits))
                    read.rmkdir(sub_dir)

                    # Save four files for each trace
                    file_names = []
                    for v in ('Frames', 'Breaks', 'Knots',
                              'ControlX', 'ControlY'):
                        n = os.path.join(sub_dir,
                                         self.expt_name + '_' + v + '.txt')
                        file_names.append(n)

                    # Save each value as an array with no empty values
                    np.savetxt(file_names[0], data[0],
                               fmt='%.0f', delimiter=dlm)
                    for n, d in zip(file_names[1:], data[1:]):
                        with open(n, 'w') as f:
                            for v in d:
                                f.write(dlm.join([fmt % x for x in v]) + '\n')
                k += (t + 1)
            if not has_stalked:
                self.delvar('StalkedPole')
            if not has_swarmer:
                self.delvar('SwarmerPole')
        elif key in ('Event', ):
            # Export pump metadata
            for i, v in enumerate(self.expt_data['Pumps']):
                n = str(i+1).join(os.path.splitext(file_name))
                with codecs.open(n, encoding='utf-8', mode='w') as f:
                    f.write(v['Solution'] + '\n')
                    f.write('%s\t%s\t%s\n' % (v['Units'], 'TimeOn', 'TimeOff'))
                    for j in range(len(v['Rate'])):
                        f.write('%.3f\t%.3f\t%.3f\n' %
                            (v['Rate'][j], v['Time'][j][0], v['Time'][j][1]))
        elif key in ('Label', ):
            data = nans.copy()
            k = 0
            for i in range(self.num_posns):
                for (t, f), v in self[i][key].iteritems():
                    j = k + t
                    data[j, f] = v
                k += (t + 1)
            np.savetxt(file_name, data, fmt='%.0f', delimiter=dlm)
        elif key in ('Generation', ):
            # Save the generation counts (start indexing from 1)
            data = nans.copy()
            k = 0
            for i in range(self.num_posns):
                for t, v in enumerate(self[i]['Gens']):
                    for u, f in enumerate(v):
                        j = k + t
                        data[j, f] = u
                k += (t + 1)
            data += 1
            np.savetxt(file_name, data, fmt='%.0f', delimiter=dlm)
        elif key in ('DivisionTime', ):
            taus = []
            for i in range(self.num_posns):
                for v in self[i]['Taus']:
                    taus.append(v)
            max_gens = max([len(v) for v in taus])
            data = np.ones((self.num_traces, max_gens)) * np.nan
            for i, v in enumerate(taus):
                n = len(v)
                data[i][:n] = v
            np.savetxt(file_name, data, fmt=fmt, delimiter=dlm)
        elif key in ('Centroid', 'StalkedPole', 'SwarmerPole'):
            for a, n in enumerate(('X', 'Y')):
                axis_name = n.join(os.path.splitext(file_name))
                data = nans.copy()
                k = 0
                for i in range(self.num_posns):
                    for (t, f), v in self[i][key].iteritems():
                        j = k + t
                        data[j, f] = v[a]
                    k += (t + 1)
                np.savetxt(axis_name, data, fmt=fmt, delimiter=dlm)
        elif key in ('FourierFit', 'FourierCoef'):
            # Export data to a new folder
            top_dir = os.path.join(self.tables_dir, self.expt_name + '_' + key)
            read.cmkdir(top_dir)

            if 'FourierFit' == key:
                # Old representation
                num_coefs = 10
                coef_range = range(num_coefs)
            elif 'FourierCoef' == key:
                # New representation
                num_coefs = 20
                coef_range = range(-num_coefs, num_coefs+1)

            for c in coef_range:
                for n in ('Real', 'Imag'):
                    file_name = os.path.join(top_dir, self.expt_name + '_' + key
                                             + str(c).zfill(2) + n + '.txt')
                    data = nans.copy()
                    k = 0
                    for i in range(self.num_posns):
                        for (t, f), v in self[i][key].iteritems():
                            j = k + t
                            data[j, f] = v[c].real if n == 'Real' else v[c].imag
                        k += (t + 1)
                    np.savetxt(file_name, data, fmt=fmt, delimiter=dlm)
        elif key in ('WidthsSmoothed', ):
            # Export data to a new Widths folder
            top_dir = os.path.join(self.tables_dir, self.expt_name + '_' + 'Widths')
            read.cmkdir(top_dir)

            num_points = 500
            for c in xrange(num_points):
                file_name = os.path.join(top_dir, self.expt_name + '_' + 'Widths'
                                         + str(c).zfill(3) + '.txt')

                # Recast the DataFrame into a matrix, padded with NaN values
                data = nans.copy()
                k = 0
                for i in range(self.num_posns):
                    for (t, f), v in self[i][key].iteritems():
                        j = k + t
                        if np.any(v):
                            data[j, f] = v[c]
                    k += (t + 1)
                np.savetxt(file_name, data, fmt=fmt, delimiter=dlm)
        elif key in ('Time', ):
            # Recast the vector into a matrix
            data = nans.copy()
            k = 0
            for i in range(self.num_posns):
                v = self[i]['TimeP']
                for t, _ in enumerate(self[i]['Trace']):
                    j = k + t
                    data[j] = v
                k += (t + 1)
            np.savetxt(file_name, data, fmt=fmt, delimiter=dlm)
        elif key in ('Mother', ):
            # Save the identity of the mother (start indexing from 1)
            data = np.ones(self.num_traces) * np.nan
            k = 0
            for i in range(self.num_posns):
                for t, v in enumerate(self[i]['Mother']):
                    if v is not None and v in self[i]['Trace']:
                        d = k + t
                        m = k + self[i]['Trace'].index(v)
                        data[d] = m
                k += (t + 1)
            data += 1
            np.savetxt(file_name, data, fmt='%.0f', delimiter=dlm)
        elif key in ('WidthMean', ):
            # Load the Widths values
            has_widths = self[0].has_key('Widths')
            if not has_widths:
                self.loadvar('Widths')

            # Recast the DataFrame into a matrix, padded with NaN values
            data = nans.copy()
            k = 0
            for i in range(self.num_posns):
                for (t, f), v in self[i]['Widths'].iteritems():
                    j = k + t
                    data[j, f] = np.nanmean(v)
                k += (t + 1)
            np.savetxt(file_name, data, fmt=fmt, delimiter=dlm)

            if not has_widths:
                self.delvar('Widths')
        else:
            # Recast the DataFrame into a matrix, padded with NaN values
            data = nans.copy()
            k = 0
            for i in range(self.num_posns):
                for (t, f), v in self[i][key].iteritems():
                    j = k + t
                    data[j, f] = v
                k += (t + 1)
            np.savetxt(file_name, data, fmt=fmt, delimiter=dlm)

        if upload:
            self.upload(key)

        if not has_var:
            self.delvar(key)

    def plotall(self):
        """
        Plot all valid variables.

        """
        [self.plot(k) for k in self.PLT_KEYS]

    def plot(self, key):
        """
        Plot the data and save to a PDF file.

        args:
            key (str): name of variable to export

        """
        if key not in self.PLT_KEYS:
            raise ValueError("'%s' is not in PLT_KEYS" % key)

        has_var = self[0].has_key(key)
        if not has_var:
            try:
                self.loadvar(key)
            except IOError:
                pass

        # Set the correct units for the axes labels
        if key in ('Area', ):
            units = '$\mu$m$^{2}$'
        elif key in ('Generation', ):
            units = 'number'
        elif key in ('DivisionTime', ):
            units = 'min'
        else:
            units = '$\mu$m'

        # Create a formatted label with spaces between strings
        idxes = [i for i, v in enumerate(key) if v.isupper()]
        idxes.append(len(key))
        label = ' '.join([key[i1:i2] for i1, i2 in zip(idxes[:-1], idxes[1:])])

        # First plot the time series data for each position
        for i, p in enumerate(self.posns):
            time = self[i]['TimeP']
            gens = self[i]['Gens']

            if key in ('Generation', 'DivisionTime'):
                num_traces = len(gens)
                num_frames = len(time)
                frames = range(num_frames) * num_traces
                traces = []
                for v in xrange(num_traces):
                    traces.extend([v] * num_frames)
                index = MultiIndex.from_arrays([traces, frames],
                                               names=('Trace', 'Frame'))
                vals = np.ones(shape=(num_traces, num_frames)) * np.nan
                if key in ('Generation', ):
                    for j, g in enumerate(gens):
                        for k, v in enumerate(g):
                            vals[j][v.start:v.stop] = k
                    vals = np.hstack(vals)
                    vals += 1
                elif key in ('DivisionTime', ):
                    for j, g in enumerate(gens):
                        d = self[i]['Taus'][j]
                        for k, v in enumerate(g):
                            vals[j][v.start:v.stop] = d[k]
                    vals = np.hstack(vals)

                data = DataFrame(vals, index=index, columns=(key, ))[key]
                data = data[np.isfinite(data)]
                if len(data) == 0: continue

                data.units = units
                data.posn = p
                data.file = os.path.join(self.figures_dir, p, key,
                                         '_'.join([self.expt_name, p, key]))
                data.label = label
                read.rmkdir(os.path.dirname(data.file))
                self.plot1Dseries(time, data, gens,
                                  bygens=False, showlog=False, showoffset=False)
            elif key in ('Area', 'Length', 'Perimeter'):
                data = Series(self[i][key], name=key)
                if len(data) == 0: continue

                data.units = units
                data.posn = p
                data.file = os.path.join(self.figures_dir, p, key,
                                         '_'.join([self.expt_name, p, key]))
                data.label = label
                read.rmkdir(os.path.dirname(data.file))
                self.plot1Dseries(time, data, gens)
            elif key in ('Centroid', 'StalkedPole', 'SwarmerPole'):
                for a in ('', 'X', 'Y'):
                    if a is '':
                        data = Series(self[i][key], name=key)
                    else:
                        data = Series([v[0] for v in self[i][key].values],
                                      index=self[i][key].index, name=key+a)
                    if len(data) == 0: continue

                    data.units = units
                    data.posn = p
                    data.file = os.path.join(self.figures_dir, p, key,
                                             '_'.join([self.expt_name,
                                                       p, key + a]))
                    data.label = label
                    read.rmkdir(os.path.dirname(data.file))
                    if a is '':
                        self.plot2Dseries(time, data, gens)
                    else:
                        data.label += ' ' + a
                        self.plot1Dseries(time, data, gens,
                                          showlog=False, showoffset=False)

        if not has_var:
            self.delvar(key)

    def plotpumpdata(self, ax=None):
        if ax is None:
            ax = plt.gca()
        y1, y2 = ax.get_ylim()
        n = len(self.expt_data['Pumps'])
        for j, d in enumerate(self.expt_data['Pumps']):
            if j == 0: continue
            c = cm.get_cmap('gray_r')(float(j) / n, 1)
            for r, (t1, t2) in zip(d['Rate'], d['Time']):
                if r == 0: continue
                x = np.asarray([t1, t2, t2, t1])
                y = np.asarray([y1, y1, y2, y2])
                ax.fill(x, y, color=c, alpha=0.2, linewidth=0)

    def plot1Dseries(self, time, data, gens,
                     bygens=True, showlog=True, showoffset=True):
        fig = plt.figure()

        # Plot time series
        fig.clear()
        ax = fig.add_subplot(111)
        x_min, x_max = float("inf"), -float("inf")
        y_min, y_max = min(data), max(data)
        alpha = min(1., max(0.1, len(data.index.levels[0]) / float(50)))
        for t in data.index.levels[0]:
            if len(data.ix[t]) == 0: continue
            y = data.ix[t]
            x = time[y.index]
            x_min = min(x_min, x[0])
            x_max = max(x_max, x[-1])
            ax.plot(x, y, '.', alpha=alpha, markersize=2)
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))
        self.plotpumpdata(ax)
        ax.set_xlabel(r'Time (min)')
        ax.set_ylabel(r'%s (%s)' % (data.label, data.units))
        ax.set_title(r'%s (%s)'
            % (self.expt_name.replace('_', ' '), data.posn))
        fig.savefig(data.file + 'VsTime.pdf', dpi=150)

        if showlog:
            # Plot with semilog y-scale
            yticks = ax.get_yticks()
            yticks = yticks[(yticks > y_min) & (yticks < y_max)]
            ax.set_yscale('log')
            ax.set_ylim((y_min, y_max))
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(v) for v in yticks])

            # Draw a maximum of 4 horizontal lines, only at integer ticks
            hlines = yticks[yticks % 1 == 0]
            dlines = max(len(hlines)/4, 1)
            hlines = hlines[::dlines]
            ax.hlines(hlines, x_min, x_max, linewidth=0.25)

            ax.set_ylabel(r'ln$\vert$%s$\vert$ (%s)' % (data.label, data.units))
            fig.savefig(data.file + 'LogVsTime.pdf', dpi=150)

        if showoffset:
            # Plot with offset
            fig.clear()
            ax = fig.add_subplot(111)
            d = y_max - y_min
            for j, t in enumerate(data.index.levels[0]):
                if len(data.ix[t]) == 0: continue
                y = data.ix[t] + j * d
                x = time[y.index]
                for k, g in enumerate(gens[j]):
                    q = (y.index >= g.start) & (y.index < g.stop)
                    if not any(q): continue
                    v = y[q]
                    u = x[q]
                    c = self.COLOR_CYCLE[k % num_colors]
                    x_min = min(x_min, x[0])
                    x_max = max(x_max, u[-1])
                    ax.plot(u, v, '-', color=c, linewidth=1)
            ax.set_xlim((x_min, x_max))
            self.plotpumpdata(ax)
            ax.set_xlabel(r'Time (min)')
            ax.set_ylabel(r'%s (offset)' % data.label)
            ax.set_title(r'%s (%s)'
                % (self.expt_name.replace('_', ' '), data.posn))
            fig.savefig(data.file + 'OffsetVsTime.pdf', dpi=150)

        if bygens:
            # Plot vs. generations
            fig.clear()
            ax = fig.add_subplot(111)
            x_min, x_max = 0, -float("inf")
            for j, t in enumerate(data.index.levels[0]):
                if len(data.ix[t]) == 0: continue
                y = data.ix[t]
                x = time[y.index]
                c = self.COLOR_CYCLE[j % num_colors]
                for g in gens[j]:
                    q = (y.index >= g.start) & (y.index < g.stop)
                    if not any(q): continue
                    v = y[q]
                    u = x[q]
                    u -= u[0]
                    x_max = max(x_max, u[-1])
                    ax.plot(u, v, '-', color=c, alpha=alpha)
            ax.set_xlim((x_min, x_max))
            ax.set_ylim((y_min, y_max))
            ax.set_xlabel(r'Time from Division (min)')
            ax.set_ylabel(r'%s (%s)' % (data.label, data.units))
            ax.set_title(r'%s (%s)'
                % (self.expt_name.replace('_', ' '), data.posn))
            fig.savefig(data.file + 'VsTimeByGen.pdf', dpi=150)

            if showlog:
                # Plot with semilog y-scale
                yticks = ax.get_yticks()
                yticks = yticks[(yticks > y_min) & (yticks < y_max)]
                ax.set_yscale('log')
                ax.set_ylim((y_min, y_max))
                ax.set_yticks(yticks)
                ax.set_yticklabels([str(v) for v in yticks])

                # Draw a maximum of 4 horizontal lines, only at integer ticks
                hlines = yticks[yticks % 1 == 0]
                dlines = max(len(hlines)/4, 1)
                hlines = hlines[::dlines]
                ax.hlines(hlines, x_min, x_max, linewidth=0.25)

                ax.set_ylabel(r'ln$\vert$%s$\vert$ (%s)' % (data.label, data.units))
                fig.savefig(data.file + 'LogVsTimeByGen.pdf', dpi=150)
        plt.close(fig)

    def plot2Dseries(self, time, data, gens, bygens=True):
        fig = plt.figure()

        # Plot time series
        fig.clear()
        ax = fig.add_subplot(111, aspect=1)
        alpha = min(1., max(0.01, len(data.index.levels[1]) / float(1e5)))
        y_lim, x_lim = [(0., v * self.PX2UM) for v in
                        self.expt_data['Image']['phase']['shape']]
        for t in data.index.levels[0]:
            if len(data.ix[t]) == 0: continue
            xy = np.vstack(data.ix[t].values)
            x = x_lim[1] - xy[:, 1]
            y = xy[:, 0]
            ax.plot(x, y, '.', alpha=alpha, markersize=2)
        ax.set_xlabel(r'%s X (%s)' % (data.label, data.units))
        ax.set_ylabel(r'%s Y (%s)' % (data.label, data.units))
        ax.set_title(r'%s (%s)' % (self.expt_name.replace('_', ' '), data.posn))
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        fig.savefig(data.file + 'VsXY.pdf', dpi=150)

        if bygens:
            # Plot offset vs. generations
            fig.clear()
            ax = fig.add_subplot(111)
            x_min, x_max = 0, -float("inf")
            alpha = min(1., max(0.1, len(data.index.levels[0]) / float(50)))
            for j, t in enumerate(data.index.levels[0]):
                if len(data.ix[t]) == 0: continue
                y = data.ix[t]
                x = time[y.index]
                c = self.COLOR_CYCLE[j % num_colors]
                for g in gens[j]:
                    q = (y.index >= g.start) & (y.index < g.stop)
                    if not any(q): continue
                    v = np.vstack(y[q].values)
                    u = x[q]
                    u -= u[0]

                    dv = np.sum(np.diff(v, axis=0) ** 2, axis=1) / np.diff(u)
                    dv = np.hstack((0, dv))

                    x_max = max(x_max, u[-1])
                    ax.plot(u, dv, '-', color=c, alpha=alpha)
            ax.set_xlim((x_min, x_max))
            ax.set_xlabel(r'Time from Division (min)')
            ax.set_ylabel(r'%s Displacement (%s/min)' % (data.label, data.units))
            ax.set_title(r'%s (%s)'
                % (self.expt_name.replace('_', ' '), data.posn))
            fig.savefig(data.file + 'DisplacementVsTimeByGen.pdf', dpi=150)
        plt.close(fig)

def main(analysis_dir, posn_nums=None):
    posns = [v for v in read.listdirs(analysis_dir, read.PATTERN['posndir'])]
    if posn_nums is not None:
        posns = [v for v in posns if int(re.match(read.PATTERN['posndir'],
                                                  v).group(2)) in posn_nums]
    return TraceData(analysis_dir, posns)
