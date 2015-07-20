#!/usr/bin/env python

"""
This module is used for interactive manual editing of automated analysis results.

"""

import os
import shutil
import sys
import zipfile

import cPickle as pickle
import numpy as np

from Tkinter import *
from ttk import Frame, Button, Label, Style

from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
#from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg

from pandas import read_pickle, DataFrame, MultiIndex, Series
from skimage.io import imread

sys.path.append(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
import read

__author__ = 'Charlie Wright'
__email__ = 'charles.s.wright@gmail.com'

# Change or disable matplotlib hotkeys
plt.rcParams['keymap.all_axes'] = ''
plt.rcParams['keymap.back'] = ''
plt.rcParams['keymap.forward'] = ''
plt.rcParams['keymap.home'] = ['H', 'h']
plt.rcParams['keymap.pan'] = ['P', 'p']
plt.rcParams['keymap.save'] = ''
plt.rcParams['keymap.xscale'] = ''
plt.rcParams['keymap.yscale'] = ''
plt.rcParams['keymap.zoom'] = ['Z', 'z']


def to_bounds(val, min_val=None, max_val=None):
    """
    Coerce a value within the lower and upper bounds.

    args:
        val (float): input value

    kwargs:
        min_val (float): minimum allowed value; if None, do not set a lower bound
        max_val (float): maximum allowed value; if None, do not set an upper bound

    """
    if min_val is not None:
        val = max(val, min_val)
    if max_val is not None:
        val = min(val, max_val)
    return val


class Edit_GUI(Frame):
    """
    This class defines the interactive GUI interface for manual editing.

    """

    def __init__(self, root):
        """
        Set up the main UI frame.

        """
        Frame.__init__(self, root)
        self.root = root
        self.style = Style()
        self.style.theme_use("default")

        self.pack(fill=BOTH, expand=1)
        self.grid_columnconfigure(3, weight=50)
        self.grid_rowconfigure(1, weight=50)

        # Create a listbox for all positions in selected the experiment
        Label(self, text="Positions").grid(row=0, column=0)
        self.posn_listbox = Listbox(self, exportselection=False)
        self.posn_listbox.grid(row=1, column=0, rowspan=2,
            padx=10, pady=5, sticky=N+S+E+W)
        self.posn_listbox.bind("<<ListboxSelect>>",
            self.on_posn_listbox_select)

        # Create a listbox for all saved traces at the selected position
        Label(self, text="Saved Traces").grid(row=0, column=1)
        self.saved_listbox = Listbox(self, exportselection=False)
        self.saved_listbox.grid(row=1, column=1, rowspan=2,
            padx=10, pady=5, sticky=N+S+E+W)
        self.saved_listbox.bind("<<ListboxSelect>>",
            self.on_saved_listbox_select)

        # Create a listbox for all division events for the selected trace
        Label(self, text="Division Events").grid(row=0, column=2)
        self.divn_listbox = Listbox(self, exportselection=False)
        self.divn_listbox.grid(row=1, column=2, rowspan=2,
            padx=10, pady=5, sticky=N+S+E+W)
        self.divn_listbox.bind("<<ListboxSelect>>",
            self.on_divn_listbox_select)

        # Create that matplotlib figure
        self.fig = Figure()
        ax1 = self.fig.add_subplot(3, 2, (1, 3))
        ax2 = self.fig.add_subplot(3, 2, (2, 4), sharex=ax1, sharey=ax1)
        ax3 = self.fig.add_subplot(3, 2, (5, 6), autoscale_on=False)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=3,
            rowspan=2, sticky=N+S+E+W)
        self.canvas.show()

        # Create that matplotlib toolbar
        frame = Frame(self)
        frame.grid(row=2, column=3)
        self.toolbar = NavigationToolbar2TkAgg(self.canvas, frame)
        self.toolbar.pack(side=LEFT, fill=BOTH, expand=1, anchor=W)

        # Set colors for image RGB display
        self.num_colors = 16
        self.color_name = 'winter'
        self.color_data = np.ones((self.num_colors, 3), dtype=np.uint8)
        for i in range(self.num_colors):
            self.color_data[i, :] = [np.uint8(v * 255) for v in
                cm.get_cmap(self.color_name, self.num_colors)(i)[:3]]


    def init_expt(self, raw_data_dir, analyses_dir, posns=None):
        """
        Get experiment information.

        args:
            input_dir (path): input directory

        kwargs:
            positions (list): list of positions to edit

        """
        self.raw_data_dir = raw_data_dir
        self.analyses_dir = analyses_dir
        self.expt_name = os.path.basename(self.analyses_dir)
        self.root.title(self.expt_name)

        # Get list of valid positions if unspecified
        if posns is None:
            posns = []
            for p in read.listdirs(self.analyses_dir, read.PATTERN['posndir']):
                if os.path.isfile(os.path.join(
                                  self.analyses_dir, p, 'edits.pickle')):
                    posns.append(p)
        if not isinstance(posns, (list, tuple, np.ndarray)):
            raise TypeError('list of positions required.')
        elif not posns:
            raise IOError('no valid positions found in %s' % self.analyses_dir)
        self.posns = sorted(posns)

        # Reset the position listbox
        self.posn_listbox.delete(0, END)
        [self.posn_listbox.insert(END, v) for v in self.posns]

        # Variable for holding temporary area traces
        self.temp = None

        # Save the coordinates of each object and the RGB images
        self.image = [dict.fromkeys(['label', 'trace', 'raw', 'rgb'])
                      for _ in (0, 1)]

        # These values will hold information about each plot
        self.plots = dict.fromkeys(['data', 'divns', 'image'])
        self.plots['image'] = [None, None]
        self.connect()

        # Initialize the GUI
        self.posn_idx = 0
        self.trace_idx = 0
        self.frame_idx = 0
        self.saved_idx = []
        self.divn_idx = []
        self.set_mother = None
        self.set_posn()


    def stop_expt(self):
        """
        Disconnect all bound variables, and save data to file.

        """
        self.disconnect()
        self.save_trace()
        self.save_posn()
        

    def set_posn(self, posn_idx=None):
        """
        Update the position to the new index. If the index is None, force reset the values at the current position.

        kwargs:
            posn_idx (int): index of the current position

        """
        if posn_idx is None:
            # Force execution at same position
            posn_idx, self.posn_idx = self.posn_idx, None
        else:
            # Execute if valid position index is different from current value
            posn_idx = to_bounds(posn_idx, 0, len(self.posns) - 1)
            if posn_idx == self.posn_idx:
                return

        # Save data at the old position then load data for the new position
        if self.posn_idx is not None:
            self.save_posn()
        self.posn_idx, old_posn_idx = posn_idx, self.posn_idx
        try:
            self.load_posn()
        except (IOError, OSError) as e:
            print e
            self.posn_idx = old_posn_idx
            self.load_posn()

        # Set listbox focus to the new position
        self.posn_listbox.select_set(self.posn_idx)
        self.posn_listbox.activate(self.posn_idx)

        # Force reset the trace index (to a saved trace, if possible)
        self.set_trace()


    def load_posn(self):
        """
        Load data for the current position.

        """
        self.posn_dir = os.path.join(self.analyses_dir,
                                     self.posns[self.posn_idx])
        b = os.path.join(self.posn_dir, 'blocks')

        # Read in values for the current position
        self.data_file = os.path.join(self.posn_dir, 'edits.pickle')
        self.data = read_pickle(self.data_file)

        # Read in values from the log file
        log_file = os.path.join(self.posn_dir, 'log.pickle')
        log_data = pickle.load(open(log_file, 'rb'))
        self.img_shape = log_data['image']['phase']['shape']
        self.img_dtype = log_data['image']['phase']['dtype']
        self.pumps = log_data['pumps']

        self.TraceList = list(self.data['Trace'])
        self.SavedList = [v for i, v in self.data['Trace'].iteritems() if
            self.data['Saved'][i]]
        self.num_traces = len(self.TraceList)
        if self.num_traces < 1:
            return
        self.num_frames = len(self.data.ix[0]['Label'])
        self.frames = np.arange(self.num_frames)
        self.time_phase = log_data['phase'][:self.num_frames]
        self.time = self.time_phase / 60
        if log_data.has_key('fluor'):
            max_time = np.max(self.time_phase)
            num_frames_fluor = np.argmin(np.abs(log_data['fluor']-max_time)) + 1
            self.time_fluor = log_data['fluor'][:num_frames_fluor]

        # Unzip phase-contrast image files and read in names of image files
        old_dir = os.curdir
        self.files = [''] * self.num_frames
        for v in read.listdirs(b, read.PATTERN['blockdir']):
            # Extract all .tif images in the input directory
            os.chdir(os.path.join(b, v))
            zipfile.ZipFile('PhaseSegment.zip').extractall()
            for f in read.listfiles('PhaseSegment', read.PATTERN['phasetif']):
                i = read.getframenum(f, read.PATTERN['phasetif'])
                if i < self.num_frames:
                    self.files[i] = os.path.join(b, v, 'PhaseSegment', f)
            os.chdir(old_dir)


    def save_posn(self):
        """
        Save data for the current position.

        """

        # Update the edited file
        self.data.to_pickle(self.data_file)

        # Update the file of saved traces, keeping only relevant variables
        df = self.data[self.data['Saved'] == True]
        if len(df) > 0:
            d = {k : list(df[k]) for k in ('Trace', 'Divns', 'Mother')}

            # Restructure the labels into a DataFrame commensurate with data
            num_saved = len(df)
            num_frames = self.num_frames
            frames = range(num_frames) * num_saved
            traces = []
            for v in xrange(num_saved):
                traces.extend([v] * num_frames)
            index = MultiIndex.from_arrays([traces, frames],
                                           names=('Trace', 'Frame'))
            data = np.hstack(df['Label'].values)
            s = DataFrame(data, index=index, columns=('Label', ))['Label']

            # Remove frames that one should not keep
            d['Label'] = s[np.hstack(df['Keep'].values) & (data > 0)]

            # Save the times directly as arrays
            d['TimeP'] = self.time_phase
            if hasattr(self, 'time_fluor'):
                d['TimeF'] = self.time_fluor

            # Make generations data from divisons
            d['Divns'] = [np.asarray(v) for v in d['Divns']]
            d['Gens'] = [[] for _ in d['Divns']]
            d['Taus'] = [[] for _ in d['Divns']]
            for i, v in enumerate(d['Divns']):
                v += 1
                for j1, j2 in zip(v[:-1], v[1:]):
                    d['Gens'][i].append(slice(j1, j2))
                    t1 = np.mean(d['TimeP'][j1-1:j1+1])
                    t2 = np.mean(d['TimeP'][j2-1:j2+1])
                    d['Taus'][i].append(t2 - t1)
            d['Taus'] = [np.asarray(v) for v in d['Taus']]

            with open(os.path.join(self.posn_dir, 'saved.pickle'), 'wb') as f:
                pickle.dump(d, f)

        # Delete unzipped directories
        b = os.path.join(self.posn_dir, 'blocks')
        for v in read.listdirs(b, read.PATTERN['blockdir']):
            shutil.rmtree(os.path.join(b, v, 'PhaseSegment'))

        # Update the log file
        read.updatelog(self.expt_name, self.posns[self.posn_idx],
                       'edit', self.analyses_dir)


    def set_trace(self, trace_idx=None):
        """
        Update the trace to the new index. If the index is None, force reset the values at the current trace.

        kwargs:
            trace_idx (int): index of the current trace

        """
        if trace_idx is None:
            # Force execution at same trace
            trace_idx, self.trace_idx = self.trace_idx, None
        else:
            # Execute if valid trace is different from current value
            trace_idx = to_bounds(trace_idx, 0, self.num_traces - 1)
            if trace_idx == self.trace_idx:
                return

        # Save data at the old trace then load data for the new trace
        if self.trace_idx is not None:
            self.save_trace()
        self.trace_idx = trace_idx
        self.load_trace()

        # Set new value of saved trace index
        if self.Saved:
            saved_idx = self.SavedList.index(self.Trace)
        else:
            saved_idx = []

        # Reset the saved trace listbox
        self.saved_listbox.delete(0, END)
        [self.saved_listbox.insert(END, str(v)) for v in self.SavedList]
        self.saved_listbox.select_clear(0, END)
        self.set_saved(saved_idx)

        # Clear all plot data
        self.plots['data'] = None

        # Force reset the frame
        self.set_frame()


    def load_trace(self):
        """
        Load data for the current trace.

        """
        for k, v in self.data.ix[self.trace_idx].iteritems():
            setattr(self, k, v)


    def save_trace(self):
        """
        Save data for the current trace.

        """
        pass


    def set_saved(self, saved_idx=[]):
        """
        Update the index of the saved trace.

        kwargs:
            saved_idx (int): index of the current saved trace

        """
        # Do not need to check bounds (always called from set_trace)
        if saved_idx == []:
            return
        self.saved_idx = saved_idx
        self.saved_listbox.select_set(self.saved_idx)
        self.saved_listbox.activate(self.saved_idx)


    def set_frame(self, frame_idx=None):
        """
        Update the frame to the new index. If the index is None, force reset the values at the current frame.

        kwargs:
            frame_idx (int): index of the current frame

        """
        if frame_idx is None:
            # Force execution at same frame
            frame_idx, self.frame_idx = self.frame_idx, None
        else:
            # Execute if valid frame is different from current value
            frame_idx = to_bounds(frame_idx, 0, self.num_frames - 2)
            if frame_idx == self.frame_idx:
                return
        self.frame_idx, old_frame_idx = frame_idx, self.frame_idx

        # Set new value of division index
        if self.frame_idx in self.Divns:
            self.divn_idx = self.Divns.index(self.frame_idx)
        else:
            self.divn_idx = []

        # Reset the division times listbox
        self.divn_listbox.delete(0, END)
        [self.divn_listbox.insert(END, str(v)) for v in self.Divns]
        self.divn_listbox.select_clear(0, END)
        self.set_divn(self.divn_idx)

        # Update image information and plots
        self.update_images(old_frame_idx)
        self.update_plots()


    def set_divn(self, divn_idx=[]):
        """
        Update the index of the division time.

        kwargs:
            saved_idx (int): index of the current division time

        """
        # Do not need to check bounds (always called from set_frame)
        if divn_idx == []:
            return
        self.divn_idx = divn_idx
        self.divn_listbox.select_set(self.divn_idx)
        self.divn_listbox.activate(self.divn_idx)


    def update_images(self, old_frame_idx=None):
        """
        Update the stored image data.

        kwargs:
            old_frame_idx: previous frame index

        """
        image_tmp = [dict.fromkeys(['label', 'trace']) for _ in (0, 1)]

        # Attempt to copy from old data in memory
        for image_idx in (0, 1):
            frame_idx = self.frame_idx + image_idx
            if old_frame_idx is not None:
                for old_image_idx in (0, 1):
                    if frame_idx == old_frame_idx + old_image_idx:
                        image_tmp[image_idx]['label'] = np.asarray(
                            self.image[old_image_idx]['label'])
                        image_tmp[image_idx]['raw'] = np.asarray(
                            self.image[old_image_idx]['raw'])
                        image_tmp[image_idx]['trace'] = list(
                            self.image[old_image_idx]['trace'])

        # Update saved image data
        for image_idx in (0, 1):
            frame_idx = self.frame_idx + image_idx
            if image_tmp[image_idx]['label'] is None:
                # Attempt to read from file
                analyses_file = self.files[frame_idx]
                raw_data_file = os.path.join(self.raw_data_dir,
                                    self.posns[self.posn_idx],
                                    os.path.basename(self.files[frame_idx]))
                try:
                    image_tmp[image_idx]['label'] = imread(analyses_file)
                except IOError:
                    image_tmp[image_idx]['label'] = np.zeros(self.img_shape,
                                                             dtype=np.uint8)
                try:
                    image_tmp[image_idx]['raw'] = imread(raw_data_file)
                except IOError:
                    image_tmp[image_idx]['raw'] = np.zeros(self.img_shape,
                                                           dtype=self.img_dtype)
            if image_tmp[image_idx]['trace'] is None:
                image_tmp[image_idx]['trace'] = [v[frame_idx]
                                                 for v in self.data['Label']]
            self.image[image_idx]['label'] = image_tmp[image_idx]['label']
            self.image[image_idx]['raw'] = image_tmp[image_idx]['raw']
            self.image[image_idx]['trace'] = image_tmp[image_idx]['trace']

            # Colorize the cells based on their identities
            num_cells = self.image[image_idx]['label'].max()
            color_map = np.zeros((num_cells + 1, 3), dtype=np.uint8)

            for k in range(1, num_cells + 1):
                if k in self.image[image_idx]['trace']:
                    trace_idx = self.image[image_idx]['trace'].index(k)
                    if trace_idx == self.trace_idx:
                        if frame_idx - 1 in self.Divns:
                            # Post-division of selected cell is cyan
                            c = [0, 100, 255]
                        elif frame_idx in self.Divns:
                            # Pre-division of selected cell is red
                            c = [255, 0, 0]
                        else:
                            # Selected cell is otherwise yellow
                            c = [255, 255, 0]
                    elif self.TraceList[trace_idx] == self.Mother:
                        # Color the mother purple
                        c = [200, 0, 200]
                    elif self.TraceList[trace_idx] in self.SavedList:
                        # Color the saved cells green
                        #c = self.color_data[np.mod(trace_idx, self.num_colors)]
                        c = [0, 100, 50]
                    else:
                        # Cells are otherwise gray
                        c = [100, 100, 100]
                else:
                    c = [100, 100, 100]
                color_map[k] = c
            self.image[image_idx]['rgb'] = (color_map[self.image[image_idx]['label']])


    def update_plots(self, view_mode=0):
        """
        Update the data plot.

        """

        # Update the images
        for i in range(2):
            j = self.frame_idx + i
            if self.plots['image'][i] is None:
                self.fig.axes[i].clear()
                self.plots['image'][i] = []
                self.plots['image'][i].append(
                    self.fig.axes[i].imshow(self.image[i]['rgb'],
                    vmin=(0, 0, 0), vmax=(255, 255, 255), aspect='equal'))
                self.plots['image'][i].append(
                    self.fig.axes[i].imshow(self.image[i]['raw'],
                    cmap=cm.gray, alpha=0.5))
                self.fig.axes[i].axis('off')
            else:
                self.plots['image'][i][0].set_data(self.image[i]['rgb'])
                self.plots['image'][i][1].set_data(self.image[i]['raw'])
        self.fig.axes[0].set_title('Frame %d' % self.frame_idx)

        # Update the area plot
        x = self.time
        y1 = np.nan * np.ones_like(x)
        y2 = y1.copy()
        y1[self.Keep] = self.Area[self.Keep]
        y2[~self.Keep] = self.Area[~self.Keep]
        x3 = x[self.frame_idx]
        y3 = self.Area[self.frame_idx]
        if view_mode >= 0:
            if view_mode == 0:
                y_min = np.nanmin(y1)
                y_max = np.nanmax(y1)
            elif view_mode == 1:
                y_min = np.nanmin(np.hstack((y1, y2)))
                y_max = np.nanmax(np.hstack((y1, y2)))
            y_diff = max(1, np.abs(y_max - y_min))
            y_min -= 1e-2 * y_diff
            y_max += 1e-2 * y_diff
        else:
            y_min, y_max = self.fig.axes[2].get_ylim()
        if self.plots['data'] is None:
            self.fig.axes[2].clear()
            self.plots['data'] = self.fig.axes[2].plot(x, y1, 'b.')
            self.plots['data'].extend(self.fig.axes[2].plot(x, y2, 'r.'))
            self.plots['data'].extend(self.fig.axes[2].plot(
                x3, y3, 'g+', ms=10, mew=3, alpha=0.8))
            self.plots['divns'] = self.fig.axes[2].vlines(
                self.time[self.Divns], y_min, y_max)
            if view_mode >= 0:
                self.fig.axes[2].set_xlim((min(self.time), max(self.time)))
                self.fig.axes[2].set_ylim(y_min, y_max)

            # Plot the pumps data
            y1, y2 = 0.95 * min(self.Area), 1.05 * max(self.Area)
            for i, d in enumerate(self.pumps):
                if self.pumps[0]['Solution'] == d['Solution']: continue
                c = cm.get_cmap('gray_r')(float(i) / len(self.pumps), 1)
                for r, (x1, x2) in zip(d['Rate'], d['Time']):
                    if r == 0: continue
                    x = np.asarray([x1, x2, x2, x1]) / 60
                    y = np.asarray([y1, y1, y2, y2])
                    self.fig.axes[2].fill(x, y, color=c, alpha=0.2, linewidth=0)
        else:
            self.plots['data'][0].set_ydata(y1)
            self.plots['data'][1].set_ydata(y2)
            self.plots['data'][2].set_xdata(x3)
            self.plots['data'][2].set_ydata(y3)
            vs = [[[v, y_min], [v, y_max]] for v in self.time[self.Divns]]
            self.plots['divns'].set_segments(vs)
        if self.Saved:
            c = 'yellow'
        else:
            c = 'black'
        self.fig.axes[2].set_ylabel(r'Area (px$^{2}$)')
        self.fig.axes[2].set_xlabel(r'Time (min)')
        self.fig.axes[2].set_title(r'Trace (%d, %d)' % self.Trace, color=c)
        self.canvas.draw()


    def connect(self):
        """
        Connect to all matplotlib events.

        """
        self.cid_button_release = self.fig.canvas.mpl_connect(
            'button_release_event', self.on_button_release)
        self.cid_key_release = self.fig.canvas.mpl_connect(
            'key_release_event', self.on_key_release)
        self.cid_button_press = self.fig.canvas.mpl_connect(
            'button_press_event',
            lambda event : self.fig.canvas._tkcanvas.focus_set())


    def disconnect(self):
        """
        Disconnect from the stored matplotlib connection IDs.

        """
        self.fig.canvas.mpl_disconnect(self.cid_button_release)
        self.fig.canvas.mpl_disconnect(self.cid_key_release)
        self.fig.canvas.mpl_disconnect(self.cid_button_press)


    def on_posn_listbox_select(self, val):
        """
        Respond to selection in position listbox.

        """
        sel = val.widget.curselection()
        if not sel:
            return
        posn_idx = int(sel[0])
        self.set_posn(posn_idx)


    def on_saved_listbox_select(self, val):
        """
        Respond to selection in saved trace listbox.

        """
        sel = val.widget.curselection()
        if not sel:
            val.widget.curselection()
        saved_idx = int(sel[0])
        trace_idx = self.TraceList.index(self.SavedList[saved_idx])
        self.set_trace(trace_idx)


    def on_divn_listbox_select(self, val):
        """
        Respond to selection in division time listbox.

        """
        sel = val.widget.curselection()
        if not sel:
            return
        divn_idx = int(sel[0])
        frame_idx = self.Divns[divn_idx]
        self.set_frame(frame_idx)


    def on_button_release(self, event):
        """
        Trigger on mouse button release over axes.

        """
        if event.inaxes:
            i = event.inaxes.get_geometry()[2] - 1
            r = int(event.ydata)
            c = int(event.xdata)
            if i in [0, 1]:
                # Set new trace if click on cell
                try:
                    v = self.image[i]['label'][r, c]
                    if v > 0:
                        k = self.image[i]['trace'].index(v)
                        if k:
                            # Set the new trace index
                            self.set_trace(k)

                            # Set the identity of the mother
                            if self.set_mother is not None:
                                if self.set_mother == self.Trace:
                                    self.Mother = None
                                else:
                                    self.Mother = self.set_mother
                                self.data['Mother'].ix[k] = self.Mother
                                self.set_mother = None
                                self.set_frame()
                except (IndexError, ValueError):
                    return
            else:
                # Set new frame based on (x, y) click, rescaled to same units
                rescale = lambda val, lims: (val - lims[0]) / np.diff(lims)

                x_lim = self.fig.axes[2].get_xlim()
                xclick = rescale(event.xdata, x_lim)
                xdata = rescale(self.time, x_lim)

                y_lim = self.fig.axes[2].get_ylim()
                yclick = rescale(event.ydata, y_lim)
                ydata = rescale(self.Area, y_lim)

                diffs = (xclick - xdata) ** 2 + (yclick - ydata) ** 2
                isfinite = np.isfinite(diffs)
                min_idx = np.argmin(diffs[isfinite])
                frame_idx = self.frames[isfinite][min_idx]
                self.set_frame(frame_idx)


    def on_key_release(self, event):
        """
        Trigger on key release.

        """
        if event.key in ['H', 'h']:
            # Update the plots
            self.plots['data'] = None
            view_mode = 0 if event.key == 'h' else 1
            self.update_plots(view_mode=view_mode)
        elif event.key in ['S', 's']:
            # Save results back to file
            self.save_trace()
        elif event.key in ['D', 'd']:
            # Add/remove division time at current frame
            if self.frame_idx in self.Divns:
                del self.Divns[self.Divns.index(self.frame_idx)]
            else:
                self.Divns.append(self.frame_idx)
                self.Divns.sort()
            self.set_frame()
        elif event.key in ['A', 'a']:
            # Remove all division times in field of view
            x_min, x_max = self.fig.axes[2].get_xlim()
            for v in np.flatnonzero((x_min <= self.time) &
                                    (self.time <= x_max)):
                if v in self.Divns:
                    del self.Divns[self.Divns.index(v)]
            self.set_frame()
        elif event.key in ['T', 't']:
            # Add/remove current cell to/from saved list
            if self.Saved:
                self.Saved = False
                del self.SavedList[self.saved_idx]
            else:
                self.Saved = True
                self.SavedList.append(self.Trace)
                self.SavedList.sort()
            self.data['Saved'].ix[self.trace_idx] = self.Saved
            self.set_trace()
        elif event.key in ['M', 'm']:
            # Set the identity of the mother in memory
            self.set_mother = self.Trace
        elif event.key in ['X', 'x']:
            # Only copy points with Keep = True and with valid size
            x_min, x_max = self.fig.axes[2].get_xlim()
            js = np.flatnonzero((x_min <= self.time) & (self.time <= x_max))
            self.temp = {}
            self.temp['js'] = js[self.Keep[js] & np.isfinite(self.Area[js])]
            self.temp['i'] = self.trace_idx
        elif event.key in ['V', 'v']:
            if self.temp is None:
                return

            # Do not replace any points with Keep = True or valid size
            js = self.temp['js']
            ks = js[~self.Keep[js] | ~np.isfinite(self.Area[js])]
            if not np.any(ks):
                return

            # Swap points from the selected trace
            i = self.temp['i']
            oldArea = self.data.ix[i]['Area']
            oldKeep = self.data.ix[i]['Keep']
            oldLabel = self.data.ix[i]['Label']
            oldDivns = self.data.ix[i]['Divns']

            self.Area[ks], oldArea[ks] = oldArea[ks], self.Area[ks]
            self.Keep[ks], oldKeep[ks] = oldKeep[ks], self.Keep[ks]
            self.Label[ks], oldLabel[ks] = oldLabel[ks], self.Label[ks]

            # Insert the division events
            for k in ks:
                if k in oldDivns and k not in self.Divns:
                    self.Divns.append(k)
                elif k in self.Divns and k not in oldDivns:
                    oldDivns.append(k)
            oldDivns.sort()
            self.Divns.sort()
            self.temp = None
            self.set_frame()
        elif event.key in ['R', 'r', 'B', 'b']:
            if event.key in ['R', 'B']:
                x_min, x_max = self.fig.axes[2].get_xlim()
                y_min, y_max = self.fig.axes[2].get_ylim()
                js = [i for i in np.flatnonzero((x_min <= self.time) &
                      (self.time <= x_max)) if y_min < self.Area[i] < y_max]
            else:
                js = self.frame_idx
            if event.key in ['R', 'r']:
                self.Keep[js] = False
            elif event.key in ['B', 'b']:
                self.Keep[js] = True
            self.update_plots(-1)
        elif event.key == 'left':
            # Go backward one frame
            self.set_frame(self.frame_idx - 1)
        elif event.key == 'right':
            # Go forward one frame
            self.set_frame(self.frame_idx + 1)
        elif event.key in ['down', 'up']:
            if not self.Divns:
                return
            v = set(self.Divns)
            v.add(self.frame_idx)
            v = sorted(v)
            k = v.index(self.frame_idx)
            if event.key == 'down':
                # Go backward one division
                if k > 0:
                    frame_idx = v[k - 1]
                    self.set_frame(frame_idx)
            elif event.key == 'up':
                # Go forward one division
                if k < len(v) - 1:
                    frame_idx = v[k + 1]
                    self.set_frame(frame_idx)
        elif event.key in ['<', ',', '>', '.']:
            if not self.SavedList:
                return
            v = set(self.SavedList)
            v.add(self.Trace)
            v = sorted(v)
            k = v.index(self.Trace)
            if event.key in ['<', ',']:
                # Go backward one division
                if k > 0:
                    trace = v[k - 1]
                    self.set_trace(self.TraceList.index(trace))
            elif event.key in ['>', '.']:
                # Go forward one division
                if k < len(v) - 1:
                    trace = v[k + 1]
                    self.set_trace(self.TraceList.index(trace))
        elif event.key in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
            if event.key == '0':
                level = 1.
            else:
                level = float(event.key) / 10.

            # Clean the data based on the percent change over time
            s = Series(data=self.Area, index=self.frames)
            if not self.Divns:
                return
            for j1, j2 in zip(self.Divns[:-1], self.Divns[1:]):
                g = slice(j1+1, j2+1)

                v = np.log(s[g]).pct_change()
                k = np.flatnonzero(np.abs(v) > level)
                self.Keep[g][k] = False
            self.update_plots(-1)
        else:
            key_press_handler(event, self.canvas, self.toolbar)


def main(expt_raw_data_dir, expt_analyses_dir, positions):
    """
    Run interactive editing GUI.

    args:
        input_dir (path): input directory

    kwargs:
        positions (list): list of positions to edit

    """
    root = Tk()
    root.geometry("1200x600+50+50")
    app = Edit_GUI(root)
    app.init_expt(expt_raw_data_dir, expt_analyses_dir, positions)
    root.mainloop()
    app.stop_expt()
