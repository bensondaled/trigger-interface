"""
Ben Deverett 2014
To run: >pythonw gui2.py
"""

#natives
import json
import os
import time as pytime

#numpy, scipy, matplotlib
import numpy as np 
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.cm as mpl_cm
from matplotlib.figure import Figure
from matplotlib import path as mpl_path
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

#opencv
import cv2
cv = cv2.cv

#custom
from core.daq import DAQ, Trigger
from core.cameras import Camera

# interface
import wx
import threading

class Experiment(wx.Frame):
    def __init__(self, parent=None, title='Interface', **kwargs):
        wx.Frame.__init__(self, parent, title=title, size=(50,100), pos=(0,30))

        self.init_variables(**kwargs)
        self.init_expt() 
        self.init_view_options()
        self.init_interface()

        self.Show(True)
        self.step()
    def init_variables(self, trigger_cycle=[], movement_frame_buffer=10, n_trials=99999999.):
        
        # time-related
        self.start_time = pytime.time()
        self.refresh_rate = 1 #ms
        self.begun = False

        # trial-related
        self.trigger_cycle=trigger_cycle

        # mask-related
        self.mask_names = ['WHEEL', 'EYE']
        self.masks = {}
        self.mask_idxs = {}
        self.mask_pts = {}
        self.movement_frame_buffer = movement_frame_buffer

        # any variables that will interact with the GUI
        self.pause = Variable('Pause', 'pause', minn=False, maxx=True, default=False, edit='toggle', dtype=bool)
        self.recording = Variable('Record', 'record', minn=False, maxx=True, default=False, edit=None, dtype=bool)
        self.movement_threshold = Variable('Movement Threshold', 'move_thresh', minn=0., maxx=255., default=60., edit='slider', dtype=int)
        self.eyelid_threshold = Variable('Eyelid Threshold', 'eye_thresh', minn=0., maxx=255., default=50., edit='slider', dtype=int)
        self.wheel_stretch = Variable('Wheel Stretch', 'wheel_stretch', minn=0., maxx=255., default=1., edit='slider', dtype=int)
        self.wheel_shift = Variable('Wheel Shift', 'wheel_shift', minn=0., maxx=255., default=0., edit='slider', dtype=int)
        self.inter_trial_min = Variable('Inter-trial Minimum', 'itm', minn=0., maxx=100., default=10., edit='slider', dtype=int)
        self.expt_name = Variable('Experiment Name', 'expt_name', minn='', maxx='', default='(enter name to begin)', edit='textbox', dtype=str)
        self.elapsed = Variable('Elapsed', 'elapsed', minn=0., maxx=99999999999., default=0., edit=None, dtype=float)
        self.plot_size = Variable('Plot Size', 'plot_size', minn=1., maxx=500., default=50., edit=None, dtype=int)
        self.n_trials = Variable('Total # of Trials', 'n_trials', minn=1., maxx=99999999., default=n_trials, edit='textbox', dtype=int)
        self.trial_number = Variable('Trial #', 'trial', minn=0, maxx=99999999., default=1, edit=None, dtype=int)

        self.variables = [self.pause, self.recording, self.movement_threshold, self.eyelid_threshold, self.inter_trial_min, self.expt_name, self.elapsed, self.plot_size, self.n_trials, self.trial_number, self.wheel_stretch, self.wheel_shift]
    def init_expt(self):
        
        # setup camera
        self.camera = Camera()
        
        # setup daq
        self.daq = DAQ(mode=DAQ.DIGITAL)
        self.analog_daq = DAQ(mode=DAQ.ANALOG)

    def init_view_options(self):
        # used only once to initalize display
        self.statuses = [self.elapsed, self.trial_number]
        self.sliders = [v for v in self.variables if v.edit=='slider']
        self.textboxes = [v for v in self.variables if v.edit=='textbox']
    def init_interface(self):
        self.init_menus()
        self.init_fonts()
        self.init_layout()
    def init_menus(self):
        self.CreateStatusBar()
        # build menus
        file_menu= wx.Menu()
        menu_item_about = file_menu.Append(wx.ID_ABOUT, "About","About this software")
        self.Bind(wx.EVT_MENU, self.on_about, menu_item_about)
        menu_item_exit = file_menu.Append(wx.ID_EXIT,"Exit","Exit")
        self.Bind(wx.EVT_MENU, self.on_exit, menu_item_exit)
        self.Bind(wx.EVT_CLOSE, self.on_exit)
        # assemble menu bar
        menuBar = wx.MenuBar()
        menuBar.Append(file_menu, 'File')
        self.SetMenuBar(menuBar)
    def init_fonts(self):
        fsize = 12
        self.FONT_STATUS_NAME = wx.Font(fsize, wx.MODERN, wx.NORMAL, wx.BOLD)
        self.FONT_STATUS_VALUE = wx.Font(fsize, wx.MODERN, wx.NORMAL, wx.NORMAL)
        self.FONT_SLIDER_NAME = wx.Font(fsize, wx.MODERN, wx.NORMAL, wx.NORMAL)
    def init_layout(self):
        szr_main = wx.FlexGridSizer(rows=2, cols=2, hgap=5, vgap=5)

        szr_ctrl = wx.GridSizer(rows=3, cols=1, vgap=2)
        # status area
        szr_status = wx.GridSizer(rows=len(self.statuses), cols=2, hgap=1, vgap=1)
        for st in self.statuses:
            w_name = wx.StaticText(parent=self, label=st.name, style=wx.ALIGN_LEFT)
            w_name.SetFont(self.FONT_STATUS_NAME)
            w_val = wx.StaticText(parent=self, style=wx.ALIGN_LEFT)
            w_val.SetLabel(str(st.v))
            w_val.SetFont(self.FONT_STATUS_VALUE)
            szr_status.Add(w_name, border=1)
            szr_status.Add(w_val, border=1)
            self.status_timer(w_val, st)()
        szr_ctrl.Add(szr_status, flag=wx.ALIGN_LEFT)
        
        # boxes area
        szr_textboxes = wx.GridSizer(rows=len(self.textboxes)*2, cols=1, hgap=0, vgap=10)
        for tb in self.textboxes:
            w_box = wx.TextCtrl(parent=self, style=wx.TE_PROCESS_ENTER)
            w_box.SetValue(str(tb.v))
            w_box.Bind(wx.EVT_TEXT_ENTER, self.on_textbox_edit(tb))
            if tb == self.expt_name:
                w_box.Bind(wx.EVT_KILL_FOCUS, self.textbox_cancel(tb))
            else:
                w_box.Bind(wx.EVT_KILL_FOCUS, self.on_textbox_edit(tb))
            w_name = wx.StaticText(parent=self, label=tb.name, style=wx.ALIGN_LEFT)
            w_name.SetFont(self.FONT_SLIDER_NAME)
            szr_textboxes.Add(w_box, flag=wx.EXPAND, border=1)
            szr_textboxes.Add(w_name, flag=wx.EXPAND, border=1)
        szr_ctrl.Add(szr_textboxes, flag=wx.ALIGN_LEFT)
        
        # buttons area
        szr_buttons = wx.GridSizer(rows=3, cols=1, hgap=0, vgap=10)
        
        self.w_pause = wx.Button(parent=self, label='Pause (ctrl+p)')
        self.w_pause.Bind(wx.EVT_BUTTON, self.on_pause)
        szr_buttons.Add(self.w_pause, flag=wx.EXPAND, border=1)
        
        w_trigger = wx.Button(parent=self, label='Trigger (ctrl+t)')
        w_trigger.Bind(wx.EVT_BUTTON, self.on_trigger)
        szr_buttons.Add(w_trigger, flag=wx.EXPAND, border=1)
        
        w_redo = wx.Button(parent=self, label='Redo (ctrl+r)')
        w_redo.Bind(wx.EVT_BUTTON, self.on_redo)
        szr_buttons.Add(w_redo, flag=wx.EXPAND, border=1)

        szr_ctrl.Add(szr_buttons, flag=wx.ALIGN_LEFT)

        szr_main.Add(szr_ctrl, flag=wx.EXPAND)

        # camera area
        self.szr_camera = wx.GridSizer(rows=1, cols=1)
        self.panel_camera = CameraPanel(parent=self, cam=self.camera)
        self.szr_camera.Add(self.panel_camera, flag=wx.EXPAND)
        szr_main.Add(self.szr_camera, flag=wx.ALIGN_RIGHT)

        # sliders area
        szr_sliders = wx.GridSizer(rows=len(self.sliders)*2, cols=1, hgap=0, vgap=10)
        for sl in self.sliders:
            w_slider = wx.Slider(parent=self, minValue=sl.minn, maxValue=sl.maxx, style=wx.SL_VALUE_LABEL)
            w_slider.SetValue(int(sl.v))
            w_slider.Bind(wx.EVT_SLIDER, self.on_slider_edit(sl))
            w_name = wx.StaticText(parent=self, label=sl.name, style=wx.ALIGN_LEFT)
            w_name.SetFont(self.FONT_SLIDER_NAME)
            szr_sliders.Add(w_slider, flag=wx.EXPAND, border=1)
            szr_sliders.Add(w_name, flag=wx.EXPAND, border=1)
        szr_main.Add(szr_sliders, flag=wx.EXPAND)


        # plot area
        self.panel_plot = PlotPanel(parent=self)
        szr_main.Add(self.panel_plot, flag=wx.EXPAND)

        # abstract area
        self.Bind(wx.EVT_CHAR_HOOK, self.on_key)

        self.szr_main = szr_main
        self.SetSizerAndFit(szr_main)
    
    # Callbacks
    def on_about(self, event):
        dlg = wx.MessageDialog( self, "C Ben Deverett 2014", "About", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()
    def on_exit(self, event):
        self.camera.release()
        wx.GetApp().Exit()
    def on_slider_edit(self, variable):
        def in_event(event):
            o = event.GetEventObject()
            variable.v = o.GetValue()

            if variable==self.movement_threshold:
                self.panel_plot.update_line('WHEEL', variable.v)
            elif variable==self.eyelid_threshold:
                self.panel_plot.update_line('EYE', variable.v)

        return in_event
    def on_textbox_edit(self, variable):
        def in_event(event):
            o = event.GetEventObject()
            variable.v = o.GetValue()

            if variable==self.expt_name:
                self.pause.v = False
                self.new_session()
        return in_event
    def textbox_cancel(self, variable):
        def in_event(event):
            o = event.GetEventObject()
            o.SetValue(variable.v)
        return in_event
    def on_pause(self, event):
        self.pause.v = not self.pause.v
        if self.pause.v:
            self.w_pause.SetLabel('Go (ctrl+g)')
        else:
            self.w_pause.SetLabel('Pause (ctrl+p)')
    def on_trigger(self, event):
        if self.begun:
            self.pause.v = False
            self.send_trigger()
            self.start_recording()
    def on_redo(self, event):
        if self.begun:
            self.trigger_cycle.redo()
            self.trial_number.v -= 1
    def status_timer(self, widg, variable):
        def on_tick():
            widg.SetLabel(str(variable.v))
            wx.CallLater(self.refresh_rate, self.status_timer(widg, variable))
        return on_tick
    def on_key(self, event):
        key = event.GetKeyCode()
        ctrl = event.ControlDown()
        if not ctrl:
            event.Skip()
        else:
            if key in [ord('q'), ord('Q')]:
                self.on_exit(None)
            elif key in [ord('r'), ord('R')]:
                self.on_redo(None)
            elif key in [ord('t'), ord('T')]:
                self.on_trigger(None)
            elif key in [ord('p'), ord('P'), ord('g'), ord('G')]:
                self.on_pause(None)
    
    # experiment stuff
    def new_session(self):
        if not os.path.isdir(self.expt_name.v):
            os.mkdir(self.expt_name.v)
            self.trial_number.v = 1

        self.reset_monitors()
        self.recording.v = False
        self.pause.v = False
        self.start_time = pytime.time()

        # set masks
        if len(self.masks)==0:     
            self.set_masks()
        self.save_masks()

        self.panel_plot.reset()
        plot_names = self.mask_names
        plot_colors = ['r','b']
        plot_threshs = [self.movement_threshold.v, self.eyelid_threshold.v]
        for c,m,th in zip(plot_colors, plot_names, plot_threshs):
            self.panel_plot.new_plot(m, c=c)
            self.panel_plot.new_line(m, th, c=c)

        self.begun = True
    def save_masks(self):
        np.save(os.path.join(self.expt_name.v,'masks'), np.atleast_1d([self.masks]))
    def set_masks(self):
        masks = {}
        fig = Figure()
        axes = fig.add_subplot(111)
        for m in self.mask_names:
            frame, timestamp = self.camera.read()
            #axes.set_title("Select mask: %s."%m)
            #axes.imshow(frame, cmap=mpl_cm.Greys_r)
            pts = []
            #while not len(pts):
            #    pts = fig.ginput(0)
            #axes.clear()
            pts = [[1,1],[50,30],[90,120],[200,350]]#TEMPORARY
            path = mpl_path.Path(pts)
            mask = np.ones(np.shape(frame), dtype=bool)
            for ridx,row in enumerate(mask):
                for cidx,pt in enumerate(row):
                    if path.contains_point([cidx, ridx]):
                        mask[ridx,cidx] = False
            self.mask_pts[m] = np.array(pts, dtype=np.int32)
            masks[m] = mask
            self.mask_idxs[m] = np.where(mask==False)
        self.masks = masks
        
    def reset_monitors(self):
        self.monitor_img_set = np.empty((self.camera.resolution[1],self.camera.resolution[0],self.movement_frame_buffer))
        self.monitor_img_set[:] = None
        self.monitor_vals = {m:np.empty(self.plot_size.v) for m in self.mask_names}
        for m in self.monitor_vals:
            self.monitor_vals[m][:] = None
    def monitor_frame(self, frame, masks=('WHEEL', 'EYE')):
        if 'WHEEL' in masks:
            if None in self.monitor_img_set:
                return 
            self.monitor_img_set = np.roll(self.monitor_img_set, 1, axis=2)
            self.monitor_img_set[:,:,0] = frame
            pts = self.monitor_img_set[self.mask_idxs['WHEEL'][0],self.mask_idxs['WHEEL'][1],:]
            std_pts = np.std(pts, axis=1)
            wval = np.mean(std_pts) * self.wheel_stretch.v + self.wheel_shift.v
            self.monitor_vals['WHEEL'] = np.roll(self.monitor_vals['WHEEL'], -1)
            self.monitor_vals['WHEEL'][-1] = wval
        if 'EYE' in masks:
            pts = frame[self.mask_idxs['EYE'][0],self.mask_idxs['EYE'][1]]
            eyval = np.mean(pts)
            self.monitor_vals['EYE'] = np.roll(self.monitor_vals['EYE'], -1)
            self.monitor_vals['EYE'][-1] = eyval
            self.update_analog_daq()

        if not self.recording.v:
            self.update_plots()
    def update_plots(self):
        for mv in self.monitor_vals:
            self.panel_plot.update_plot(mv, self.monitor_vals[mv])
    def update_analog_daq(self):
        if self.monitor_vals['EYE'][-1] != None:
            val = self.monitor_vals['EYE'][-1]
            val = self.normalize(val, oldmin=0., oldmax=255., newmin=self.analog_daq.minn, newmax=self.analog_daq.maxx)          
            tr = Trigger(msg=val)
            self.analog_daq.trigger(tr)
    def normalize(self, val, oldmin, oldmax, newmin, newmax):
        return ((val-oldmin)/oldmax) * (newmax-newmin) + newmin
    def query_for_trigger(self):
        if pytime.time()-self.start_time < self.inter_trial_min.v:
            return False
        return self.monitor_vals['WHEEL'][-1] < self.movement_threshold.v
    def send_trigger(self):
        self.daq.trigger(self.trigger_cycle.next)
        print "Sent trigger #%i"%(self.trial_number.v+1)
    def start_recording(self):
        self.recording.v = True
        self.start_time = pytime.time()
        self.trial_number.v += 1

        name = self.expt_name.v
        trial = self.trial_number.v
        
        self.filename = os.path.join(name,'trial%i'%(trial))
        if os.path.isfile(self.filename):
            i = 1
            while os.path.isfile(os.path.join(name,'trial%i_redo%i'%(trial,i))):
                i += 1
            self.filename = os.path.join(name,'trial%i_redo%i.npz'%(trial,i))
        
        self.writer = cv2.VideoWriter(self.filename+'.avi',0,30,frameSize=self.camera.resolution,isColor=self.camera.color_mode)
        self.time = []
        self.reset_monitors()
    def end_trial(self):
        self.recording.v = False
        self.start_time = pytime.time()
        np.savez_compressed(self.filename+'.npz', time=self.time)         
        self.writer.release()
        self.filename = None

        if self.trial_number.v == self.n_trials.v:
            self.on_exit(None)
    def next_frame(self):
        frame, timestamp = self.camera.read()
        cv2.polylines(frame, [self.mask_pts[m] for m in self.mask_names if m in self.masks.keys()], 1, (255,255,255), thickness=2)
        self.panel_camera.update_image(frame)

        if not self.begun:
            return

        if self.recording.v:
            self.writer.write(frame)
            self.time.append(timestamp)
            if pytime.time()-self.start_time >= self.trigger_cycle.current.duration:
                self.end_trial()
        
        elif not self.recording.v:
            if not self.pause.v:
                if self.query_for_trigger():
                    self.send_trigger()
                    self.start_recording()

            self.monitor_frame(frame, masks=('WHEEL','EYE'))

    def step(self):
        self.elapsed.v = round(pytime.time() - self.start_time, 3)
        self.next_frame()

        wx.CallLater(self.refresh_rate, self.step)

class Variable(object):
    def __init__(self, name, short, minn=0., maxx=999999., default=0., edit=None, dtype=str):
        self.name = name
        self.short = short
        self.minn = minn
        self.maxx = maxx
        self.default = default
        self.edit = edit #button, textbox, slider, None
        self.dtype = dtype

        self._value = self.dtype(self.default)
        self.v = self._value
    @property
    def v(self):
        return self._value
    @v.setter
    def v(self, new):
        self._value = self.dtype(new)
        if self._value > self.maxx:
            self._value = self.maxx
        if self._value < self.minn:
            self._value = self.minn
        self._value = self.dtype(new)

class CameraPanel(wx.Panel):
    def __init__(self, cam, parent=None):
        wx.Panel.__init__(self, parent, size=cam.resolution)
        self.parent = parent

        self.cam = cam
        self.width = self.cam.resolution[0]
        self.height = self.cam.resolution[1]
        self.img = wx.BitmapFromBuffer(self.width, self.height, np.zeros(self.cam.resolution))

        self.Bind(wx.EVT_PAINT, self.on_paint)
    def on_paint(self, event):
        dc = wx.BufferedPaintDC(self)
        dc.DrawBitmap(self.img, 0, 0)
    def update_image(self, frame):
        self.img.CopyFromBuffer(cv2.cvtColor(frame,cv.CV_GRAY2BGR))
        self.Refresh()

class PlotPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.parent = parent
        self.fig = Figure()
        self.axes = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.fig)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 1, wx.GROW)
        self.SetSizer(self.sizer)
        self.Fit()

        self.reset()
    @property
    def size(self):
        return self.parent.plot_size.v
    def new_plot(self, name, c):
        self.plots[name] = self.axes.plot(np.zeros(self.size), c)[0] 
        self.axes.set_ylim([-1, 255])
    def new_line(self, name, val, c):
        self.lines[name] = self.axes.plot(np.repeat(val, self.size), c+'--')[0]
    def update_line(self, name, val):
        self.lines[name].set_ydata(np.repeat(val, self.size))
        self.canvas.draw()
    def reset(self):
        self.plots = {}
        self.lines = {}
        saelf.aces.clear()
    def update_plot(self, name, data):
        if len(data) != self.size:
            data = np.append(data, np.repeat(None, self.size-len(data)))
        self.plots[name].set_ydata(data)
        self.canvas.draw()

class TriggerCycle(object):
    def __init__(self, triggers=[]):
        self.triggers = np.array(triggers)
        self.current = Trigger(msg=[0,0,0,0], duration=0.0, name='(no trigger yet)')
    @property
    def next(self):
        n = self.triggers[0]
        self.current = n
        self.triggers = np.roll(self.triggers, -1)
        return n
    def redo(self):
        self.triggers = np.roll(self.triggers, 1)
        self.current = self.triggers[-1]
    def metadata(self):
        md = {}
        md['triggers'] = [t.metadata() for t in self.triggers]
        return md
    def __str__(self):
        return str([t.name for t in self.triggers])

if __name__ == '__main__':
    app = wx.App(False)
    CS = Trigger([0,0,0,0], name='CS', duration=5.0)
    US = Trigger([0,0,0,0], name='US', duration=5.0)
    tc = TriggerCycle([CS, US, CS, CS, CS, CS])
    e = Experiment(trigger_cycle=tc)
    app.MainLoop()

