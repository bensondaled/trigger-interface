#natives
import json
import os
import time as pytime

#numpy, scipy, matplotlib
import numpy as np 
import matplotlib
matplotlib.use('WXAgg')
import pylab as pl
import matplotlib.cm as mpl_cm
from matplotlib import path as mpl_path
from matplotlib.figure import Figure
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

class MainFrame(wx.Frame):
    def __init__(self, experiment, parent=None, title='Interface'):
        self.exp = experiment

        wx.Frame.__init__(self, parent, title=title, size=(50,100), pos=(0,30))
        
        self.init_menus()
        self.init_fonts()
        self.init_layout()

        self.main_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.step)
        self.main_timer.Start(1)

        self.Show(True)
    def init_fonts(self):
        fsize = 12
        self.FONT_STATUS_NAME = wx.Font(fsize, wx.MODERN, wx.NORMAL, wx.BOLD)
        self.FONT_STATUS_VALUE = wx.Font(fsize, wx.MODERN, wx.NORMAL, wx.NORMAL)
        self.FONT_SLIDER_NAME = wx.Font(fsize, wx.MODERN, wx.NORMAL, wx.NORMAL)

    def init_layout(self):
        sizer_main = wx.GridSizer(rows=2, cols=3, hgap=5, vgap=5)

        # status area
        sizer_status = wx.GridSizer(rows=len(self.exp.statuses), cols=2, hgap=10, vgap=5)
        self.statuses = {}
        for st in self.exp.statuses:
            st = self.exp.statuses[st]
            if not st.show:
                continue
            name = wx.StaticText(parent=self, label=st.name, style=wx.ALIGN_RIGHT)
            name.SetFont(self.FONT_STATUS_NAME)
            val = wx.TextCtrl(parent=self, style=wx.ALIGN_LEFT)
            val.SetValue(str(st.value))
            val.SetFont(self.FONT_STATUS_VALUE)
            val.SetEditable(False)
            self.statuses[st.short] = val
            sizer_status.Add(name, flag=wx.EXPAND, border=1)
            sizer_status.Add(val, flag=wx.EXPAND, border=1)
        sizer_main.Add(sizer_status, flag=wx.EXPAND)
        
        # variables area (sliders)
        sizer_sliders = wx.GridSizer(rows=self.exp.n_sliders_for_interface*2, cols=1, hgap=0, vgap=10)
        self.sliders = {}
        for sl in self.exp.variables:
            sl = self.exp.variables[sl]
            if not sl.show or sl.style!='slider':
                continue
            slider = wx.Slider(parent=self, minValue=sl.minn, maxValue=sl.maxx, style=wx.SL_VALUE_LABEL)
            slider.SetValue(int(sl.value))
            slider.Bind(wx.EVT_SLIDER, self.on_variable_edit(sl.short))
            self.sliders[sl.short] = slider
            name = wx.StaticText(parent=self, label=sl.name, style=wx.ALIGN_LEFT)
            name.SetFont(self.FONT_SLIDER_NAME)
            sizer_sliders.Add(slider, flag=wx.EXPAND, border=1)
            sizer_sliders.Add(name, flag=wx.EXPAND, border=1)
        sizer_main.Add(sizer_sliders, flag=wx.EXPAND)
        # variables area (boxes)
        sizer_boxes = wx.GridSizer(rows=self.exp.n_boxes_for_interface*2, cols=1, hgap=0, vgap=10)
        self.boxes = {}
        for sl in self.exp.variables:
            sl = self.exp.variables[sl]
            if not sl.show or sl.style!='box':
                continue
            box = wx.TextCtrl(parent=self, style=wx.TE_PROCESS_ENTER)
            box.SetValue(str(sl.value))
            box.Bind(wx.EVT_TEXT_ENTER, self.on_variable_edit(sl.short))
            self.boxes[sl.short] = box
            name = wx.StaticText(parent=self, label=sl.name, style=wx.ALIGN_LEFT)
            name.SetFont(self.FONT_SLIDER_NAME)
            sizer_boxes.Add(box, flag=wx.EXPAND, border=1)
            sizer_boxes.Add(name, flag=wx.EXPAND, border=1)
        sizer_main.Add(sizer_boxes, flag=wx.EXPAND)
        
        # buttons area
        sizer_buttons = wx.GridSizer(rows=len(self.exp.buttons), cols=1, hgap=0, vgap=10)
        for idx,but in enumerate(self.exp.buttons):
            button = wx.Button(parent=self, label=but)
            button.Bind(wx.EVT_BUTTON, self.on_button(but))
            sizer_buttons.Add(button, flag=wx.EXPAND, border=1)
        sizer_main.Add(sizer_buttons, flag=wx.EXPAND)

        # camera area
        sizer_camera = wx.GridSizer(rows=1, cols=1)
        self.panel_camera = CameraPanel(parent=self, cam=self.exp.camera)
        sizer_camera.Add(self.panel_camera, flag=wx.EXPAND)
        sizer_main.Add(sizer_camera, flag=wx.EXPAND)

        # plot area
        sizer_plot = wx.GridSizer(rows=1, cols=1)
        self.panel_plot = PlotPanel(parent=self)
        
        plot_names = self.exp.variables['mask_names'].value
        plot_colors = ['r','b']
        plot_threshs = [self.exp.variables[i].value for i in self.exp.variables['mask_threshs'].value]
        for c,m,th in zip(plot_colors, plot_names, plot_threshs):
            self.panel_plot.new_plot(m, c=c)
            self.panel_plot.new_line(m, th, c=c)
        sizer_plot.Add(self.panel_plot, flag=wx.EXPAND)
        sizer_main.Add(sizer_plot, flag=wx.EXPAND)

        self.SetSizerAndFit(sizer_main)
    def init_menus(self):
        self.CreateStatusBar()
        
        # build menus
        file_menu= wx.Menu()
        menu_item_about = file_menu.Append(wx.ID_ABOUT, "About","About this software")
        self.Bind(wx.EVT_MENU, self.on_about, menu_item_about)
        menu_item_exit = file_menu.Append(wx.ID_EXIT,"Exit","Exit")
        self.Bind(wx.EVT_MENU, self.on_exit, menu_item_exit)

        # assemble menu bar
        menuBar = wx.MenuBar()
        menuBar.Append(file_menu, 'File')
        self.SetMenuBar(menuBar)
    def on_about(self, event):
        dlg = wx.MessageDialog( self, "C Ben Deverett 2014", "About", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()
    def on_exit(self, event):
        self.Close(True)
    def on_variable_edit(self, name):
        def in_event(event):
            self.exp.update_variable(name)
        return in_event
    def on_button(self, name):
        def in_event(event):
            self.exp.process_button(name)
        return in_event
    def step(self, e):
        self.exp.step()
        # update statuses
        for s in self.exp.statuses:
            self.statuses[s].SetValue(str(self.exp.statuses[s].value))

class CameraPanel(wx.Panel):
    def __init__(self, cam, parent=None):
        wx.Panel.__init__(self, parent)
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
    def step(self, show=True):
        exp = self.parent.exp

        frame, time = self.cam.read()
        cv2.polylines(frame, [exp.mask_pts[m] for m in exp.variables['mask_names'].value if m in exp.masks.keys()], 1, (255,255,255), thickness=2)

        if show:
            self.update_image(frame)

        exp.frame_count += 1
        if exp.RECORDING:
            exp.writer.write(frame)
            exp.time.append(time)
        if exp.masks:
            exp.monitor_frame(frame, masks=('WHEEL','EYE'))
        
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

        self.plots = {}
        self.lines = {}
    @property
    def size(self):
        return self.parent.exp.variables['mon_plot_size'].value
    def new_plot(self, name, c):
        self.plots[name] = self.axes.plot(np.zeros(self.size), c)[0] 
        self.axes.set_ylim([-1, 255])
    def new_line(self, name, val, c):
        self.lines[name] = self.axes.plot(np.repeat(1., self.size), c+'--')[0]
    def update_line(self, name, val):
        self.lines[name].set_ydata(np.repeat(val, self.size))
        self.canvas.draw()
    def update_plot(self, name, data):
        if len(data) != self.size:
            data = np.append(data, np.repeat(None, self.size-len(data)))
        self.plots[name].set_ydata(data)
        self.canvas.draw()


class Experiment(object):
    def __init__(self, camera=None, trigger_cycle=None, config=None):
        self.config = config
        
        # setup camera
        self.camera = camera
        assert type(self.camera) == Camera
        self.camera.read()
        
        # setup daq
        self.daq = DAQ(mode=DAQ.DIGITAL)
        self.analog_daq = DAQ(mode=DAQ.ANALOG)

        # setup experiment variables
        self.buttons = ['Trigger', 'Pause', 'Continue', 'Redo Trigger']
        self.trigger_cycle=trigger_cycle
        self.init_variables()
        self.masks = {}
        self.mask_idxs = {}
        self.mask_pts = {}

        # setup interface
        self.app = wx.App(False)
        self.interface = MainFrame(self)

        # begin first session
        self.new_session()

    def init_variables(self):
        self.statuses = [
                Variable(name='Since Last Trigger', short='since_last', default=-1, user=False, dtype=float),
                Variable(name='Trial #', short='trial', default=0, user=False, dtype=int),
                Variable(name='Paused', short='paused', default=False, user=False, dtype=bool),
                Variable(name='Trigger Cycle', short='trig_cycle', default=self.trigger_cycle, user=False),
        ]
        self.variables = [
                Variable(name='Movement Threshold', short='move_thresh', default=3., minn=0, maxx=100, user=True, dtype=float),
                Variable(name='Eyelid Threshold', short='eye_thresh', default=3., minn=0, maxx=100, user=True, dtype=float),
                Variable(name='Inter-trial Minimum (s)', short='iti', default=7., maxx=120, user=True, dtype=float),
                Variable(name='Wheel Translation', short='wheel_t', default=0., maxx=400, user=True, dtype=float),
                Variable(name='Wheel Stretch', short='wheel_s', default=1., maxx=50, user=True, dtype=float),
                Variable(name='Display Resample', short='resample', default=1., user=True, style='box', dtype=float),
                Variable(name='Movement Frame Buffer', short='move_frames', default=10, user=True, dtype=float),
                Variable(name='Monitor Plot Size', short='mon_plot_size', default=40, user=True, dtype=float),
                Variable(name='# of Trials', short='n_trials', default=9999, user=True, style='box', dtype=int),
                Variable(name='Name', short='name', default='noname', user=True, style='box', dtype=str),
                Variable(name='Mask Names', short='mask_names', default=['WHEEL','EYE'], user=False, show=False, dtype=list),
                Variable(name='Mask Threshs', short='mask_threshs', default=['move_thresh','eye_thresh'], user=False, show=False, dtype=list),
        ]

        self.statuses = {s.short:s for s in self.statuses}
        self.variables = {v.short:v for v in self.variables}

        self.n_sliders_for_interface = len([v for v in self.variables.items() if v[1].show and v[1].style=='slider'])
        self.n_boxes_for_interface = len([v for v in self.variables.items() if v[1].show and v[1].style=='box'])

        if self.config:
            for var in self.config:
                try:
                    self.variables[var] = self.config[var]
                except KeyError:
                    raise Exception('Invalid parameter given in config parameter: \'%s\''%var)

    def update_variable(self, short):
        interface_ctrls = dict(self.interface.sliders.items() + self.interface.boxes.items())
        self.variables[short].value = interface_ctrls[short].GetValue()

        if short == 'name':
            self.new_session()
        elif short == 'move_thresh':
            self.interface.panel_plot.update_line('WHEEL', self.variables['move_thresh'].value)
        elif short == 'eye_thresh':
            self.interface.panel_plot.update_line('EYE', self.variables['eye_thresh'].value)
    def process_button(self, name):
        if name == 'Pause':
            self.PAUSED = True
            self.update_status()
        elif name == 'Continue':
            self.PAUSED = False
        elif name == 'Trigger':
            self.PAUSED = False
            self.send_trigger()
            self.start_recording()
        elif name == 'Redo Trigger':
            self.trigger_cycle.redo()
            self.statuses['trial'].value -= 1
    def update_status(self):
        self.statuses['paused'].value = self.PAUSED
        self.statuses['since_last'].value = pytime.time()-self.last_trial_off
        self.statuses['trig_cycle'].value = str(self.trigger_cycle)

    def save_masks(self):
        np.save(os.path.join(self.variables['name'].value,'masks'), np.atleast_1d([self.masks]))
    def set_masks(self):
        mask_names = self.variables['mask_names'].value
        masks = {}
        for m in mask_names:
            frame, timestamp = self.camera.read()
            pl.figure()
            pl.title("Select mask: %s."%m)
            pl.imshow(frame, cmap=mpl_cm.Greys_r)
            pts = []
            while not len(pts):
                pts = pl.ginput(0)
            pl.close()
            path = mpl_path.Path(pts)
            mask = np.ones(np.shape(frame), dtype=bool)
            for ridx,row in enumerate(mask):
                for cidx,pt in enumerate(row):
                    if path.contains_point([cidx, ridx]):
                        mask[ridx,cidx] = False
            self.mask_pts[m] = np.array(pts, dtype=np.int32)
            masks[m] = mask
            self.mask_idxs[m] = np.where(mask==False)
        self.save_masks()
        self.masks = masks
    def new_session(self):
        name = self.variables['name'].value
        if not os.path.isdir(name):
            os.mkdir(name)
            self.statuses['trial'].value = 0

        self.reset_monitors()
        self.RECORDING = False
        self.PAUSED = False
        self.last_trial_off = pytime.time()
        self.frame_count = 0

        # set masks
        if len(self.masks)==0:     
            self.set_masks()
        self.save_masks()
    def monitor_frame(self, frame, masks=('WHEEL', 'EYE')):
        if 'WHEEL' in masks:
            if None in self.monitor_img_set:
                return 
            self.monitor_img_set = np.roll(self.monitor_img_set, 1, axis=2)
            self.monitor_img_set[:,:,0] = frame
            pts = self.monitor_img_set[self.mask_idxs['WHEEL'][0],self.mask_idxs['WHEEL'][1],:]
            std_pts = np.std(pts, axis=1)
            wval = np.mean(std_pts) * self.variables['wheel_s'].value + self.variables['wheel_t'].value
            self.monitor_vals['WHEEL'] = np.roll(self.monitor_vals['WHEEL'], -1)
            self.monitor_vals['WHEEL'][-1] = wval
        if 'EYE' in masks:
            pts = frame[self.mask_idxs['EYE'][0],self.mask_idxs['EYE'][1]]
            eyval = np.mean(pts)
            self.monitor_vals['EYE'] = np.roll(self.monitor_vals['EYE'], -1)
            self.monitor_vals['EYE'][-1] = eyval
            self.update_analog_daq()

        if not self.RECORDING:
            self.update_plots()
    def update_plots(self):
        for mv in self.monitor_vals:
            self.interface.panel_plot.update_plot(mv, self.monitor_vals[mv])
    def update_analog_daq(self):
        if self.monitor_vals['EYE'][-1] != None:
            val = self.monitor_vals['EYE'][-1]
            val = self.normalize(val, oldmin=0., oldmax=255., newmin=self.analog_daq.minn, newmax=self.analog_daq.maxx)          
            tr = Trigger(msg=val)
            self.analog_daq.trigger(tr)
    def normalize(self, val, oldmin, oldmax, newmin, newmax):
        return ((val-oldmin)/oldmax) * (newmax-newmin) + newmin
    def query_for_trigger(self):
        if pytime.time()-self.last_trial_off < self.variables['iti'].value:
            return False
        return self.monitor_vals['WHEEL'][-1] < self.variables['move_thresh'].value
    def send_trigger(self):
        self.daq.trigger(self.trigger_cycle.next)
        print "Sent trigger #%i"%(self.statuses['trial'].value+1)
    def start_recording(self):
        self.RECORDING = pytime.time()
        self.statuses['trial'].value += 1

        name = self.variables['name'].value
        trial = self.statuses['trial'].value
        
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
        self.RECORDING = False
        self.last_trial_off = pytime.time()
        np.savez_compressed(self.filename+'.npz', time=self.time)         
        self.writer.release()
        self.filename = None

    def step(self):

        if self.RECORDING:
            self.interface.panel_camera.step(show=False)
            if pytime.time()-self.RECORDING >= self.trigger_cycle.current.duration:
                self.end_trial()
        else:
            if not self.PAUSED:
                self.interface.panel_camera.step(show=True)
                if self.query_for_trigger():
                    self.send_trigger()
                    self.start_recording()

                self.update_status()
    def reset_monitors(self):
        self.monitor_img_set = np.empty((self.camera.resolution[1],self.camera.resolution[0],self.variables['move_frames'].value))
        self.monitor_img_set[:] = None
        self.monitor_vals = {m:np.empty(self.variables['mon_plot_size'].value) for m in self.variables['mask_names'].value}
        for m in self.monitor_vals:
            self.monitor_vals[m][:] = None
    def go(self):
        self.app.MainLoop()


class Variable(object):
    def __init__(self, name, short, minn=0., maxx=999999., default=0., user=False, show=True, style='slider', dtype=str):
        self.name = name
        self.short = short
        self.minn = minn
        self.maxx = maxx
        self.default = default
        self.user = user
        self.show = show
        self.style = style
        self.dtype = dtype

        self._value = self.default
        self.value = self._value
    @property
    def value(self):
        return self._value
    @value.setter
    def value(self, new):
        self._value = self.dtype(new)


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
    cam = Camera()
    CS = Trigger([0,0,0,0], name='CS', duration=5.0)
    US = Trigger([0,0,0,0], name='US', duration=5.0)
    tc = TriggerCycle([CS, US, CS, CS, CS, CS])
    e = Experiment(camera=cam, trigger_cycle=tc)
    e.go()
