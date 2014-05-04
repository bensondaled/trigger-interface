try:
    import PyDAQmx as pydaq
except:
    import csv as pydaq
import numpy as np

class Trigger(object):
    def __init__(self, msg=[], duration=None, name='noname'):
        self.duration = duration
        self._msg = None
        self.msg = msg
        self.name = name

    @property
    def msg(self):
        return self._msg
    @msg.setter
    def msg(self, msg):
        if type(msg) == list or type(msg) == np.array:
            self._msg = np.array(msg).astype(np.uint8)
        else:
            self._msg = np.array(msg).astype(np.float64)

    def metadata(self):
        md = {}
        md['duration'] = self.duration
        md['msg'] = str(self.msg)
        md['name'] = self.name
        return md

class DAQ(object):
    ANALOG = 0
    DIGITAL = 1
    def __init__(self, mode, port_digital="Dev1/Port1/Line0:3", port_analog="Dev1/ao0"):
        self.mode = mode
        if self.mode == self.ANALOG:
            self.port = port_analog
            self.minn = 0.0
            self.maxx = 5.0
        elif self.mode == self.DIGITAL:
            self.port = port_digital
        self.clear_trig = Trigger(msg=[0,0,0,0])
        try:
            self.task = pydaq.TaskHandle()
            pydaq.DAQmxCreateTask("", pydaq.byref(self.task))
            if self.mode == self.DIGITAL:
                pydaq.DAQmxCreateDOChan(self.task, self.port, "OutputOnly", pydaq.DAQmx_Val_ChanForAllLines)
            elif self.mode == self.ANALOG:
                pydaq.DAQmxCreateAOVoltageChan(self.task, self.port,"", self.minn,self.maxx,pydaq.DAQmx_Val_Volts,None)
            
            pydaq.DAQmxStartTask(self.task)
        except:
            self.task = None
            print "DAQ task did not successfully initialize."
    def trigger(self, trig):
        if self.task:
            if self.mode == self.DIGITAL:
                pydaq.DAQmxWriteDigitalLines(self.task,1,1,10.0,pydaq.DAQmx_Val_GroupByChannel,trig.msg,None,None)
                pydaq.DAQmxWriteDigitalLines(self.task,1,1,10.0,pydaq.DAQmx_Val_GroupByChannel,self.clear_trig.msg,None,None)
            elif self.mode == self.ANALOG:
                pydaq.DAQmxWriteAnalogF64(self.task,1,1,10.0,pydaq.DAQmx_Val_GroupByChannel,trig.msg,None,None)
        else:
            print "DAQ task not functional. Attempted to write %s."%str(trig.msg)
    def release(self):
        if self.task:
            pydaq.DAQmxStopTask(self.task)
            pydaq.DAQmxClearTask(self.task)

    def metadata(self):
        md = {}
        md['port'] = self.port
        
        return md
