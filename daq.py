import PyDAQmx as pydaq
import numpy as np

class Trigger(object):
        def __init__(self, msg=[], duration=1.0):
                self.duration = duration
                self._msg = None
                self.msg = msg

        @property
        def msg(self):
                return self._msg
        @msg.setter
        def msg(self, msg):
                self._msg = np.array(msg).astype(np.uint8)

        def metadata(self):
                md = {}
                md['duration'] = self.duration
                md['msg'] = str(self.msg)

                return md

class DAQ(object): 
        def __init__(self, port="Dev1/Port1/Line0:3"): 
                self.port = port
                self.clear_trig = Trigger(msg=[0,0,0,0])
                try: 
                        self.task = pydaq.TaskHandle() 
                        pydaq.DAQmxCreateTask("", pydaq.byref(self.task)) 
                        pydaq.DAQmxCreateDOChan(self.task, self.port, "OutputOnly", pydaq.DAQmx_Val_ChanForAllLines) 
                        pydaq.DAQmxStartTask(self.task) 
                except: 
                        self.task = None
                        print "DAQ task did not successfully initialize."
        def trigger(self, trig): 
                if self.task: 
                        pydaq.DAQmxWriteDigitalLines(self.task,1,1,10.0,pydaq.DAQmx_Val_GroupByChannel,trig.msg,None,None)
                        pydaq.DAQmxWriteDigitalLines(self.task,1,1,10.0,pydaq.DAQmx_Val_GroupByChannel,self.clear_trig.msg,None,None)						
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
