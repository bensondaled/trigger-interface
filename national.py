from PyDAQmx import *
import numpy as np

# Declaration of variable passed by reference


taskHandle = TaskHandle()
written = int32()
data = np.array([1,1,1,1]).astype(np.uint8)

try:
    # DAQmx Configure Code
    DAQmxCreateTask("",byref(taskHandle))

    DAQmxCreateDOChan(taskHandle,"Dev1/Port1/Line0:3","OutputOnly",DAQmx_Val_ChanForAllLines)
    
    # DAQmxCreateAIVoltageChan(taskHandle,"Dev1/ai0","",DAQmx_Val_Cfg_Default,-10.0,10.0,DAQmx_Val_Volts,None)
    
    # DAQmxCfgSampClkTiming(taskHandle,"",10000.0,DAQmx_Val_Rising,DAQmx_Val_FiniteSamps,1000)

    # DAQmx Start Code
    DAQmxStartTask(taskHandle)

    # DAQmx Read Code
    DAQmxWriteDigitalLines(taskHandle,1,1,10.0,DAQmx_Val_GroupByChannel,data,None,None)
    ##DAQmxReadAnalogF64(taskHandle,1000,10.0,DAQmx_Val_GroupByChannel,data,1000,byref(read),None)

    print "Acquired %d points"%written.value
except DAQError as err:
    print "DAQmx Error: %s"%err
finally:
    if taskHandle:
        # DAQmx Stop Code
        DAQmxStopTask(taskHandle)
        DAQmxClearTask(taskHandle)
