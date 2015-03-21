from ctypes import c_int, c_void_p, c_char_p, c_float, c_uint16, c_uint32, c_uint8
from ctypes import Structure, byref
import ctypes

# camera sensor parameters
CLEYE_AUTO_GAIN = 0 #[false, true]
CLEYE_GAIN = 1 #[0, 79]
CLEYE_AUTO_EXPOSURE = 2 #[false, true]
CLEYE_EXPOSURE = 3#[0, 511]
CLEYE_AUTO_WHITEBALANCE = 4 #[false, true]
CLEYE_WHITEBALANCE_RED = 5#[0, 255]
CLEYE_WHITEBALANCE_GREEN = 6#[0, 255]
CLEYE_WHITEBALANCE_BLUE = 7#[0, 255]
# camera linear transform parameters
CLEYE_HFLIP = 8#[false, true]
CLEYE_VFLIP = 9#[false, true]
CLEYE_HKEYSTONE = 10#[-500, 500]
CLEYE_VKEYSTONE = 11#[-500, 500]
CLEYE_XOFFSET = 12#[-500, 500]
CLEYE_YOFFSET = 13#[-500, 500]
CLEYE_ROTATION = 14#[-500, 500]
CLEYE_ZOOM = 15#[-500, 500]
# camera non-linear transform parameters
CLEYE_LENSCORRECTION1 = 16#[-500, 500]
CLEYE_LENSCORRECTION2 = 17#[-500, 500]
CLEYE_LENSCORRECTION3 = 18#[-500, 500]
CLEYE_LENSBRIGHTNESS = 19#[-500, 500]

#CLEyeCameraColorMode
CLEYE_MONO_PROCESSED = 0
CLEYE_COLOR_PROCESSED = 1
CLEYE_MONO_RAW = 2
CLEYE_COLOR_RAW = 3
CLEYE_BAYER_RAW = 4

#CLEyeCameraResolution
CLEYE_QVGA = 0
CLEYE_VGA = 1



class GUID(Structure):
    _fields_ = [("Data1", c_uint32),
             ("Data2", c_uint16),
             ("Data3", c_uint16),
             ("Data4", ctypes.c_uint8 * 8)]
    def __str__(self):
        return "%X-%X-%X-%s" % (self.Data1, self.Data2, self.Data3, ''.join('%02X'%x for x in self.Data4))


lib = "CLEyeMulticam.dll"
dll = ctypes.cdll.LoadLibrary(lib)
dll.CLEyeGetCameraUUID.restype = GUID
dll.CLEyeCameraGetFrame.argtypes = [c_void_p, c_char_p, c_int]
dll.CLEyeCreateCamera.argtypes = [GUID, c_int, c_int, c_float]

def CLEyeGetCameraCount():
    return dll.CLEyeGetCameraCount()

def CLEyeCameraGetFrameDimensions(cam):
    width = c_int()
    height = c_int()
    dll.CLEyeCameraGetFrameDimensions(cam, byref(width), byref(height))
    return width.value, height.value
    
def CLEyeGetCameraParameter(cam, param):
    return dll.CLEyeGetCameraParameter(cam, param)
    
def CLEyeSetCameraParameter(cam, param, value):
    return dll.CLEyeSetCameraParameter(cam, param, value)
    
def CLEyeGetCameraUUID(index):
    return dll.CLEyeGetCameraUUID(index)
    
def CLEyeCreateCamera(uuid, color_mode, resolution_mode, fps):
    return dll.CLEyeCreateCamera(uuid, color_mode, resolution_mode, fps)

def CLEyeCameraStart(cam):
    return dll.CLEyeCameraStart(cam)
    
def CLEyeCameraStop(cam):
    return dll.CLEyeCameraStop(cam)
    
def CLEyeDestroyCamera(cam):
    return dll.CLEyeDestroyCamera(cam)
    
def CLEyeCameraGetFrame(cam, frame, timeout):
    return dll.CLEyeCameraGetFrame(cam, frame, timeout)
    
class Ps3Eye():
    def __init__(self, index, color_mode, resolution_mode, fps):
        self.cam = CLEyeCreateCamera(CLEyeGetCameraUUID(index), color_mode, resolution_mode, fps)
        self.bytes_per_pixel = 1
        if color_mode not in (CLEYE_MONO_PROCESSED, CLEYE_MONO_RAW):
            self.bytes_per_pixel = 4
        self.x, self.y = CLEyeCameraGetFrameDimensions(self.cam)
        
    def configure(self, settings):
        for param, value in settings:
            CLEyeSetCameraParameter(self.cam, param, value)
            
    def start(self):
        return CLEyeCameraStart(self.cam)
    
    def stop(self):
        return CLEyeCameraStop(self.cam)
        
    def get_frame(self, buffer=None, timeout=1):
        buffer = buffer or ctypes.create_string_buffer(self.x * self.y * self.bytes_per_pixel) 
        got=CLEyeCameraGetFrame(self.cam, buffer, timeout)
        return (got,buffer)
        
    def __del__(self):
        self.stop()
        CLEyeDestroyCamera(self.cam)
        
def test_Ps3Eye():
    import matplotlib.pyplot as plt
    import numpy
    eye = Ps3Eye(0, CLEYE_MONO_PROCESSED, CLEYE_VGA, 75)  
    settings = [ (CLEYE_AUTO_GAIN, 1), \
                 (CLEYE_AUTO_EXPOSURE, 1),\
                 (CLEYE_AUTO_WHITEBALANCE, 1)]
    eye.configure(settings)
    eye.start()
  
    frame = eye.get_frame()
    frame = numpy.fromstring(frame, numpy.dtype('uint8')).reshape(eye.y, eye.x)
    im = plt.imshow(frame)
    plt.show()

    
    
def main():
    import cv
    from array import array
    print 'camera count', CLEyeGetCameraCount()
    print 'UUID for first camera', CLEyeGetCameraUUID(0)
    cam = CLEyeCreateCamera(CLEyeGetCameraUUID(0), CLEYE_MONO_PROCESSED, CLEYE_VGA, 75)
    #print cam
    CLEyeSetCameraParameter(cam, CLEYE_AUTO_GAIN, 0)
    CLEyeSetCameraParameter(cam, CLEYE_AUTO_EXPOSURE, 0)
    CLEyeSetCameraParameter(cam, CLEYE_AUTO_WHITEBALANCE, 1)
    #if set to auto just ignored
    CLEyeSetCameraParameter(cam, CLEYE_GAIN, 60)
    CLEyeSetCameraParameter(cam, CLEYE_EXPOSURE, 25)
    CLEyeSetCameraParameter(cam, CLEYE_WHITEBALANCE_RED, 50)
    CLEyeSetCameraParameter(cam, CLEYE_WHITEBALANCE_BLUE, 50)
    CLEyeSetCameraParameter(cam, CLEYE_WHITEBALANCE_GREEN, 50)
    print "auto gain", CLEyeGetCameraParameter(cam, CLEYE_AUTO_GAIN)
    print "auto exposure", CLEyeGetCameraParameter(cam, CLEYE_AUTO_EXPOSURE)
    print "auto whitebalance", CLEyeGetCameraParameter(cam, CLEYE_AUTO_WHITEBALANCE)
    print "gain", CLEyeGetCameraParameter(cam, CLEYE_GAIN)
    print "exposure", CLEyeGetCameraParameter(cam, CLEYE_EXPOSURE)
    print "red", CLEyeGetCameraParameter(cam, CLEYE_WHITEBALANCE_RED)
    print "green", CLEyeGetCameraParameter(cam, CLEYE_WHITEBALANCE_BLUE)
    print "blue", CLEyeGetCameraParameter(cam, CLEYE_WHITEBALANCE_GREEN)
    
    
    x, y = CLEyeCameraGetFrameDimensions(cam)
    #print x, y
    CLEyeCameraStart(cam)
    frame = ctypes.create_string_buffer(x * y * 4) 
    cv.NamedWindow( "camera", 1 ) 
    imagen=cv.CreateImage((x, y), 8, 1)
    mat = cv.CreateMat(x, y, 0)

    i = 0
    while True:
        i += 1
        CLEyeCameraGetFrame(cam, frame, 100)
        cv.SetData(imagen, array('B', frame[:]).tostring())
        cv.ShowImage( "camera", imagen )
        #cv.SaveImage("r:\\Cap\\%i.jpg" % i, imagen)
        c = cv.WaitKey(1)
        if c == 27:
            break
        if i == 100:
            i = 0
    CLEyeCameraStop(cam)                
    CLEyeDestroyCamera(cam)                

if __name__ == '__main__':
    #main()
    pass
    #test_Ps3Eye()