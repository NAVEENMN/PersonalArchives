import cv2
import ctypes
import numpy as np
import win32api as wapi
import win32gui, win32ui, win32con, win32api
import cv2, os
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle

SendInput = ctypes.windll.user32.SendInput
IMAGE_WIDTH           = 320
IMAGE_HEIGHT          = 240
IMAGE_CHANNELS        = 1 
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


keyList = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys

def grab_screen(region=None):

    hwin = win32gui.GetDesktopWindow()

    if region:
            left,top,x2,y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

def Actions(action):
	# Straight
    print (action)
    if action[0] == 1:
        PressKey(W)
        ReleaseKey(A)
        ReleaseKey(D)
        ReleaseKey(S)
    # Left
    if action[1] == 2:
        PressKey(S)
        ReleaseKey(A)
        time.sleep(t_time)
        ReleaseKey(D)
        ReleaseKey(S)
    # Right
    if action[2] == 3:
        PressKey(D)
        ReleaseKey(A)
        time.sleep(t_time)
        ReleaseKey(D)
        ReleaseKey(S)
    # Slow
    if action[3] == 4:
        PressKey(A)
        ReleaseKey(S)
        ReleaseKey(W)
        time.sleep(t_time)
        ReleaseKey(D)


def get_pressed_key():
	#output= ['W','A','D','S']
	output = [0,0,0,0] #no action
	keys = key_check()
	if 'W' in keys:
		output[0] = 1
	if 'A' in keys:
		output[1] = 1
	if 'D' in keys:
		output[2] = 1
	if 'S' in keys:
		output[3] = 1
	if 'T' in keys:
		exit(0)
	return output

def load_image(data_dir, image_file):
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

def resize(image):
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def preprocess(image):
    image = resize(image)
    return image

def batch_generator(data_dir, image_paths, actions, batch_size, is_training):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    actions_to_take = np.empty([batch_size, 4])
    while True:
        i = 0
        for index in np.random.permutation(len(image_paths)):
            image_file = image_paths[index]
            action = actions[index]
            image = load_image(data_dir, image_file)
            image = preprocess(image)
            image = cv2.Canny(image,10,120)
            images[i] = np.reshape(image, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
            actions_to_take[i] = action
            i += 1
            if i == batch_size:
                break
        yield images, actions_to_take
