'''
configuration parameters 
'''

IMG_SIZE = 128
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3) 
WINDOW_SIZE = 5
WINDOW_IMG_SHAPE = (WINDOW_SIZE, IMG_SIZE, IMG_SIZE, 3)
DIM_PREDICTIONS = 3 # 3 for translation, 3 more probably (euler angles) for translation + rotation
DISP_H = 544
DISP_W = 960