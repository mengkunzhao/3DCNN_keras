#Helper Function to read video frames

import numpy as np
import cv2

class Videoto3D:

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def video3d(self, filename, color, skip=True ):
        cap = cv2.VideoCapture(filename)

        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if skip:
            frames = [x * nframe / self.depth for x in range(self.depth)]
        else:
            frames = [x for x in range(self.depth)]
        framearray = []
        for i in range(self.depth):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
            ret, frame = cap.read()
            if ret == False:
                framearray = []
                break
            frame_temp = cv2.resize(frame, (self.height, self.width))
            if color:
                framearray.append(frame_temp/255)
            else:
                framearray.append(cv2.cvtColor(frame_temp, cv2.COLOR_BGR2GRAY)/255)

        cap.release()
        return np.array(framearray)

