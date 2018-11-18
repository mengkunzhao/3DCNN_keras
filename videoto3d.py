import numpy as np
import cv2


class Videoto3D:

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def video3d(self, filename, skip=True):
        cap = cv2.VideoCapture(filename)
       # print(filename)
        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if skip:
            frames = [x * nframe / self.depth for x in range(self.depth)]
        else:
            frames = [x for x in range(self.depth)]
        framearray = []
        retcheck = []
        frameshapecheck = []
        for i in range(self.depth):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
            ret, frame = cap.read()
            retcheck.append(ret)
            if frame is None:
                break
            frameshapecheck.append(frame.shape)
            frame_temp = cv2.resize(frame, (self.height, self.width))
            framearray.append(frame_temp)
            file2label = filename
          #  print("frame loaded")


        cap.release()
        return np.array(framearray) ,retcheck,frameshapecheck



    def get_UCF_classname(self, filename):
        return filename[filename.find('_') + 1:filename.find('_', 2)]

