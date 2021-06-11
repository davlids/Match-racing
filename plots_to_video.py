import cv2
import os
from os.path import isfile, join

dir_nr = 0
pathIn= f'./Race plots/Plots_{dir_nr}'
pathOut = f'./Race plots/Saved videos/{dir_nr}.avi'
fps = 1
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]


files.sort(key = lambda x: x[5:-4])
files.sort()
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

files.sort(key = lambda x: int(x[4:-4]))

for i in range(len(files)):
    filename=pathIn + '/' + files[i]

    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    
    frame_array.append(img)
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for i in range(len(frame_array)):
    out.write(frame_array[i])
out.release()