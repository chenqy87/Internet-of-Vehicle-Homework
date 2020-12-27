'''
Created on 2020-12-10

@author: aiyu
'''

import cv2
from process_pic import process_frame_pic
from process import process_frame
from process_pic_backup import process_frame_pic_backup
from process_pic_backup2 import process_frame_pic_backup2



mode = 2

if mode == 1:
    img = cv2.imread(r"F:\data\zuoye\s.jpg")
    img = cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4)))
    process_frame_pic(img)
elif mode == 2:
    videopath = r"G:\test.AVI"
    video = cv2.VideoCapture(videopath)
    while True:
        flag,frame = video.read()
        frame_aug = process_frame(frame)
        cv2.imshow("aug_frame",frame_aug)
        if cv2.waitKey(25) == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()    
      
elif mode == 3:
    img = cv2.imread(r"F:\data\zuoye\normal_situation\normal_situation\682.BMP")
    process_frame_pic_backup(img)  
else:
    img = cv2.imread(r"F:\data\zuoye\car_1.jpg")
    img = cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4)))
    process_frame_pic_backup2(img) 
    



