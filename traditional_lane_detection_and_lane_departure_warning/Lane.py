'''
Created on 2020-12-10

@author: aiyu
'''
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from copy import copy

class Lane():
    def __init__(self):
        self.detected = False
        self.cur_fitx = None
        self.cur_fity = None
        self.prev_fitx = []
        self.current_poly = [np.array([False])]
        self.prev_poly = [np.array([False])]
        
    def average_pre_lanes(self):
        tmp = copy(self.prev_fitx)
        tmp.append(self.cur_fitx)
        self.mean_fitx = np.mean(tmp,axis = 0)
        
    def append_fitx(self):
        if len(self.prev_fitx) == 4:
            self.prev_fitx.pop(0)
        self.prev_fitx.append(self.mean_fitx)
        
    def process(self,ploty):
        self.cur_fity = ploty
        self.average_pre_lanes()
        self.append_fitx()
        self.prev_poly = self.current_poly
        