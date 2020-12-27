'''
Created on 2020-12-5

@author: aiyu
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import skimage
from copy import copy
from timeit import default_timer as timer
from Lane import Lane

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255.*abs_sobel/np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)

    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """ threshold according to the direction of the gradient

    :param img:
    :param sobel_kernel:
    :param thresh:
    :return:
    """

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # 5) Create a binary mask where direction thresholds are met
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output


def gradient_pipeline(image, ksize = 3, sx_thresh=(20, 100), sy_thresh=(20, 100), m_thresh=(30, 100), dir_thresh=(0.7, 1.3)):

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=sx_thresh)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=sy_thresh)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, thresh=m_thresh)
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=dir_thresh)
    combined = np.zeros_like(mag_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    # combined[(gradx == 1)  | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined


def threshold_col_channel(channel, thresh):

    binary = np.zeros_like(channel)
    binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1

    return binary

# s_thresh, sx_thresh, dir_thresh, m_thresh, r_thresh = (120, 255), (20, 100), (0.7, 1.3), (30, 100), (200, 255)

def find_edges(img, s_thresh=(120, 255), sx_thresh=(20, 100), dir_thresh=(0.7, 1.3)):

    img = np.copy(img)
    # Convert to HSV color space and threshold the s channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    s_binary = threshold_col_channel(s_channel, thresh=s_thresh)

    # Sobel x
    sxbinary = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=sx_thresh)
    # mag_binary = mag_thresh(img, sobel_kernel=3, thresh=m_thresh)
    # # gradient direction
    dir_binary = dir_threshold(img, sobel_kernel=3, thresh=dir_thresh)
    #
    # # output mask
    combined_binary = np.zeros_like(s_channel)
    combined_binary[(( (sxbinary == 1) & (dir_binary==1) ) | ( (s_binary == 1) & (dir_binary==1) ))] = 1

    # add more weights for the s channel
    c_bi = np.zeros_like(s_channel)
    c_bi[( (sxbinary == 1) & (s_binary==1) )] = 2

    ave_binary = (combined_binary + c_bi)

    return ave_binary


def warper(img, M):

    # Compute and apply perspective transform
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

## fit the lane line
def full_search(binary_warped, input_scale, frame_width, frame_height, visualization=False):

    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img = out_img.astype('uint8')

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = np.floor(10/input_scale)
    # Set minimum number of pixels found to recenter window
    minpix = np.floor(5/input_scale)
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if visualization:
            cv2.rectangle(out_img,(int(win_xleft_low),int(win_y_low)),(int(win_xleft_high),int(win_y_high)),(0,255,0), 2)
            cv2.rectangle(out_img,(int(win_xright_low),int(win_y_low)),(int(win_xright_high),int(win_y_high)),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Visualization

    # Generate x and y values for plotting
    if visualization:
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # plt.subplot(1,2,1)
        plt.imshow(out_img)
        # plt.imshow(binary_warped)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim((0, frame_width))
        plt.ylim((frame_height, 0))
        plt.show()

    return left_fit, right_fit



def window_search(left_fit, right_fit, binary_warped, input_scale, frame_width, frame_height, margin=100, visualization=False):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's easier to find line pixels with windows search
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if visualization:
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        # And you're done! But let's visualize the result here as well
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        out_img = out_img.astype('uint8')
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim((0, frame_width))
        plt.ylim((frame_height, 0))

        plt.show()

    return left_fit, right_fit



def tracker(binary_sub, right_lane, left_lane, ploty, input_scale,visualization=False):
    
    frame_width = binary_sub.shape[1]
    frame_height = binary_sub.shape[0]    
    left_fit, right_fit = window_search(left_lane.prev_poly, right_lane.prev_poly, binary_sub, input_scale, 
                                        frame_width, frame_height, margin=100/input_scale, 
                                        visualization=visualization)

    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    std_value = np.std(right_fitx - left_fitx)
    if std_value < (85 /input_scale):
        left_lane.detected = True
        right_lane.detected = True
        left_lane.current_poly = left_fit
        right_lane.current_poly = right_fit
        left_lane.cur_fitx = left_fitx
        right_lane.cur_fitx = right_fitx
        # global tt
        # tt = tt + 1
    else:
        left_lane.detected = False
        right_lane.detected = False
        left_lane.current_poly = left_lane.prev_poly
        right_lane.current_poly = right_lane.prev_poly
        left_lane.cur_fitx = left_lane.prev_fitx[-1]
        right_lane.cur_fitx = right_lane.prev_fitx[-1]

    return right_lane, left_lane



def detector(binary_sub, right_lane, left_lane, ploty, input_scale, visualization=False):
    frame_width = binary_sub.shape[1]
    frame_height = binary_sub.shape[0]
    left_fit, right_fit = full_search(binary_sub, input_scale, frame_width, frame_height, visualization=visualization)
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    std_value = np.std(right_fitx - left_fitx)
    if std_value < (85 /input_scale):
        left_lane.current_poly = left_fit
        right_lane.current_poly = right_fit
        left_lane.cur_fitx = left_fitx
        right_lane.cur_fitx = right_fitx
        left_lane.detected = True
        right_lane.detected = True
    else:
        left_lane.current_poly = left_lane.prev_poly
        right_lane.current_poly = right_lane.prev_poly
        if len(left_lane.prev_fitx) > 0:
            left_lane.cur_fitx = left_lane.prev_fitx[-1]
            right_lane.cur_fitx = right_lane.prev_fitx[-1]
        else:
            left_lane.cur_fitx = left_fitx
            right_lane.cur_fitx = right_fitx
        left_lane.detected = False
        right_lane.detected = False
    return right_lane, left_lane

def measure_lane_curvature(ploty, leftx, rightx, lane_width, frame_height, input_scale, visualization=False):

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

     # choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/(frame_height/input_scale) # meters per pixel in y dimension
    xm_per_pix = lane_width/(700/input_scale) # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')

    if leftx[0] - leftx[-1] > 200/input_scale:
        curve_direction = 'Left curve'
    elif leftx[-1] - leftx[0] > 100/input_scale:
        curve_direction = 'Right curve'
    else:
        curve_direction = 'Straight'

    return (left_curverad+right_curverad)/2.0, curve_direction

def off_center(left, mid, right, lane_width):
    a = mid - left
    b = right - mid
    width = right - left

    if a >= b:  # driving right off
        offset = a / width * lane_width - lane_width /2.0
    else:       # driving left off
        offset = lane_width /2.0 - b / width * lane_width

    return offset


def compute_car_offcenter(ploty, left_fitx, right_fitx, lane_width, img):

    # Create an image to draw the lines on
    height = img.shape[0]
    width = img.shape[1]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    bottom_l = left_fitx[height-1]
    bottom_r = right_fitx[0]

    offcenter = off_center(bottom_l, width*0.5, bottom_r, lane_width)

    return offcenter, pts

def create_output_frame(offcenter, pts, undist_ori, fps, curvature, curve_direction, output_frame_scale, 
                        input_scale, frame_height ,frame_width, M_b, M_inv, binary_sub, threshold=0.6):

    undist_ori = cv2.resize(undist_ori, (0,0), fx=1/output_frame_scale, fy=1/output_frame_scale)
    w = undist_ori.shape[1]
    h = undist_ori.shape[0]

    undist_birdview = warper(cv2.resize(undist_ori, (0,0), fx=1/2, fy=1/2), M_b)

    color_warp = np.zeros_like(undist_ori).astype(np.uint8)

    # create a frame to hold every image
    whole_frame = np.zeros((int(h*2.5),int(w*2.34), 3), dtype=np.uint8)


    if abs(offcenter) > threshold:  # car is offcenter more than 0.6 m
        # Draw Red lane
        cv2.fillPoly(color_warp, np.int_([pts]), (255, 0, 0)) # red
    else: # Draw Green lane
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))  # green

    newwarp = cv2.warpPerspective(color_warp, M_inv, (int(frame_width/input_scale), int(frame_height/input_scale)))

    # Combine the result with the original image    # result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    newwarp_ = cv2.resize(newwarp,None, fx=input_scale/output_frame_scale, fy=input_scale/output_frame_scale, interpolation = cv2.INTER_LINEAR)

    output = cv2.addWeighted(undist_ori, 1, newwarp_, 0.3, 0)

    ############## generate the combined output frame only for visualization purpose ################
    whole_frame[40:40+h, 20:20+w, :] = undist_ori
    whole_frame[40:40+h, 60+w:60+2*w, :] = output
    whole_frame[int(220+h/2):int(220+2*h/2), 20:int(20+w/2), :] = undist_birdview
    whole_frame[int(220+h/2):int(220+2*h/2), int(40+w/2):40+w, 0] = cv2.resize((binary_sub*255).astype(np.uint8), (0,0), fx=1/2, fy=1/2)
    whole_frame[int(220+h/2):int(220+2*h/2), int(40+w/2):40+w, 1] = cv2.resize((binary_sub*255).astype(np.uint8), (0,0), fx=1/2, fy=1/2)
    whole_frame[int(220+h/2):int(220+2*h/2), int(40+w/2):40+w, 2] = cv2.resize((binary_sub*255).astype(np.uint8), (0,0), fx=1/2, fy=1/2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if offcenter >= 0:
        offset = offcenter
        direction = 'Right'
    elif offcenter < 0:
        offset = -offcenter
        direction = 'Left'

    info_road = "Road Status"
    info_lane = "Lane info: {0}".format(curve_direction)
    info_cur = "Curvature {:6.1f} m".format(curvature)
    info_offset = "Off center: {0} {1:3.1f}m".format(direction, offset)
    info_framerate = "{0:4.1f} fps".format(fps)
    info_warning = "Warning: offcenter > 0.6m (use higher threshold in real life)"

    cv2.putText(whole_frame, "Departure Warning System", (23,25), font, 0.8, (255,255,0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, "Origin", (22,70), font, 0.6, (255,255,0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, "Augmented", (40+w+25,70), font, 0.6, (255,255,0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, "Bird's View", (22+30,70+35+h), font, 0.6, (255,255,0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, "Lanes", (22+225,70+35+h), font, 0.6, (255,255,0), 1, cv2.LINE_AA)
    cv2.putText(whole_frame, info_road, (40+w+50,70+35+h), font, 0.8, (255,255,0), 1,cv2.LINE_AA)
    cv2.putText(whole_frame, info_warning, (35+w,60+h), font, 0.4, (255,255,0), 1,cv2.LINE_AA)
    cv2.putText(whole_frame, info_lane, (40+w+50,70+35+40+h), font, 0.8, (255,255,0), 1,cv2.LINE_AA)
    cv2.putText(whole_frame, info_cur, (40+w+50,70+35+80+h), font, 0.8, (255,255,0), 1,cv2.LINE_AA)
    cv2.putText(whole_frame, info_offset, (40+w+50,70+35+120+h), font, 0.8, (255,255,0), 1,cv2.LINE_AA)
    cv2.putText(whole_frame, info_framerate, (40+w+250,70), font, 0.6, (255,255,0), 1,cv2.LINE_AA)

    return whole_frame

points = []

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x,y])
        
#         cv2.circle(img, (x, y), 1, (255, 0, 0), thickness = -1)
#         cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                     1.0, (0,0,0), thickness = 1)
#         cv2.imshow("image", img)

def roi_mask(img, corner_points):
    # 创建掩膜
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, corner_points, 255)
    mask = np.array(mask,dtype = np.uint8)
    masked_img = cv2.bitwise_and(img, img,mask = mask)
    return masked_img

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    # 统计概率霍夫直线变换
#     img = np.array(img,dtype = np.int16)
    lines = cv2.HoughLinesP(img, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
    # 新建一副空白画布
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
#     draw_lines(drawing, lines)     # 画出直线检测结果

    return drawing, lines

def draw_lines(img, lines, color=[0, 0, 255], thickness=1):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def draw_lanes(img, lines, color=[255, 0, 0], thickness=8):
    # a. 划分左右车道
    mid = int(img.shape[1]/2)
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 < mid:
                left_lines.append(line)
                continue
            if x2 > mid:
                right_lines.append(line)
                continue

    if (len(left_lines) <= 0 or len(right_lines) <= 0):
        return

    # b. 清理异常数据
    left_lines = clean_lines(left_lines, 0.1,"l")
    right_lines = clean_lines(right_lines, 0.1,"r")
    
    left_points = []
    right_points = []
    for line in left_lines:
        x1 = line[0][0]
        y1 = line[1][0]
        x2 = line[2][0]
        y2 = line[3][0]
        left_points.append((x1, y1))
        left_points.append((x2, y2))

    for line in right_lines:
        x1 = line[0][0]
        y1 = line[1][0]
        x2 = line[2][0]
        y2 = line[3][0]      
        right_points.append((x1, y1))
        right_points.append((x2, y2))
    h = int(img.shape[0]*0.7)

    left_results = least_squares_fit(left_points, int(img.shape[0]*0.65), img.shape[0])
    right_results = least_squares_fit(right_points, int(img.shape[0]*0.65), img.shape[0])
    
    if left_results[1][0] > left_results[0][0]:
        left_results[1][0] = left_results[0][0]
        
    if right_results[1][0] < right_results[0][0]:
        right_results[1][0] = right_results[0][0]
    
    if left_results[1][1] >= left_results[0][1] or left_results[1][1] <= h: 
        left_results[1][1] = img.shape[0]
        
    if right_results[1][1] >= right_results[0][1] or right_results[1][1] <= h: 
        right_results[1][1] = img.shape[0]
        
    if left_results[0][0] > right_results[0][0]:
        right_results[0][0] = left_results[0][0]

        
        
    # 注意这里点的顺序，从左下角开始按照顺序构造梯形
    vtxs = np.array([[left_results[1], left_results[0], right_results[0], right_results[1]]])


    # d. 填充车道区域
    cv2.fillPoly(img, vtxs, (0, 255, 0))

    # 或者只画车道线
    # cv.line(img, left_results[0], left_results[1], (0, 255, 0), thickness)
    # cv.line(img, right_results[0], right_results[1], (0, 255, 0), thickness)
    
def clean_lines(lines, threshold,string):
    # 迭代计算斜率均值，排除掉与差值差异较大的数据
    slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    slope = np.array(slope)
    lines = np.array(lines)
    lines = np.reshape(lines,(-1,4,1))
    if string == "l":
        maxidx = np.argmax(slope)
    elif string == "r":
        maxidx = np.argmin(slope)
    lines_backup = lines[maxidx,:,:]
    lines_backup = np.array(lines_backup)
    lines_backup = lines_backup[np.newaxis,:,:]
    while len(lines) > 0:
        index = []
        for i in range(len(slope)):
            if slope[i] < 0.65 and slope[i] > -0.65:
                index.append(i)
        slope = np.delete(slope,index,axis = 0)
        lines = np.delete(lines,index,axis = 0)
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        if len(diff) == 0:
            return lines_backup
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slope = np.delete(slope,idx,axis = 0)
            lines = np.delete(lines,idx,axis = 0)
        else:
            break
    return lines
            
def least_squares_fit(point_list, ymin, ymax):
    # 最小二乘法拟合
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]
    


    fit = np.polyfit(y, x, 2)
    fit_fn = np.poly1d(fit)  # 获取拟合的结果
 
    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))
    return [[xmin, ymin], [xmax, ymax]]



def process_an_image(img,img_resize):

    # 高斯滤波核大小
    blur_ksize = 5
    # Canny边缘检测高低阈值
    canny_lth = 50
    canny_hth = 150
     
    # 1. 灰度化、滤波和Canny
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 1)
    cv2.imshow("blur_gray",blur_gray)
    img_abf = cv2.bilateralFilter(blur_gray,9,75,75)
    cv2.imshow("img_abf",img_abf)
    img_abf = cv2.cvtColor(img_abf,cv2.COLOR_BGR2GRAY)
    # Equalize the histogram to account for better contrast in the images.
    img_eh = cv2.equalizeHist(img_abf);
    
    cv2.imshow("img_eh",img_eh)
    edges = cv2.Canny(img_eh, canny_lth, canny_hth)  
    cv2.imshow("edges",edges)    
    rows, cols = edges.shape[0],edges.shape[1]
    
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#     points = np.array([[(0, rows), (int(cols*0.12),int(rows*0.58)), (int(cols*0.88),int(rows*0.58)), (cols, rows)]])
    # [[[0 540], [460 325], [520 325], [960 540]]]
    points_0 = np.array(points)
    points_0 = points_0[np.newaxis,:,:]
    roi_edges = roi_mask(edges, points_0)
    cv2.imshow("roi",roi_edges)
    roi_edges = np.array(roi_edges,dtype = "uint8")*255
#     cv2.imshow("roi",roi_edges)

#     roi_edges = cv2.cvtColor(roi_edges,cv2.COLOR_BGR2GRAY)
    # 霍夫变换参数
    rho = 1
    theta = np.pi / 180
    threshold = 15
    min_line_len = 40
    max_line_gap = 50

    drawing, lines = hough_lines(roi_edges, rho, theta, threshold, min_line_len, max_line_gap)
    if lines is None:
        return img_resize
    draw_lanes(drawing, lines)
    
    result = cv2.addWeighted(img_resize, 0.9, drawing, 0.2, 0)
    return result



def process_frame_pic_backup2(img):
    start = timer()
#     w_0,h_0 = img.shape[0],img.shape[1]
#     new_w = int(w_0*0.8)
#     img = img[:new_w,:,:]
    input_scale = 1
    img_resize = cv2.resize(img, (0,0), fx=1/input_scale, fy=1/input_scale)
     
    img_binary = process_an_image(img_resize,img_resize)
    
    cv2.imshow("aug_frame",img_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#     w_1,h_1 = img.shape[0],img.shape[1]
#     x = np.floor(np.array([int(h_1*0.19),int(h_1*0.39),int(0.566*h_1),int(0.756*h_1)])/input_scale)
#     y = np.floor(np.array([int(w_1*0.95),int(w_1*0.7),int(w_1*0.7),int(w_1*0.95)])/input_scale)
#     x = x[:,np.newaxis]
#     y = y[:,np.newaxis]
#     src_0 = np.concatenate((x,y),axis = 1)
#     
#     X = np.floor(np.array([int(h_1*0.2),int(h_1*0.1),int(0.9*h_1),int(0.78*h_1)])/input_scale)
#     Y = np.floor(np.array([int(w_1*0.95),int(w_1*0.5),int(w_1*0.5),int(w_1*0.95)])/input_scale)
#     X = X[:,np.newaxis]
#     Y = Y[:,np.newaxis]
#     dst_0 = np.concatenate((X,Y),axis = 1)
#     
#     src_1 = np.array(src_0,dtype = np.int32)
#     dst_1 = np.array(dst_0,dtype = np.int32)
#     src_1 = src_1[np.newaxis,:,:]
#     dst_1 = dst_1[np.newaxis,:,:]
#     
#     src = np.float32(src_0) 
#     dst = np.float32(dst_0)
#       
#     M = cv2.getPerspectiveTransform(src, dst)
#     binary_warped = warper(img_binary, M)
# 
#     binary_sub = np.zeros_like(binary_warped)
#     binary_sub[:, int(20/input_scale):int(-20/input_scale)]  = binary_warped[:, int(20/input_scale):int(-20/input_scale)]
#         
#     left_lane = Lane()
#     right_lane = Lane()
#         
#     ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
#     if left_lane.detected:  # start tracker
#         right_lane,left_lane = tracker(binary_sub, right_lane, left_lane, ploty, input_scale, False)
#     else:  # start detector
#         right_lane,left_lane = detector(binary_sub, right_lane, left_lane, ploty, input_scale, False)
#             
#     left_lane.process(ploty)
#     right_lane.process(ploty)
#         
#     lane_width = 3.5
#     curvature, curve_direction = measure_lane_curvature(ploty, left_lane.mean_fitx, right_lane.mean_fitx,
#                                                         lane_width, w_1, input_scale)
#         
#     offcenter, pts = compute_car_offcenter(ploty, left_lane.mean_fitx, right_lane.mean_fitx, 
#                                            lane_width, img_resize)
#         
#     output_frame_scale = 1
#     M_b = M.copy()
#     M_inv = cv2.getPerspectiveTransform(dst, src)
#     end = timer()
#     fps = 1.0 / (end - start)
#     output = create_output_frame(offcenter, pts, img, fps, curvature, curve_direction, output_frame_scale,
#                                  input_scale, w_1, h_1, M_b, M_inv, binary_sub)
# 
#     cv2.imshow("picture",output)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()             
