import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

dist_pickle = pickle.load(open( "camera_cal.p", "rb" ))

# window settings
window_width = 50 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids

def findLanes(binary_warped, visualize=False):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    xpixels = histogram.shape[0]
    #leftx_base = np.argmax(histogram[:midpoint])
    #rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    left_start = int(0.15*xpixels)
    leftx_base = np.argmax(histogram[left_start:int(0.45*xpixels)]) + left_start
    right_start = int(0.55*xpixels)
    rightx_base = np.argmax(histogram[right_start:int(0.85*xpixels)]) + right_start

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
    margin = 70
    # Set minimum number of pixels found to recenter window
    minpix = 50
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
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
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
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    if visualize:
        f, ax1 = plt.subplots(1, 1, figsize=(10,10))

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()


    return (left_fitx, right_fitx, ploty, out_img, left_fit, right_fit)


def findNextLane(binary_warped, left_fit, right_fit, visualize=False):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 70
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin))
                      & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin))
                       & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    if visualize:
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        f, ax1 = plt.subplots(1, 1, figsize=(10,10))
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()
    
    return (left_fitx, right_fitx, ploty, out_img, left_fit, right_fit)

def thresholdImage(img, sobelx_thres=(30, 100), s_thres = (170, 255), visualize=False):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = sobelx_thres[0]
    thresh_max = sobelx_thres[1]
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = s_thres[0]
    s_thresh_max = s_thres[1]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    sobelxs = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelxs = np.absolute(sobelxs) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobels = np.uint8(255*abs_sobelxs/np.max(abs_sobelxs))
    ssxbinary = np.zeros_like(scaled_sobels)
    ssxbinary[(scaled_sobels >= thresh_min) & (scaled_sobels <= thresh_max)] = 1
    
    l_thres = (30, 100)
    l_channel = hls[:,:,1]
    l_thresh_min = l_thres[0]
    l_thresh_max = l_thres[1]

    sobelxl = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelxl = np.absolute(sobelxl) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobell = np.uint8(255*abs_sobelxl/np.max(abs_sobelxl))
    l_binary = np.zeros_like(l_channel)
    l_binary[(scaled_sobell >= l_thresh_min) & (scaled_sobell <= l_thresh_max)] = 1


    
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    # Sobel x channel - green
    # s channel - blue
    color_binary = np.dstack(( l_binary, sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1) | (l_binary == 1)] = 1

    #color_binary = np.dstack(( l_binary, ssxbinary, s_binary))
    #combined_binary = np.zeros_like(s_binary)
    #combined_binary[(s_binary == 1) | (ssxbinary == 1) | (l_binary == 1)] = 1


    # Plotting thresholded images
    if visualize:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('H channel')
        ax1.imshow(hls[:,:,0])
        ax1.grid('on')
        ax2.set_title('L channel')
        ax2.imshow(hls[:,:,1])
        ax2.grid('on')
        plt.show()
        
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('S channel')
        ax1.imshow(hls[:,:,2])
        ax1.grid('on')
        ax2.set_title('image')
        ax2.imshow(img)
        ax2.grid('on')
        plt.show()

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))
        yi = 600
        ax1.set_title('HLS at y={}'.format(yi))
        ax1.plot(s_channel[yi,:])
        ax1.plot(l_channel[yi,:])
        ax1.plot(hls[yi,:,0])
        ax1.plot(scaled_sobels[yi,:])
        ax1.legend(('S', 'L', 'H', 'Sx'))

        yi = 475
        ax2.set_title('l grad at y={}'.format(yi))
        ax2.plot(scaled_sobell[yi,:])

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,5))
        yi = 472
        ax1.set_title('l grad at y={}'.format(yi))
        ax1.plot(scaled_sobell[yi,:])

        yi = 470
        ax2.set_title('l grad at y={}'.format(yi))
        ax2.plot(scaled_sobell[yi,:])


        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('L gradient threshold')
        ax1.imshow(l_binary, cmap='gray')

        ax2.set_title('S channel threshold')
        ax2.imshow(s_binary, cmap='gray')
        plt.show()

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('Stacked thresholds')
        ax1.imshow(color_binary*255)

#        ax2.set_title('Combined S channel and gradient thresholds')
#        ax2.imshow(combined_binary, cmap='gray')
#        plt.show()

        ax2.set_title('S gradient threshold')
        ax2.imshow(ssxbinary, cmap='gray')
        plt.show()


        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

    view = np.dstack((combined_binary, combined_binary, combined_binary))*255
    img_size = (view.shape[1], view.shape[0])

    slope = (680.-430.) / (265.-627.)
    ydes = 450
    xdes = int((ydes-680.) / slope + 265.)

    slope2 = (430.-680.) / (655.-1049.)
    xdes2 = int((ydes-680.)/slope2 + 1045.)

    src = np.float32([[265, 680], [xdes, ydes], [xdes2, ydes], [1045, 680]])
    dst = np.float32([[400, 700], [400, 100], [880, 100], [880, 700]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    if visualize:
        thickness=2
        cv2.line(view, (src[0][0], src[0][1]), (src[1][0], src[1][1]), [0, 255, 0], thickness)
        cv2.line(view, (src[3][0], src[3][1]), (src[2][0], src[2][1]), [0, 255, 0], thickness)
        cv2.line(view, (src[1][0], src[1][1]), (src[2][0], src[2][1]), [0, 255, 0], thickness)
        ax1.imshow(view)
    
    warped = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)

    if visualize:
        warpedimg = np.dstack((warped, warped, warped))*255
        cv2.line(warpedimg, (dst[0][0], dst[0][1]), (dst[1][0], dst[1][1]), [0, 255, 0], thickness)
        cv2.line(warpedimg, (dst[3][0], dst[3][1]), (dst[2][0], dst[2][1]), [0, 255, 0], thickness)

        ax2.imshow(warpedimg)
        plt.show()
    return warped, Minv

def highlightLane(undist, warped, left_fitx, right_fitx, ploty, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

def findAngleOfCurvature(leftx, rightx, ploty, xdim, lanewidth_px=440, lanelength_px=720):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/lanelength_px # meters per pixel in y dimension
    xm_per_pix = 3.7/lanewidth_px # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    y_eval = np.max(ploty)
    # Calculate the new radii of curvature
    left_curverad = ( ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) 
                      / np.absolute(2*left_fit_cr[0]) )
    right_curverad = ( ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5)
                       / np.absolute(2*right_fit_cr[0]) )
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    
    posCenter = ((leftx[-1] + rightx[-1]) - xdim)/2 * xm_per_pix
    #print(posCenter, 'm')

    return (left_curverad, right_curverad, posCenter)


def annotateImage(img, leftRad, rightRad, posCenter):
    ls = "left radius: {0:.0f} m".format(leftRad)
    rs = "right radius: {0:.0f} m".format(rightRad)
    cs = "dist: {0:.2f} m".format(posCenter)

    cv2.putText(img, ls, (100, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3)
    cv2.putText(img, rs, (100, 60), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3)
    cv2.putText(img, cs, (100, 90), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3)


def processImage(img, sobelx_thres=(20, 100), s_thres = (170, 255)):
    global left_fit
    global right_fit
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']
    img = cv2.undistort(img, mtx, dist, None, mtx)
    warped, Minv = thresholdImage(img, sobelx_thres, s_thres, visualize=True)
    left_fitx, right_fitx, ploty, out_img, left_fit, right_fit = findLanes(warped, visualize=True)
    aocl, aocr, posCenter = findAngleOfCurvature(left_fitx, right_fitx, ploty, img.shape[1])
    result = highlightLane(img, warped, left_fitx, right_fitx, ploty, Minv)
    annotateImage(result, aocl, aocr, posCenter)
    return result

def processImageSeq(img, sobelx_thres=(20, 100), s_thres = (170, 255), visualize=False):
    global left_line, right_line
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']
    img = cv2.undistort(img, mtx, dist, None, mtx)
    warped, Minv = thresholdImage(img, sobelx_thres, s_thres, visualize=visualize)
    if (left_line.best_fit != None) and (right_line.best_fit != None):
        left_fitx, right_fitx, ploty, out_img, left_fit, right_fit = findNextLane(
        warped, left_fit, right_fit, visualize=visualize)        
    else:
        left_fitx, right_fitx, ploty, out_img, left_fit, right_fit = findLanes(warped, visualize=visualize)

    aocl, aocr, posCenter = findAngleOfCurvature(left_fitx, right_fitx, ploty, img.shape[1])
    result = highlightLane(img, warped, left_fitx, right_fitx, ploty, Minv)
    annotateImage(result, aocl, aocr, posCenter)
    
#    if left_line.recent_xfitted :
#    left_line.recent_xfitted.append(left_fitx[-1])
#    left_line.recent_xfitted.popleft()
#    right_line.recent_xfitted.append(right_fitx[-1])
#    right_line.recent_xfitted.popleft()
#
#    left_line.bestx = np.mean(left_line.recent_xfitted)
#    right_line.bestx = np.mean(right_line.recent_xfitted)
    
    
    
    return result

# Define a class to receive the characteristics of each line detection
from collections import deque
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = deque([])
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = deque([np.array([False])])
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

left_line = Line()
right_line = Line()

def resetLaneFinder():
    global left_fit, right_fit, left_line, right_line
    left_fit = None
    right_fit = None
    left_line = Line()
    right-line = Line()