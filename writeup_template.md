## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistort]: ./output_images/undistort.png "Undistorted"
[undistortroad]: ./output_images/undistortedroad.png "Road Transformed"
[binary]: ./output_images/binary.png "Binary Example"
[lanefit]: ./output_images/lanefit.png "Fit Visual"
[annotated]: ./output_images/annotated.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./camera_calibration.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][undistort]

I take the transformation matrices and write them to a pickled file camera_cal.p for usage during lane detection.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][undistortroad]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 216 through 248 in `laneutils.py`).  Here's an example of my output for this step. The image was transformed into grayscale and HLS. I used the Sobel transform on the x dimension of the grayscaled image, Sobel x on the L channel, and value thresholding on the S channel.

![alt text][binary]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is on lines 335 through 354 in the file `laneutils.py`. I chose the hardcode the source and destination points in the following manner:

```python
   slope = (680.-430.) / (265.-627.)
    ydes = 450
    xdes = int((ydes-680.) / slope + 265.)

    slope2 = (430.-680.) / (655.-1049.)
    xdes2 = int((ydes-680.)/slope2 + 1045.)

    src = np.float32([[265, 680], [xdes, ydes], [xdes2, ydes], [1045, 680]])
    dst = np.float32([[400, 700], [400, 100], [880, 100], [880, 700]])
```

I pick points along straight lane lines to define the transformation trapezoid. I can slide up and down this trapezoid to define the far point of the transform using the variable ydes. This transforms to a fixed rectangular box in dst.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][binary]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

`findLanes` function in laneutils.py starting line 55 uses the histogram method in a box to look for lane lines. I narrowed the box from the lecture to be 15% - 45% of the x dimension for the left lane and 55% to 85% for the right lane. Given my transformation points, I expect the lane to fall within this box and eliminate detection of other boundaries like railings.

Then starting from the bottom successive boxes in the y dimension are used to select pixels for fitting the lane line polynomial. Each successive box is recentered around the mean x position of the thresholded pixels previous box. The box has a width of `margin=70` pixels.

This cloud of x,y points that are presumably associated with the lane line is used with polyfit to find a best fit 2nd order polynomial to all these points.

Once a polynomial is identified, subsequent frames use the polynomial from the previous detection to determine a window of `margin=70` pixels to select pixels for polynomial fitting for updating the lane lines.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 399 through 417 in my code in `laneutils.py`. Using the formula provided in the lectures, the 2nd order polynomial coefficients determine the radius of curvature, using appropriate scalings between pixels to real world distance. The position of the lane lines relative to the video frame is computed by taking the bottom pixels of the left and right lane polynomial expansions, finding the midpoint, and subtracting from the middle of the frame. This is converted from pixels to meters using the same scale factors for curvature calculation.

![alt text][lanefit]


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 365 through 382 in my code in `yet_another_file.py` in the function `map_lane()`.  This takes the plotted lane lines, fills a polygon region, and warps it back to the lane perspective. I also annotated the image with radius of curvature and distance from center using cv2.putText. Here is an example of my result on a test image:

![alt text][annotated]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I followed the approach given in the lecture.
1. Undistort the image
2. Convert to HLS and grayscale, find appropriate thresholds to identify lanes on the channels
3. Transform the perspective to find a radius of curvature and identify distances

I then smooth the detection results by 10 frames, and have sanity checks to make sure the updates do not diverge too much from the averages.

It is difficult to find a robust set of thresholds given the variation of road surfaces. Color appearance can also be affected by lighting, such as night driving.

A further improvement would be to spend more time on tuning the thresholds and make it adaptive to different road conditions.

Another problem I notice is that when the vehicle goes over a bump, the suspension of the vehicle changes pitch or the camera mount wobbles. This skews the transformation and makes the assumptions for the perspective transformation incorrect. Also, going up or down a hill also violates the assumptions. The horizon is no longer where it is expected to be.

To make this more robust, there needs to be 
1. image stabilization
2. horizon detection
3. vehicle slope detection
4. automatic identification of features that can be used to transform the perspective, e.g. straight lane line detection, or something more sophisticated.
