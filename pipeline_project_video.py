import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg


# Define a class to receive the characteristics of each line detection
class Line_nature():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # finding 4 warped points(a,b,c,d) in the left
        self.recent_leftpoint = []
        # finding 4 warped points(a,b,c,d) in the right
        self.recent_rightpoint = []
        # judge whether the pipline is straight line or curval line 
        self.is_straight_line = False
        # the value of radius_of_curvature
        self.radius_of_curvature = None
        # fit of pipline in the left of road 
        self.recent_leftfit = []
        # fit of pipline in the right of road
        self.recent_rightfit = []
        # the distance between vehicle position and center of road 
        self.recent_diff_center = None
        

def camera_calibration(images_path, image_size):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)   
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(images_path+'/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
         
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners, ret)

    cv2.destroyAllWindows()        
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size,None,None)
    
    #save the mtx and dist as calib_dist_pickle.p
    out_path =  images_path+"/calib_dist_pickle.p"   
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( out_path, "wb" ) )
     
    return out_path


def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def r_select(img, thresh=(200, 255)):
    R = img[:,:,0]
    binary = np.zeros_like(R)
    binary[(R > thresh[0]) & (R <= thresh[1])] = 1
    return binary

def color_mask(hsv,low,high):
    # Return mask from HSV 
    mask = cv2.inRange(hsv, low, high)
    return mask

def apply_color_mask(hsv,img,low,high):
    # Apply color mask to image
    mask = cv2.inRange(hsv, low, high)
    res = cv2.bitwise_and(img,img, mask= mask)
    return res

def apply_yellow_white_mask(img):
    image_HSV = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    yellow_hsv_low  = np.array([ 0,  100,  100])
    yellow_hsv_high = np.array([ 80, 180, 255])
    white_hsv_low  = np.array([ 90,   0,   170])
    white_hsv_high = np.array([ 160,  80, 255])   
    mask_yellow = color_mask(image_HSV,yellow_hsv_low,yellow_hsv_high)
    mask_white = color_mask(image_HSV,white_hsv_low,white_hsv_high)
    mask_YW_image = cv2.bitwise_or(mask_yellow,mask_white)
    return mask_YW_image


def hls_select(img, channel='S', thresh=(90, 255)):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel == 'S':
        X = hls[:, :, 2]
    elif channel == 'H':
        X = hls[:, :, 0]
    elif channel == 'L':
        X = hls[:, :, 1]
    else:
        #print('illegal channel !!!')
        return
    binary_output = np.zeros_like(X)
    binary_output[(X > thresh[0]) & (X <= thresh[1])] = 1
    return binary_output

def combine_filters(img):
    r_binary = r_select(undist_pipline, thresh=(220, 255))
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(35, 255))
    l_binary = hls_select(img, channel='L', thresh=(100, 200))
    s_binary = hls_select(img, channel='S', thresh=(175, 255))
    yw_binary = apply_yellow_white_mask(undist_pipline)
    yw_binary[(yw_binary !=0)] = 1
    combined_lsx = np.zeros_like(gradx)
    #combined_lsx[((l_binary == 1) & (s_binary == 1) | (gradx == 1) | (yw_binary == 1))] = 1
    combined_lsx[((s_binary==1)&(l_binary ==1)|(yw_binary ==1)|(r_binary == 1))] = 1
    return combined_lsx

class Point(object):
    x =0
    y= 0
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class Line(object):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2


def GetLinePara(line):
    line.a =line.p1.y - line.p2.y;
    line.b = line.p2.x - line.p1.x;
    line.c = line.p1.x *line.p2.y - line.p2.x * line.p1.y;


def GetCrossPoint(l1,l2):

    GetLinePara(l1);
    GetLinePara(l2);
    d = l1.a * l2.b - l2.a * l1.b
    if d ==0:
        return -1
    
    p=Point()
    p.x = (int)(l1.b * l2.c - l2.b * l1.c)*1.0 / d
    p.y = (int)(l1.c * l2.a - l2.c * l1.a)*1.0 / d
    return p;


def slop_line(point1, point2):
    a = (point1.y - point2.y)/(point1.x - point2.x)
    b = point1.y - point1.x*a
    return a, b
    

def region_thresholds(x,y):
    left_bottom = [200,680]
    right_bottom = [1130,680]
    apex_right = [740,460]
    apex_left = [580,460]

    
    fit_left = np.polyfit((left_bottom[0], apex_left[0]), (left_bottom[1], apex_left[1]), 1)
    fit_top = np.polyfit((apex_left[0], apex_right[0]), (apex_left[1], apex_right[1]), 1)
    fit_right = np.polyfit(( right_bottom[0], apex_right[0]), (right_bottom[1],apex_right[1]), 1)
    fit_bottom = np.polyfit((right_bottom[0], left_bottom[0]), (right_bottom[1], left_bottom[1]), 1)

    if((y > (x*fit_left[0] + fit_left[1])) & (y > (x*fit_top[0] + fit_top[1])) & (y > (x*fit_right[0] + fit_right[1])) & (y < (x*fit_bottom[0] + fit_bottom[1]))):
        return 1
    else:
        return 0 
    
    
def perspective_transform(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


def find_line_fit(img, nwindows=9, margin=100, minpix=50):
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
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
    
    # to plot
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Fit a second order polynomial to each
    #left_fit = np.polyfit(lefty, leftx, 2)
    #right_fit = np.polyfit(righty, rightx, 2)
    return out_img, lefty, leftx, righty, rightx


# Generate x and y values for plotting
def get_fit_xy(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return left_fitx, right_fitx, ploty


def project_back(wrap_img, origin_img, left_fitx, right_fitx, ploty, M):
    warp_zero = np.zeros_like(wrap_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = perspective_transform(color_warp, M)
    # Combine the result with the original image
    result = cv2.addWeighted(origin_img, 1, newwarp, 0.3, 0)
    return result

def radius_curvature(ploty, left_fit, right_fit):
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    #print(left_curverad, right_curverad)

    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    left_curverad = round(left_curverad, 2)
    right_curverad = round(right_curverad, 2)

    #print(left_curverad, 'm', right_curverad, 'm')
    return left_curverad, right_curverad

def distance_center(image, leftx, rightx, last_diff, iflag):
    # meters per pixel in x dimension
    xm_per_pix = 3.7/700 
    # the vehicle position
    image_midpoint = np.int(image.shape[1]/2)
    # the center of pipline
    center_pipline = (leftx + rightx)/2
    #the distance between vehicle position and center of pipline  
    diff_center = np.absolute((center_pipline - image_midpoint)//2) * xm_per_pix
    # when the pipline of last frame is good 
    if iflag == 1:
        diff_center = (diff_center + last_diff)*0.5
        
    diff_center = round(diff_center, 2)

    if image_midpoint > center_pipline:
        vehicle_position = 'vehicle is '+str(diff_center)+'m right of center'
    if image_midpoint < center_pipline:
        vehicle_position = 'vehicle is '+str(diff_center)+' m left of center'
    if image_midpoint == center_pipline:
        vehicle_position = 'vehicle is in the center '
      
    return vehicle_position, diff_center

def text_input_image(image, text_curature, text_distance):
    #  the text font
    font=cv2.FONT_HERSHEY_SIMPLEX
    image_show = np.zeros_like(image, np.uint8)
    # the content of txt
    left_curveration_txt = 'Radius of Curature = '+str(text_curature)+' m'

    img=cv2.putText(image_show,left_curveration_txt,(200,100),font,1.5,(255,255,255),3)
    img=cv2.putText(image_show,text_distance,(200,150),font,1.5,(255,255,255),3)
    img_input = cv2.addWeighted(image, 0.8, img, 1, 0)
       
    return img_input

def diff_slops_straight_line(img, lefty, leftx, righty, rightx):

    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    left_fit = np.polyfit(lefty, leftx, 1)
    right_fit = np.polyfit(righty, rightx, 1)
    line_left_fitx = left_fit[0]*ploty + left_fit[1]
    line_right_fitx = right_fit[0]*ploty + right_fit[1]
    # judge different between leftline and rightline when the pipline is straight line
    diff_slop = np.absolute(left_fit[0] - right_fit[0])
    
    return diff_slop




# camera calibration
import_image = 'camera_cal/calibration1.jpg'
image = cv2.imread(import_image)

file_patch = 'camera_cal'
image_size = (image.shape[1], image.shape[0])

# function camera_calibration save  mtx, and dist that from cv2.calibrateCamera in the calib_dist_pickle.p  
out_path = camera_calibration(file_patch, image_size)
# load  the mtx and dist
dist_pickle = pickle.load( open(out_path, "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# define a Line() class to keep track of all the interesting parameters I measure from frame to frame
recent_line = Line_nature()
current_line = Line_nature()

# save video
video_name='project_video.mp4'
save_out_video = './project_video_output.avi'

cap = cv2.VideoCapture(video_name)    
fps = cap.get(cv2.CAP_PROP_FPS)  
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')  
outVideo = cv2.VideoWriter(save_out_video,fourcc,fps,size) 


if cap.isOpened():  
    rval,frame = cap.read()  
    print('start')  
else:  
    rval = False  
    print('False')  

tot=1     
    
while rval:
    rval,origin_img = cap.read() 

    try:
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
    except:
        current_line.detected = False
        continue

    # Distortion-corrected image
    undist_pipline = cv2.undistort(origin_img, mtx, dist, None, mtx)

    # Apply each of the thresholding function
    binary = combine_filters(undist_pipline)

    combined_binary_image = np.dstack(( binary, binary, binary))*255


    # Looking for 4 warped points(a,b,c,d) 
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 3     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 6 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
    
    line_image = np.copy(combined_binary_image)*0 # creating a blank to draw lines on
    region_binary = np.copy(binary) 
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    #lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
    lines = cv2.HoughLinesP(region_binary, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)
        
    
    # gather left_points right_point from the lines that have been classfied by hough
    left_points = []
    right_points =[]
    center_col = 640 # in order to classfy left points and right points
    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1,y1,x2,y2 in line:
            if (region_thresholds(x1,y1)):
                if x1 < center_col:
                    left_points.append((x1,y1))
                if x1 > center_col:
                    right_points.append((x1,y1))
                if(region_thresholds(x2,y2)):
                    if x2 < center_col:
                        left_points.append((x2,y2))
                    if x2 > center_col:
                        right_points.append((x2,y2))
                                    
                                                                 
    #left_pipline
    rows,cols = line_image.shape[:2]
    points = np.array(left_points)
    
    try:
        [vx,vy,x,y] = cv2.fitLine(points, cv2.DIST_L2,1,0.01,0.01)
    except:
        current_line.detected = False
        continue
    
    line_lefty = int((-x*vy/vx)+y)
    line_righty = int(((cols-x)*vy/vx)+y)

    #limit the lenth of left line
    p1=Point(cols-1,line_righty)
    p2=Point(0,line_lefty)
    line1=Line(p1,p2)
    p3=Point(0,470)
    p4=Point(1210,470)
    line2=Line(p3,p4)
    
    Pc = GetCrossPoint(line1,line2);
    if Pc == -1:
        current_line.detected = False
        continue
           
    # point a
    ax = int(Pc.x)
    ay = int(Pc.y)                  
    
    #second line
    p5=Point(0,720)
    p6=Point(1210,720)
    line3=Line(p5,p6)
    Pc = GetCrossPoint(line1,line3);
    # point d
    dx = int(Pc.x)
    dy = int(Pc.y)  
    
    #line_image = cv2.line(line_image,(0,line_lefty),(ax,ay),(255,0,0),3)
        
    #right_line
    points = np.array(right_points)
    try:
        [vx,vy,x,y] = cv2.fitLine(points, cv2.DIST_L2,1,0.01,0.01)
    except:
        current_line.detected = False
        continue 

    line_lefty = int((-x*vy/vx) + y)
    line_righty = int(((cols-x)*vy/vx)+y)

            
    #limit the lenth of right line
    p1=Point(cols-1,line_righty)
    p2=Point(0,line_lefty)
    line1=Line(p1,p2)
    p3=Point(0,470)
    p4=Point(1210,470)
    line2=Line(p3,p4)
    

    Pc = GetCrossPoint(line1,line2);
    if Pc == -1:
        current_line.detected = False
        continue
    #point b
    bx = int(Pc.x)
    by = int(Pc.y)

    #second line
    Pc = GetCrossPoint(line1,line3);
    #point c
    cx = int(Pc.x)
    cy = int(Pc.y)  
    
    #line_image = cv2.line(line_image,(cols-1,line_righty),(bx,by),(255,0,0),3)
    #lines_edges = cv2.addWeighted(line_image, 0.8, combined_binary_image, 1, 0) 
    
    # if it is a bad fitpoint, delete this frame
    if ((ax > center_col)|(ax < 0)| (dx > center_col)|(dx <0)):
        recent_line.detected = False
        continue
    if ((bx < center_col)|(bx > 1280)|(cx < center_col)|(cx >1280)):
        recent_line.detected = False
        continue
    
   
    #avarage with last frame lines    
    if(recent_line.detected== True):
        ax = int((recent_line.recent_leftpoint[0][0]+ax)*0.5)
        ay = int((recent_line.recent_leftpoint[0][1]+ay)*0.5)
        dx = int((recent_line.recent_leftpoint[1][0]+dx)*0.5)
        dy = int((recent_line.recent_leftpoint[1][1]+dy)*0.5)
        bx = int((recent_line.recent_rightpoint[0][0]+bx)*0.5)
        by = int((recent_line.recent_rightpoint[0][1]+by)*0.5)
        cx = int((recent_line.recent_rightpoint[1][0]+cx)*0.5)
        dy = int((recent_line.recent_rightpoint[1][1]+cy)*0.5)
    

    current_line.recent_leftpoint = []
    current_line.recent_rightpoint = []    
    current_line.recent_leftpoint.append([ax,ay])
    current_line.recent_leftpoint.append([dx,dy])
    current_line.recent_rightpoint.append([bx,by])
    current_line.recent_rightpoint.append([cx,cy])
        

    
    # "birds-eye view"
    offset = 115
    img_size = (cols,rows)

    # Perspective Transform src_corners and dist corners
    src_corners = np.float32([(ax, ay), (bx, by), (cx, cy), (dx, dy)])
    dst_corners = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                              [img_size[0]-offset, img_size[1]-offset], [offset, img_size[1]-offset]])

    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    Minv = cv2.getPerspectiveTransform(dst_corners, src_corners)
    warped_image = cv2.warpPerspective(binary, M, img_size, flags = cv2.INTER_LINEAR )

    # Detect lane pixels and fit to find the lane boundary
    out_img, lefty, leftx, righty, rightx = find_line_fit(warped_image)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)   
    
    if(recent_line.detected== True):
        if(recent_line.is_straight_line== False):
            left_fit[0] = (recent_line.recent_leftfit[0]+left_fit[0])*0.5
            left_fit[1] = (recent_line.recent_leftfit[1]+left_fit[1])*0.5
            left_fit[2] = (recent_line.recent_leftfit[2]+left_fit[2])*0.5
            right_fit[0] = (recent_line.recent_rightfit[0]+left_fit[0])*0.5
            right_fit[1] = (recent_line.recent_rightfit[1]+left_fit[1])*0.5
            right_fit[2] = (recent_line.recent_rightfit[2]+left_fit[2])*0.5

    current_line.recent_leftfit = left_fit
    current_line.recent_rightfit = right_fit
    
    left_fitx, right_fitx, ploty = get_fit_xy(warped_image, left_fit, right_fit)


    #Warp the detected lane boundaries back onto the original image
    result = project_back(binary, undist_pipline, left_fitx, right_fitx, ploty, Minv)

    #Determine the curvature of the lane and vehicle position with respect to center.
    left_curverad, right_curverad = radius_curvature(ploty, left_fit, right_fit)  
    current_line.is_straight_line = False

    
    # judge whether is straight line , if the pipline is straight, set the value of curvature is 0 
    diff_curverad = np.absolute(left_curverad - right_curverad)
    # judge whether curvature is too big, and then whether the pipline is straight or it fit line is not good
    if ((left_curverad>=1200) ):
        diff_slop = diff_slops_straight_line(warped_image,lefty, leftx, righty, rightx)
        if diff_slop < 0.2:
            left_curverad = 0
            right_curverad = 0
            recent_line.is_straight_line = True
        else:
            recent_line.detected = False
            continue
			
    # the right line is easy to appear bad fitline    
    if (right_curverad >=1200)&(left_curverad < 1200):
        right_curverad = left_curverad;
        
    if (right_curverad < 1200)&(left_curverad < 1200)&(diff_curverad>=500):
        average_curverad = 0.8*left_curverad + 0.2*right_curverad
    else:
        average_curverad = 0.5*left_curverad + 0.5*right_curverad
        
    if(recent_line.detected== True):
        if(recent_line.is_straight_line == False):
            average_curverad = (average_curverad + recent_line.radius_of_curvature)*0.5

   
    average_curverad = round(average_curverad,2)
    current_line.radius_of_curvature = average_curverad
     
    # if the difference between vehicle position and the center is exist, iflag is 1 that is used in distance_center function   
    last_diff_center = 0.00
    iflag = 0
    if(recent_line.detected== True):
        last_diff_center = recent_line.recent_diff_center
        iflag = 1   

    vehicle_position_text, current_diff_center = distance_center(undist_pipline, dx, cx, last_diff_center, iflag)
    current_line.recent_diff_center = current_diff_center

    result_text = text_input_image(result, average_curverad, vehicle_position_text)
    
    tot+=1
    print('tot',tot)

    #output_image = 'save_video_7/image_out'+str(tot)+'.jpg'
       
    result_text = result_text[:,:,::-1]
    #cv2.imwrite(output_image, result_text)   
    
    recent_line = current_line
    
    outVideo.write(result_text)
    cv2.waitKey(1)

cap.release()  
outVideo.release()  
cv2.destroyAllWindows() 
    
print('saved finished')
    
