# **Finding Lane Lines on the Road** 

## Lane Finding Pipeline 

- convert to **grayscale**.

- reuduce noise by Gaussian smoothing ,choose the **kernel_size**.

- Canny edge detector，using **opencv Canny function,**two parameters: **low_threshold** and **high_threshold** are thresholds for edge detection.John Canny recommended a low to high ratio of 1:2 or 1:3.

- Region of Interest Mask,using a **quadrilateral mask**.

- using **huogh transform algorithm** find the lines. parameters are:

  **rho**-distance resolution of the accumulator in pixels

  **theta**-angel resolution of the accumulator in radians

  **threshold**-accumulator threshole parameter

  **min_line_len**-minimum line length,line segments shorter than that are rejected.

  **max_line_gap**-Maximum allowed gap between points on the same line to link them.

  The output of function is vector of lines,each line represented by a 4-element vector(x1,y1,x2,y2).

```python
def lane_find(image):
    kernel_size=3
    low_threshold = 50
    high_threshold = 180

    gray = grayscale(image)
    blur_gray=gaussian_blur(gray, kernel_size)
    edges=canny(blur_gray, low_threshold, high_threshold)
    
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(imshape[1]*12/25, imshape[0]*6/10), (imshape[1]*14/25, imshape[0]*6/10), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges =region_of_interest(edges, vertices)

    rho = 1 
    theta = np.pi/180 
    threshold = 15     
    min_line_len = 50 
    max_line_gap = 30    

    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap) 
    color_edges = np.dstack((edges, edges, edges)) 
    lines_edges=weighted_img(image, line_image, α=0.8, β=1., γ=0.)
    return lines_edges
```

## modified **draw_lines** function

- separating line segments by their slope ((y2-y1)/(x2-x1)) to left and right.
- Fit an equation left and right .
- find the y axis of the up point,set the y axis of the bottom ,left and right lane .
- according to left and right equation get  x axis of points
- draw the line on image

```python
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    x_right=[]
    y_right=[]
    x_left=[]
    y_left=[]
    imshape = image.shape
    #separating line segments by their slope ((y2-y1)/(x2-x1)) to left and right
    for line in lines:
        for x1,y1,x2,y2 in line:
            slop=((y2-y1)/(x2-x1))
            if slop>0.01 :
                x_right.extend((x1,x2))
                y_right.extend((y1,y2))
            elif slop<-0.01 :
                x_left.extend((x1,x2))
                y_left.extend((y1,y2))
    # Fit an equation left and right
    slope_left,intercept_left =np.polyfit(x_left,y_left,1)
    slope_right,intercept_right =np.polyfit(x_right,y_right,1)
    #Find the up point of the y axis
    y2l=min(y_left)
    y2r=min(y_right)
    #set the bottom point of the yx axis
    y1l=imshape[0]
    y1r=imshape[0]
    
    #Find the x axis from the equation  
    x2l=int((y2l- intercept_left) / slope_left)
    x2r=int((y2r- intercept_right) / slope_right)
    x1l=int((y1l- intercept_left) / slope_left)
    x1r=int((y1r- intercept_right) / slope_right)
    
    #draw the line based on two points 
    cv2.line(img, (x1r, y1r), (x2r, y2r), color, thickness)
    cv2.line(img, (x1l, y1l), (x2l, y2l), color, thickness)
```

# Reflection


##  potential shortcomings of my current pipeline

- Parameters can be tuned achieves better performance.
- It would not work if  other lines （which are not lane lines）detect  in ROI.




## Suggest possible improvements my pipeline

- Use color information to mask can filter the noise which are not lane lines