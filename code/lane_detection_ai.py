#Import necessary libraries
import cv2 
import numpy as np
import torch
from torchvision import transforms

# Load pretrained model
model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
model.eval()

# model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=False)
# model.load_state_dict(torch.load("custom_lane_model.pth"))
# model.eval()



transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


#Function to preprocess the image to detect yellow and white lanes
def preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gblur = cv2.GaussianBlur(gray,(5,5),0)
    white_mask = cv2.threshold(gblur,200,255,cv2.THRESH_BINARY)[1]
    lower_yellow = np.array([0,100,100])
    upper_yellow = np.array([210,255,255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return mask

#Function that defines the polygon region of interest
def regionOfInterest(img, polygon):
    mask = np.zeros_like(img)
    x1, y1 = polygon[0]
    x2, y2 = polygon[1]
    x3, y3 = polygon[2]
    x4, y4 = polygon[3]
    m1 = (y2-y1)/(x2-x1)
    m2 = (y3-y2)/(x3-x2)
    m3 = (y4-y3)/(x4-x3)
    m4 = (y4-y1)/(x4-x1)
    b1 = y1 - m1*x1
    b2 = y2 - m2*x2
    b3 = y3 - m3*x3
    b4 = y4 - m4*x4
    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if i>=m1*j+b1 and i>=m2*j+b2 and i>=m3*j+b3 and i<=m4*j+b4:
                mask[i][j] = 1

    masked_img = np.multiply(mask, img)
    return masked_img

#Function that warps the image
def warp(img, source_points, destination_points, destn_size):
    matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    warped_img = cv2.warpPerspective(img, matrix, destn_size)
    return warped_img

#Function that unwarps the image
def unwarp(img, source_points, destination_points, source_size):
    matrix = cv2.getPerspectiveTransform(destination_points, source_points)
    unwarped_img = cv2.warpPerspective(img, matrix, source_size)
    return unwarped_img

#Function that gives the left fit and right fit curves for the lanes in birdeye's view
def fitCurve(img):
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 50
    margin = 100
    minpix = 50
    window_height = int(img.shape[0]/nwindows)
    y, x = img.nonzero()
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_indices = []
    right_lane_indices = []
    
    for window in range(nwindows):
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_indices = ((y >= win_y_low) & (y < win_y_high) & (x >= win_xleft_low) & (x < win_xleft_high)).nonzero()[0]
        good_right_indices  = ((y >= win_y_low) & (y < win_y_high) & (x >= win_xright_low) & (x < win_xright_high)).nonzero()[0]
        left_lane_indices.append(good_left_indices)
        right_lane_indices.append(good_right_indices)
        if len(good_left_indices) > minpix:
            leftx_current = int(np.mean(x[good_left_indices]))
        if len(good_right_indices) > minpix:
            rightx_current = int(np.mean(x[good_right_indices]))
        
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)
    leftx = x[left_lane_indices]
    lefty = y[left_lane_indices]
    rightx = x[right_lane_indices]
    righty = y[right_lane_indices]

    # SAFETY CHECK (VERY IMPORTANT)
    # if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
    #     return None, None

    if len(leftx) < 50 or len(rightx) < 50:
        return None, None

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    

    return left_fit, right_fit

#Function that give pixel location of points through which the curves of detected lanes passes
def findPoints(img_shape, left_fit, right_fit):
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    return pts_left, pts_right

#Function that fills the space between the detected lane curves
def fillCurves(img_shape, pts_left, pts_right):
    pts = np.hstack((pts_left, pts_right))
    img = np.zeros((img_shape[0], img_shape[1], 3), dtype='uint8')
    cv2.fillPoly(img, np.int_([pts]), (0,0, 255))
    return img

#Function that converts a one channel image into a three channel image
def oneToThreeChannel(binary):
    img = np.zeros((binary.shape[0], binary.shape[1], 3), dtype='uint8')
    img[:,:,0] = binary
    img[:,:,1] = binary
    img[:,:,2] = binary
    return img

#Function that draws the curves of detected lanes on an image
def drawCurves(img, pts_left, pts_right):
    img = oneToThreeChannel(img)
    cv2.polylines(img, np.int32([pts_left]), isClosed=False, color=(0,0,255), thickness=10)
    cv2.polylines(img, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=10)
    return img

#Function that concatenates various images to one image
def concatenate(img1, img2, img3, img4, img5):
    offset = 50
    img3 = setOffset(img3, offset)
    img4 = setOffset(img4, offset)
    img1 = cv2.resize(img1, (950,550), interpolation = cv2.INTER_AREA)
    img2 = cv2.resize(img2, (330,180), interpolation = cv2.INTER_AREA)
    img3 = cv2.resize(img3, (165,370), interpolation = cv2.INTER_AREA)
    img4 = cv2.resize(img4, (165,370), interpolation = cv2.INTER_AREA)
    result = np.concatenate((img3, img4), axis = 1)
    result = np.concatenate((img2, result))
    result = np.concatenate((img1, result), axis = 1)
    result = np.concatenate((result, img5), axis = 0)
    return result

#Function that outputs the radius of curvature
def radiusOfCurvature(img, left_fit, right_fit):
    y_eval = img.shape[0]/2
    left_radius = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / (2*left_fit[0])
    right_radius = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / (2*right_fit[0])
    avg_radius = (left_radius+right_radius)/2 
    return round(left_radius,2), round(right_radius,2), round(avg_radius,2)

#Function that outputs image containing radius value
def informationWindow(left_radius, right_radius, radius):
    window = np.zeros((170, 1280, 3), dtype='uint8')
    window[:,:,0] = 249
    window[:,:,1] = 242
    window[:,:,2] = 227
    text1 = '(1) : Detected white and yellow markings, (2) : Warped image, (3) : Curve fitting'
    window = cv2.putText(window, text1, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv2.LINE_AA)
    text2 = 'Left Curvature : ' + str(left_radius) + ', Right Curvature : ' + str(right_radius)
    window = cv2.putText(window, text2, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv2.LINE_AA)
    text3 = 'Average Curvature : ' + str(radius)
    window = cv2.putText(window, text3, (50,150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv2.LINE_AA)
    return window

#Function that predicts turn
def addTurnInfo(img, radius):
    if radius >= 10000:
        img = cv2.putText(img, 'Go Straight', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
    if radius >= 0 and radius < 10000:
        img = cv2.putText(img, 'Turn Right', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
    if radius <= 0:
        img = cv2.putText(img, 'Turn Left', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
    img = cv2.putText(img, '(1)', (1000,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv2.LINE_AA)
    img = cv2.putText(img, '(2)', (1000,230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv2.LINE_AA)
    img = cv2.putText(img, '(3)', (1165,230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv2.LINE_AA)
    
    return img

#Function that adds extra blank space in front of the image
def setOffset(img, offset):
    blank = np.zeros((img.shape[0], offset, 3), dtype = 'uint8')
    img = np.concatenate((blank, img), axis = 1)
    return img



def ai_lane_detection(frame):
    img = transform(frame).unsqueeze(0)

    with torch.no_grad():
        output = model(img)['out'][0]

    mask = output.argmax(0).byte().cpu().numpy()
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    # Create colorful overlay
    color_mask = np.zeros_like(frame)

    # Highlight important classes (road-like areas)
    # Different colors for different classes (more "AI-like")
    color_mask[mask == 0] = [0, 0, 0]        # background
    color_mask[mask == 1] = [255, 0, 0]      # class 1 → blue
    color_mask[mask == 2] = [0, 255, 0]      # class 2 → green
    color_mask[mask == 3] = [0, 0, 255]      # class 3 → red
    color_mask[mask > 3] = [255, 255, 0]     # others → yellow

    # Blend original + mask
    result = cv2.addWeighted(frame, 0.5, color_mask, 0.5, 0)

    return result




video = cv2.VideoCapture(0)

if not video.isOpened():
    print("ERROR: Video not opening")
    exit()

out = cv2.VideoWriter(r"E:\ADITI\ENGINEERING\RVU\SEM8\MAJOR PROJECT\PROJECT 3\lane-detection\results\live_output.avi",cv2.VideoWriter_fourcc(*'XVID'), 25, (1280,720))
print("Generating video output...\n")


mode = "cv"   # change to "ai" later


while True:
    isTrue, frame = video.read()

    if not isTrue:
        break

    if mode == "ai":
        frame = cv2.resize(frame, (960, 540))
        result = ai_lane_detection(frame)
        result = cv2.resize(result, (1280, 720))

        cv2.putText(result, "AI MODE", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

        cv2.putText(result, f"MODE: {mode.upper()}", (900,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        out.write(result)
        cv2.imshow("AI Lane Detection", result)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('m'):
            mode = "ai" if mode == "cv" else "cv"
            print(f"Switched to {mode.upper()} mode")

        continue

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)[1]
    processed_img = preprocessing(frame)
    height, width = processed_img.shape
    polygon = [(int(width*0.15), int(height*0.94)), (int(width*0.45), int(height*0.62)), (int(width*0.58), int(height*0.62)), (int(0.95*width), int(0.94*height))]
    masked_img = regionOfInterest(processed_img, polygon)
    source_points = np.float32([[int(width*0.49), int(height*0.62)], [int(width*0.58), int(height*0.62)], [int(width*0.15), int(height*0.94)], [int(0.95*width), int(0.94*height)]])
    destination_points = np.float32([[0,0], [400,0], [0, 960], [400, 960]])
    warped_img_size = (400, 960)
    warped_img_shape = (960, 400)
    warped_img = warp(masked_img, source_points, destination_points, warped_img_size)
    kernel = np.ones((11,11), np.uint8)
    opening = cv2.morphologyEx(warped_img, cv2.MORPH_CLOSE, kernel)
    left_fit, right_fit = fitCurve(opening)

    if left_fit is None or right_fit is None:
        cv2.imshow("Lane Detection", frame)
        continue


    pts_left, pts_right = findPoints(warped_img_shape, left_fit, right_fit)



    fill_curves = fillCurves(warped_img_shape, pts_left, pts_right)
    unwarped_fill_curves = unwarp(fill_curves, source_points, destination_points, (width, height))
    window1 = cv2.addWeighted(frame, 1, unwarped_fill_curves, 1, 0)
    left_radius, right_radius, avg_radius = radiusOfCurvature(warped_img, left_fit, right_fit)
    window2 = oneToThreeChannel(thresh)
    window3 = oneToThreeChannel(warped_img)
    window4 = drawCurves(warped_img, pts_left, pts_right)
    window5 = informationWindow(left_radius, right_radius, avg_radius)
    result = concatenate(window1, window2, window3, window4, window5)


    # Calculate lane center
    lane_center = (pts_left[0][-1][0] + pts_right[0][0][0]) / 2

    # Car center
    car_center = frame.shape[1] / 2

    # Offset
    offset = car_center - lane_center

    # Warning
    if abs(offset) > 50:
        cv2.putText(result, "Lane Departure!", (50,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        

    result = addTurnInfo(result, avg_radius)

    cv2.putText(result, f"MODE: {mode.upper()}", (900,50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)



    out.write(result)

    cv2.imshow("Lane Detection", result)

    key = cv2.waitKey(40) & 0xFF

    if key == ord('q'):
        print("Exiting live...")
        break
    elif key == ord('m'):
        mode = "ai" if mode == "cv" else "cv"
        print(f"Switched to {mode.upper()} mode")

print("Video output generated.\n")

out.release()
video.release()
cv2.destroyAllWindows()

