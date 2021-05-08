import cv2 as cv

def rescaleFrame(frame, scale=1.0):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    # cropped = img[50:200, 200:400]

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def toGrey(frame):
    # Converting to grayscale
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return frame


# May be useful? White edges on black background
# More blur -> less edges
#
# Detects moving reddot well with contour, but is blind to steady dot
def edgeCascade(frame):
    # Blur
    # frame = cv.GaussianBlur(frame, (7,7), cv.BORDER_DEFAULT)

    # Edge Cascade
    frame = cv.Canny(frame, 125, 175)
    return frame


# eroding somehow makes lines thicker
# 
def sharpoon(frame):
    # Dilating / blur=
    #frame = cv.dilate(frame, (7,7), iterations=3)

    # Eroding / shapren?
    frame = cv.erode(frame, (7,7), iterations=3)

    return frame


# from https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf