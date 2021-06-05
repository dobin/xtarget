# import the necessary packages
import numpy as np
import argparse
import imutils
import sys
import cv2


def getWarp(image, source):
    # load the ArUCo dictionary, grab the ArUCo parameters, and detect
    # the markers
    print("[INFO] detecting markers...")
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    # if we have not found four markers in the input image then we cannot
    # apply our augmented reality technique
    if len(corners) != 4:
        print("[INFO] could not find 4 corners...exiting")
        sys.exit(0)

    # otherwise, we've found the four ArUco markers, so we can continue
    # by flattening the ArUco IDs list and initializing our list of
    # reference points
    print("[INFO] constructing augmented reality visualization...")
    ids = ids.flatten()
    refPts = []
    # loop over the IDs of the ArUco markers in top-left, top-right,
    # bottom-right, and bottom-left order
    for i in (923, 1001, 241, 1007):
        # grab the index of the corner with the current ID and append the
        # corner (x, y)-coordinates to our list of reference points
        j = np.squeeze(np.where(ids == i))
        corner = np.squeeze(corners[j])
        refPts.append(corner)

    # unpack our ArUco reference points and use the reference points to
    # define the *destination* transform matrix, making sure the points
    # are specified in top-left, top-right, bottom-right, and bottom-left
    # order
    (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
    dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
    dstMat = np.array(dstMat)
    # grab the spatial dimensions of the source image and define the
    # transform matrix for the *source* image in top-left, top-right,
    # bottom-right, and bottom-left order
    (srcH, srcW) = source.shape[:2]
    srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])
    # compute the homography matrix
    (H, _) = cv2.findHomography(srcMat, dstMat)

    return H, srcMat, dstMat
    

def translate(x, y, M):
    p = np.array((x,y,1)).reshape((3,1))
    temp_p = M.dot(p)
    sum = np.sum(temp_p ,1)
    px = int(round(sum[0]/sum[2]))
    py = int(round(sum[1]/sum[2]))

    return px, py

def toworld(x,y, inversehomographymatrix):
    imagepoint = [x, y, 1]
    worldpoint = np.array(np.dot(inversehomographymatrix,imagepoint))
    scalar = worldpoint[2]
    xworld = worldpoint[0]/scalar
    yworld = worldpoint[1]/scalar
    return xworld, yworld


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image containing ArUCo tag")
    ap.add_argument("-s", "--source", required=True,
        help="path to input source image that will be put on input")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    image = imutils.resize(image, width=600)
    (imgH, imgW) = image.shape[:2]
    # load the source image from disk
    source = cv2.imread(args["source"])

    H, srcMat, dstMat = getWarp(image, source)
    H_inv = np.linalg.inv(H)
    warped = cv2.warpPerspective(source, H, (imgW, imgH))
    output, outputMask = blend(warped, image, dstMat)

    #cv2.drawContours(output, [dstMat], -1, (0, 255, 0), 2)

    # 90 / 113
    # 250 / 317

    #dst = toworld(90, 113, H)    # gives 290/358, not quiete correct
    #x = 120
    #y = 36
    x = 290
    y = 358
    px, py = translate(x, y, H_inv)  # gives 290/358
    cv2.circle(source, (px, py), 7, (255, 0, 0), 3)

    cv2.circle(output, (x, y), 7, (255, 0, 0), 3)
    cv2.imshow("x", source)
    #print("DST: " + str(dst))
    print("X:  {}   Y: {}".format(x, y))
    print("PX: {}  PY: {}".format(px, py))
    
    print("H: " + str(H))
    print("srcMat: " + str(srcMat))
    print("dstMat: " + str(dstMat))

    #cv2.imshow("Input", image)
    #cv2.imshow("Source", source)
    cv2.imshow("dstMat", outputMask)
    cv2.imshow("warped", warped)
    cv2.imshow("OpenCV AR Output", output)
    cv2.waitKey(0)


def blend(warped, image, dstMat):
    (imgH, imgW) = image.shape[:2]

    # construct a mask for the source image now that the perspective warp
    # has taken place (we'll need this mask to copy the source image into
    # the destination)
    mask = np.zeros((imgH, imgW), dtype="uint8")
    cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255), cv2.LINE_AA)

    # this step is optional, but to give the source image a black border
    # surrounding it when applied to the source image, you can apply a
    # dilation operation
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, rect, iterations=2)

    # create a three channel version of the mask by stacking it depth-wise,
    # such that we can copy the warped source image into the input image
    maskScaled = mask.copy() / 255.0
    maskScaled = np.dstack([maskScaled] * 3)
    # copy the warped source image into the input image by (1) multiplying
    # the warped image and masked together, (2) multiplying the original
    # input image with the mask (giving more weight to the input where
    # there *ARE NOT* masked pixels), and (3) adding the resulting
    # multiplications together
    warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
    imageMultiplied = cv2.multiply(image.astype(float), 1.0 - maskScaled)
    output = cv2.add(warpedMultiplied, imageMultiplied)
    output = output.astype("uint8")

    return output, mask


main()