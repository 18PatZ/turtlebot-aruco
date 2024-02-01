# The following code is used to watch a video stream, detect a Charuco board, and use
# it to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import numpy as np
import cv2
import cv2.aruco as aruco
import os
import pickle


def drawArucoMarkers(img, corners, ids):
    if len(corners) > 0:
        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            cv2.line(img, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(img, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(img, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(img, bottomLeft, topLeft, (0, 255, 0), 2)

            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(img, (cX, cY), 4, (0, 0, 255), -1)

            # draw the ArUco marker ID on the image
            cv2.putText(img, str(markerID),
                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
    return img


def drawProjected(img, objectPoints, rvec, tvec, cameraMatrix, distCoeffs, color, thickness):
    projected, _ = cv2.projectPoints(objectPoints = np.array(objectPoints), rvec=rvec, tvec=tvec, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)

    points = np2list(projected)

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]
        cv2.line(img, np2cvi(p1), np2cvi(p2), color, thickness)

    return img


def flatten(arr):
    return np.array(arr).flatten()

def np2cvi(p):
    return (int(p[0]), int(p[1]))

def np2list(projected_to_cam):
    return [p[0] for p in projected_to_cam]



class ArucoDetector:

    def __init__(self, cameraMatrix, distCoeffs):
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

        # Check for camera calibration data
        if cameraMatrix is None or distCoeffs is None:
            print("Calibration issue. Camera and distortion matrix must be provided!")
            exit()

        # Constant parameters used in Aruco methods
        # ARUCO_PARAMETERS = aruco.DetectorParameters()
        # ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
        self.ARUCO_PARAMETERS = aruco.DetectorParameters_create()
        self.ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)

        # self.detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)

        self.markerLength = markerLength = 12.1#54#50.8#(18.83 * 16/20) / 1000#0.05;
        self.objPoints = np.array([
            [-markerLength/2, markerLength/2, 0],
            [markerLength/2, markerLength/2, 0],
            [markerLength/2, -markerLength/2, 0],
            [-markerLength/2, -markerLength/2, 0]
        ])


    def process(self, QueryImg, target_id):

        target = None
        
        # grayscale image
        gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
    
        # Detect Aruco markers
        # corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.ARUCO_DICT, parameters=self.ARUCO_PARAMETERS)
    
        # # Refine detected markers
        # # Eliminates markers not part of our board, adds missing markers to the board
        # corners, ids, rejectedImgPoints, recoveredIds = self.detector.refineDetectedMarkers(
        #         image = gray,
        #         board = self.CHARUCO_BOARD,
        #         detectedCorners = corners,
        #         detectedIds = ids,
        #         rejectedCorners = rejectedImgPoints,
        #         cameraMatrix = self.cameraMatrix,
        #         distCoeffs = self.distCoeffs)  

        if corners is not None and len(corners) > 0:
           
            # Outline all of the markers detected in our image
            QueryImg = drawArucoMarkers(QueryImg, corners, ids)
            QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))

            #single_rvecs, single_tvecs = aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)  
            
            for (corner, id) in zip(corners, ids):
                retval, rvec, tvec = cv2.solvePnP(
                    objectPoints=self.objPoints, 
                    imagePoints=corner, 
                    cameraMatrix=self.cameraMatrix, 
                    distCoeffs=self.distCoeffs)           
                QueryImg = cv2.drawFrameAxes(QueryImg, self.cameraMatrix, self.distCoeffs, rvec, tvec, self.markerLength/2)#0.015)

                data = (corner, rvec, tvec)
                if id == target_id:
                    target = data
                    print("RVec:", rvec)
                    print("TVec:", tvec)

        marker_rotation = [float('NaN'), float('NaN'), float('NaN')]
        marker_translation = [float('NaN'), float('NaN'), float('NaN')]

        if target is not None:
            marker_rotation = flatten(target[1])
            marker_translation = flatten(target[2])

            stepX = stepY = 10.8

            gridSize = 10

            sizeX = sizeY = gridSize * stepX

            for i in range(gridSize+1):
                x = i * stepX - sizeX/2
                # draw y line
                drawProjected(QueryImg, [
                    [x, -sizeY/2, 0],
                    [x, sizeY/2, 0],
                ], rvec, tvec, self.cameraMatrix, self.distCoeffs, (0, 255, 255), 1)

                y = i * stepY - sizeY/2
                # draw x line
                drawProjected(QueryImg, [
                    [-sizeX/2, y, 0],
                    [sizeX/2, y, 0],
                ], rvec, tvec, self.cameraMatrix, self.distCoeffs, (0, 255, 255), 1)

        has_marker = target is not None
        return has_marker, marker_rotation, marker_translation, QueryImg
