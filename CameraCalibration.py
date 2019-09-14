
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt

class CameraCalibration():
    def __init__(self):
        pass

    # ===========================================================
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def find_chess(frame_input, chess_size=(6, 6)):
        status = None
        print("chess...")

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chess_size[0]*chess_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chess_size[0], 0:chess_size[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        frame_gray = cv2.cvtColor(frame_input, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(frame_gray, (chess_size[0], chess_size[1]), None)

        # If found, add object points, image points (after refining them)
        frame_output = None
        if ret == True:
            status = "checkmate!"
            print(status)
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(frame_gray, 
                                        corners, 
                                        (11, 11), 
                                        (-1, -1), 
                                        criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            frame_output = cv2.drawChessboardCorners(frame_input, (chess_size[0], chess_size[1]), corners2, ret)
            plt.imshow(frame_output)
            plt.show()

        if frame_output is None:
            frame_output = frame_input

        return frame_output, objpoints, imgpoints, status

    # ===========================================================
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def calibrateCoefficients(frame_input, objpoints, imgpoints):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, 
                                                           imgpoints, 
                                                           frame_input.shape[::-1], 
                                                           None, 
                                                           None)
        tot_error = 0
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            tot_error += error

        print("total error: ", mean_error/len(objpoints))

        return ret, mtx, dist, rvecs, tvecs

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def testbench(video_source=2):
        capture = cv2.VideoCapture(video_source)

        count_frame = 0
        while 1:
            #  ++++++++++++++++++++++++++++++++++++++++++++++++
            print('calibrate state...')
            status = None
            while status is None: 
                status = None
                ret, frame_input = capture.read()
                print(count_frame)
                count_frame += 1

                frame_chess, objpoints, imgpoints, status = CameraCalibration.find_chess(frame_input)
                plt.imshow(frame_chess)
                plt.show()

            #  ++++++++++++++++++++++++++++++++++++++++++++++++
            frame_gray = cv2.cvtColor(frame_input, cv2.COLOR_BGR2GRAY)
            plt.imshow(frame_gray)
            plt.show()

            ret, mtx, dist, rvecs, tvecs = CameraCalibration.calibrateCoefficients(frame_gray, objpoints, imgpoints)

            h,  w = frame_gray.shape[:2]
            newcameramtx, roi =cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

            #  ++++++++++++++++++++++++++++++++++++++++++++++++
            print('test state...')
            while 1:
                ret, frame_input = capture.read()

                frame_gray = cv2.cvtColor(frame_input,cv2.COLOR_BGR2GRAY)
                h,  w = frame_gray.shape[:2]
                newcameramtx, roi =cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

                frame_undist = cv2.undistort(frame_input, mtx, dist, None, newcameramtx)
                x,y,w,h = roi
                print(x,y,w,h)
                # frame_undist = frame_undist[y:y+h, x:x+w]

                frame_concat = np.concatenate((frame_undist, frame_input), axis=1)

                plt.imshow(frame_concat)
                plt.show()
                
                # ----------------------------------------------------------
                # Esc -> EXIT while
                # while 1:
                #   k = cv2.waitKey(1) & 0xff
                #   if k ==13 or k==27:
                #     break

                # if k == 27:
                #     break
                # ----------------------------------------------------------

        capture.release()
        cv2.destroyAllWindows()

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def getPhoto(video_source=0):
        capture = cv2.VideoCapture(video_source)

        while 1: 
            ret, frame_input = capture.read()

            frame_line = frame_input
            frame_output = cv2.line(frame_line,
                                    (0, frame_line.shape[0]//2),
                                    (frame_line.shape[1], frame_line.shape[0]//2),
                                    (255,0,0),
                                    1)

            frame_output = cv2.line(frame_line,
                                    (frame_line.shape[1]//2, 0),
                                    (frame_line.shape[1]//2, frame_line.shape[0]),
                                    (255,0,0),
                                    1)

            cv2.imshow("Video", frame_line)

            # ------------------------------------------------------------------------------------------------------------------
            # Esc -> EXIT while
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            # ------------------------------------------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------------------------
        ret, frame_input = capture.read()
        frame_input = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
        
        plt.imshow(frame_input)
        plt.xticks([])
        plt.yticks([])
        plt.show()

        # ----------------------------------------------------------------------------------------------------------------------
        capture.release()
        cv2.destroyAllWindows()




# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# CameraCalibration.testbench(video_source=2)