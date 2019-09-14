import numpy as np
from matplotlib import pyplot as plt
import time
import cv2
from Framefilter import Framefilter
from CameraCalibration import CameraCalibration

# class Vision():
# 	def __init__(self,
# 				 video_source=0):

# 		self.video_source = video_source
# 		self.record= cv2.VideoCapture(video_source)	
#         self.record.set(cv2.CAP_PROP_FRAME_WIDTH, 320.0)
#         self.record.set(cv2.CAP_PROP_FRAME_HEIGHT, 240.0)

#         self.main_thread = threading.Thread(target=self.main_loop)
# 		self.main_thread.start()

# 	def main_loop(self):
#         #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++
#         while(1):
#             ret, frame_input = self.record.read()
#             frame_input = cv2.resize(frame_input, (320, 240), interpolation = cv2.INTER_AREA)

#             fps = self.record.get(cv2.CAP_PROP_FPS)

#             frame_gray = Framefilter.color_drop(frame_input)
#             frame_edge = Framefilter.horizontal_edges_extraction(frame_gray)
#             histogram = Framefilter.signature_histogram_generation(frame_edge)
#             frame_histogram = Framefilter.plot_line(histogram, frame_gray.shape) 
#             frame_concat = Framefilter.concat_frame(frame_edge, frame_histogram, axis=1)
 
#             font = cv2.FONT_HERSHEY_SIMPLEX 
#             cv2.putText(frame_concat,
#                         str(fps),
#                         (10, 50), 
#                         font, 
#                         0.5,
#                         (255, 255, 255),
#                         2,
#                         cv2.LINE_AA)

#             Framefilter.display_frame(frame_concat)
#             #Framefilter.print_frame_mean(frame_concat)

#             k = cv2.waitKey(30) & 0xff
#             if k == 27:
#                 break

#         #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++ 

class StereoVision():
    def __init__(self):
        pass

    # ===========================================================
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def camera_status(frame_input):
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(frame_L_input, frame_R_input)
        return disparity

    # ===========================================================
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def stereo_bm(frame_L_input, frame_R_input):
        stereo = cv2.StereoBM_create(numDisparities=32, blockSize=25)
        disparity = stereo.compute(frame_L_input, frame_R_input)
        return disparity

    # ===========================================================
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def plot_depth_map(frame_L_input, frame_R_input):
        #===============================================================================
        frame_disparity = StereoVision.stereo_bm(frame_L_input, frame_R_input)

        fig = plt.figure(figsize=(11,22))
        #===============================================================================  
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(frame_L_input, 'gray')
        ax1.grid(True)
        # plt.xticks([])
        # plt.yticks([])
        ax1.set_aspect('auto')
        #===============================================================================  
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(frame_R_input, 'gray')
        ax2.grid(True)
        # plt.xticks([])
        # plt.yticks([])
        ax2.set_aspect('auto')
        #===============================================================================  
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(frame_disparity, 'inferno')
        ax3.grid(True)
        # plt.xticks([])
        # plt.yticks([])
        ax3.set_aspect('auto')
        #===============================================================================
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(frame_disparity, 'gray')
        ax4.grid(True)
        # plt.xticks([])
        # plt.yticks([])
        ax4.set_aspect('auto')
        #===============================================================================         
        fig.subplots_adjust(wspace=0, hspace=0)

        plt.show()

        return frame_disparity

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def testbench(video_source=0):
        record= cv2.VideoCapture(video_source)

        # record.set(cv2.CAP_PROP_FPS, 10)
        # record.set(cv2.CAP_PROP_FRAME_WIDTH, 320.0)
        # record.set(cv2.CAP_PROP_FRAME_HEIGHT, 240.0)

        print("start")
        #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++
        while 1:
            #===============================================================================
            for i in range(30):
                print(i)
                ret_L, frame_L_input = record.read()

            frame_L_gray = Framefilter.color_drop(frame_L_input)
            frame_L_blur = Framefilter.blur(frame_L_gray)

            plt.imshow(frame_L_blur)
            plt.show()

            #===============================================================================
            for i in range(30):
                print(i)
                ret_R, frame_R_input = record.read()

            frame_R_gray = Framefilter.color_drop(frame_R_input)
            frame_R_blur = Framefilter.blur(frame_R_gray)

            frame_concat = Framefilter.concat_frame(frame_L_input, frame_R_input, axis=1)
            plt.imshow(frame_concat, 'gray')
            plt.show()

            #===============================================================================
            frame_disparity = StereoVision.plot_depth_map(frame_L_gray, frame_R_gray)

            # ----------------------------------------------------------------------------------------------------------
            # Esc -> EXIT while
            # print("standby")
            # while 1:
            #   k = cv2.waitKey(1) & 0xff
            #   if k ==13 or k==27:
            #     break

            # if k == 27:
            #     break
            break
            # ----------------------------------------------------------------------------------------------------------

        #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
        record.release()
        cv2.destroyAllWindows()

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def image_depth_map(frame_L_file = "hero_3_L.png", frame_R_file = "hero_3_R.png"):
        frame_L_input = cv2.imread(frame_L_file)
        frame_R_input = cv2.imread(frame_R_file)        

        frame_L_gray = Framefilter.color_drop(frame_L_input)
        frame_L_blur = Framefilter.blur(frame_L_gray)

        frame_R_gray = Framefilter.color_drop(frame_R_input)
        frame_R_blur = Framefilter.blur(frame_R_gray)

        #=======================================================
        frame_disparity = StereoVision.stereo_bm(frame_L_input, frame_R_input)

        # frame_disparity = StereoVision.plot_depth_map(frame_L_blur, frame_R_blur)

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def translation_frame(frame_input, n_x=5, n_y=0):
        M = np.float32([[1, 0, n_x], [0, 1, n_y]])
        frame_output = cv2.warpAffine(frame_input, M, (frame_input.shape[1], frame_input.shape[0]))
        return frame_output

    @staticmethod
    def mm_filter(frame_input, frame_buffer, frame_pt):
        frame_output = np.zeros(frame_input.shape)

        if frame_pt > len(frame_buffer) - 1:
            frame_pt = 0

        frame_buffer[frame_pt] = frame_input
        for frame in frame_buffer:
            frame_output = frame_output + frame

        # frame_output = frame_output/len(frame_buffer)

        frame_pt = frame_pt + 1

        return frame_output, frame_buffer, frame_pt

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    #  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def run_depth_map(video_source_L = 4, video_source_R = 2):

        source_L = cv2.VideoCapture(video_source_L)
        source_R = cv2.VideoCapture(video_source_R)

        CAMERA_WIDTH = 640
        CAMERA_HEIGHT = 480

        source_L.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        source_L.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        source_L.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        source_R.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        source_R.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        source_R.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        #=======================================================
        stereo = cv2.StereoBM_create(numDisparities=32, blockSize=25)

        #=======================================================
        buffer_pt = 50
        frame_buffer = []
        for i in range(buffer_pt):
            frame_buffer.append(np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH)))

        #=======================================================
        while 1:
            if not (source_L.grab() and source_R.grab()):
                print("No more frames")
                break
            ret, frame_L_input = source_L.retrieve()
            ret, frame_R_input = source_R.retrieve()
            
            #ret, frame_L_input = source_L.read()
            #ret, frame_R_input = source_R.read()
     

            frame_L_gray = Framefilter.color_drop(frame_L_input)
            frame_L_blur = Framefilter.blur(frame_L_gray)

            frame_R_gray = Framefilter.color_drop(frame_R_input)
            frame_R_blur = Framefilter.blur(frame_R_gray)


            frame_L_line = frame_L_blur
            cv2.line(frame_L_line,
                     (0, frame_L_line.shape[0]//2),
                     (frame_L_line.shape[1], frame_L_line.shape[0]//2),
                     (255,0,0),
                     1)

            cv2.line(frame_L_line,
                     (frame_L_line.shape[1]//2, 0),
                     (frame_L_line.shape[1]//2, frame_L_line.shape[0]),
                     (255,0,0),
                     1)

            frame_R_line = frame_R_blur
            cv2.line(frame_R_line,
                     (0, frame_R_line.shape[0]//2),
                     (frame_R_line.shape[1], frame_R_line.shape[0]//2),
                     (255,0,0),
                     1)

            cv2.line(frame_R_line,
                     (frame_R_line.shape[1]//2, 0),
                     (frame_R_line.shape[1]//2, frame_R_line.shape[0]),
                     (255,0,0),
                     1)

            frame_concat = Framefilter.concat_frame(frame_L_line, frame_R_line, axis=1)

            frame_disparity = stereo.compute(frame_L_blur, frame_R_blur)
            
            frame_mm_disparity, frame_buffer, frame_pt = StereoVision.mm_filter(frame_disparity, frame_buffer, buffer_pt)

            cv2.imshow("Video", frame_concat)
            cv2.imshow("disparity", frame_mm_disparity/256)
            
            # --------------------------------------------------
            # Esc -> EXIT while
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                print(frame_disparity.shape)
                break
            # --------------------------------------------------



# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#Framefilter.testbench(video_source="./dataset/drone_1.mp4")
#Framefilter.testbench(video_source="./dataset/driver_3.mp4")
# Framefilter.testbench(video_source=2)
StereoVision.run_depth_map()
#StereoVision.image_depth_map(frame_L_file = "ambush_5_left.jpg", frame_R_file = "ambush_5_right.jpg")
#StereoVision.image_depth_map(frame_L_file = "hero_3_L.png", frame_R_file = "hero_3_R.png")