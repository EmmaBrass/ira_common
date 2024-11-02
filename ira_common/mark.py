import cv2
import numpy as np
import logging
import skimage.io
import skimage.morphology
from fil_finder import FilFinder2D
import astropy.units as u

class Mark:

    def __init__(self, debug: bool):
        self.mask = None
        self.color_rgb = None
        self.color_name = None
        self.skeleton = None
        self.contour = None
        self.length = None
        self.real_area = None
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.rect_area = None
        self.min_rect_area = None
        self.convex_hull_area = None
        self.skel_end_points = None
        self.skel_midpoint_x = None
        self.skel_midpoint_y = None
        self.skel_dx = None
        self.skel_dy = None
        self.gradient = None
        self.type = None
        self.debug = debug

    def analyse(self, new_image):
        self.find_color_rgb(new_image)
        self.find_color_name()
        self.find_skeleton()
        self.find_mark_length()
        self.find_skel_points()
        self.find_rect_area()
        self.test_type()

    def find_color_rgb(self, new_image):
        """
        Function to find, set, and return the color of the mark in RGB.
        """

        if (self.mask).all() == None:
            raise ValueError("The mark mask has not been calculated yet.")

        # erode the mask a little to ensure we are getting the right color fo the paint stroke and no background is included
        mask_eroded = cv2.erode(self.mask, None, iterations=4)
        # give mean color of the region in the contour
        mark_color = cv2.mean(new_image, mask=mask_eroded)[:3]
        print("Average paint stroke color is: %s in BGR", mark_color)
        
        self.color_rgb = [mark_color[2], mark_color[1], mark_color[0]]
        print("Average paint stroke color is: %s in RGB", self.color_rgb)

        return self.color_rgb


    # Define function to classify colors
    def find_color_name(self):
        """
        Function to find, set, and return the color name as a string.
        """

        if self.color_rgb == None:
            raise ValueError("The mark's RGB color has not been calculated yet.")

        # Define color chart colors and their names
        chart_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]])
        color_names = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']

        # Calculate distances between RGB color and chart colors
        distances = np.sqrt(np.sum((chart_colors - self.color_rgb) ** 2, axis=1))
        # Find index of closest color
        index = np.argmin(distances)
        
        # Return name of closest color
        self.color_name = color_names[index]

        return self.color_name

    
    def find_skeleton(self):
        """
        Find the skeleton of the image, set it, and return it.
        Requires the mark mask to have been calculated.
        """

        if (self.mask).all() == None:
            raise ValueError("The mark mask has not been calculated yet.")
        
        # Skeletonize the image
        skel = skimage.morphology.skeletonize(self.mask)

        # Find the longest path in the image
        fil = FilFinder2D(skel, distance=250 * u.pc, mask=skel)
        fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
        fil.medskel(verbose=False)
        fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length')

        skeleton_longpath = (fil.skeleton_longpath * 255).astype("uint8")
        if self.debug == True:
            cv2.imshow('skeleton_longpath', skeleton_longpath)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        self.skeleton = skeleton_longpath

        return self.skeleton

    def find_mark_length(self):
        """
        Find the length of the mark.
        Must be carried out after skeletonise.
        """

        if (self.skeleton).all() == None:
            raise ValueError("The mark's skeleton has not been calculated yet.")

        # Calculate the perimeter (length) of the skeleton
        contours, _ = cv2.findContours(self.skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.contour = contours[0]

        skeleton_length = (cv2.arcLength(contours[0], closed=False))/2
        print("Length of skeleton: %s", skeleton_length)

        self.length = skeleton_length 

        return self.length

    def find_skel_points(self):
        """
        A function to find the start and end points of the skeleton line.
        """

        if (self.skeleton).all() == None:
            raise ValueError("The mark's skeleton has not been calculated yet.")

        # Find the start and end points of the skeleton line
        line_points = np.argwhere(self.skeleton == 255)

        end_points = []    
        for p in line_points:
            x = p[1]
            y = p[0]
            n = 0        
            n += self.skeleton[y - 1,x]
            n += self.skeleton[y - 1,x - 1]
            n += self.skeleton[y - 1,x + 1]
            n += self.skeleton[y,x - 1]    
            n += self.skeleton[y,x + 1]    
            n += self.skeleton[y + 1,x]    
            n += self.skeleton[y + 1,x - 1]
            n += self.skeleton[y + 1,x + 1]
            n /= 255        
            if n == 1:
                end_points.append(p)

        self.skel_end_points = end_points # as (y,x)
        print("End points are: %s", self.skel_end_points)

        # Find dy, dx, midpoints, and gradient of the skeleton line
        dy = self.skel_end_points[1][0]-self.skel_end_points[0][0]
        dx = self.skel_end_points[1][1]-self.skel_end_points[0][1]
        midpoint_y = self.skel_end_points[0][0] + dy/2
        midpoint_x = self.skel_end_points[0][1] + dx/2

        self.skel_dx = dx
        self.skel_dy = dy

        self.skel_midpoint_y = midpoint_y
        self.skel_midpoint_x = midpoint_x

        if dx == 0:
            self.gradient = 0
        elif dy == 0:
            self.gradient = float('inf')
        else:
            self.gradient = dy/dx

        return self.skel_end_points

    def find_rect_area(self):
        """
        This function allows us to find the minimum and maximum x and y values for the mark.
        Note this is for the whole mark, not just the skeleton.
        It then finds the rectangle area for the whole mark.
        """

        if (self.mask).all() == None:
            raise ValueError("The mark mask has not been calculated yet.")

        y_coords, x_coords = np.where(self.mask == 255)

        # Get the maximum and minimum x and y values
        max_x = np.max(x_coords)
        min_x = np.min(x_coords)
        max_y = np.max(y_coords)
        min_y = np.min(y_coords)

        print("Max x value for rect area: %s", max_x)
        print("Min x value for rect area: %s", min_x)
        print("Max y value for rect area: %s", max_y)
        print("Min y value for rect area: %s", min_y)

        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

        self.rect_area = (max_x-min_x)*(max_y-min_y)

        print("Rectangular area: %s", self.rect_area)

        return self.rect_area

    # types of mark:
    # straight, curve, blob
    # TODO add in classification based on the SIZE of the mark as well

    def test_type(self):
        """Classify the mark as straight, curve, or blob."""

        # Straight if length of the skeleton is approx equal to length of rotated bounding box.  
        # Get the dimensions of the rotated rectangle
        width = self.min_rect_area[1][0]
        height = self.min_rect_area[1][1]
        # Find the length of the longest side
        longest_side_length = max(width, height)
        shortest_side_length = min(width, height)
        # Use ratio of real area / min rect area to compare blob vs curve
        area_min_rect = width*height
        print("real area / min rect area: %s", self.real_area/area_min_rect)
        # Final decision
        if (self.length/longest_side_length > 0.8 and self.length/longest_side_length < 1.2 and
            longest_side_length/shortest_side_length > 2.5 and self.real_area/area_min_rect >= 0.7):
            print("The mark is straight")
            self.type = 'straight'
            return self.type
        elif self.real_area/area_min_rect >= 0.7:
            print("The mark is a blob")
            self.type = 'blob'
            return self.type  
        else:
            print("The mark is a curve")
            self.type = 'curve'
            return self.type



        # inverse_mask = cv.bitwise_not(self.mask)
        # params = cv.SimpleBlobDetector_Params()
        # params.filterByArea = False
        # params.filterByConvexity = False
        # params.filterByInertia = True
        # params.maxInertiaRatio = 0.15 # tweak this to change the straightness threshold.
        # params.minInertiaRatio = 0
        # # Low intertia = more of a straight line
        # detector = cv.SimpleBlobDetector_create(params)
        # keypoints = detector.detect(inverse_mask)
        # points = []
        # for keypoint in keypoints:
        #     points.append(keypoint.pt)
        # self.logger.debug("Type detector - key points: %s", points)
        # # if points is not empty, the mark is a curve.
        # if len(points) != 0:
        #     print("The mark is a curve")
        #     self.type = 'curve'
        #     return self.type
        # else:
              


        # # Find contours of the skeleton
        # contours, _ = cv.findContours(self.skeleton, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        # # Find convex hull of the skeleton
        # hull = cv.convexHull(contours[0])
        # # Calculate area enclosed by convex hull
        # area = cv.contourArea(hull)
        # self.convex_hull_area = area
        # print("The area enclosed by the convex hull is: %s", area)
        # # if ratio of convex hull area to length is below a threshold, then a straight line
        # if (self.convex_hull_area/self.length < 35):
        #     print("Straight line! Convex hull / length ratio: %s", self.convex_hull_area/self.length)
        #     self.type = 'straight'
        # else:
        #     print("Not a straight line! Convex hull / length ratio: %s", self.convex_hull_area/self.length)
        #     self.type = 'curve'

        # # Test if a blob
        # # use aspect ratio (width/height) vs real area.
        # # if square area is much larger than real area, likely a line
        # # but this only covers diagonals...
        # # square area is similar to real area AND aspect ratio is far from one, then a line still
        # # IF square area is similar to real area AND aspect ratio is quite close to one (quite square) then a BLOB!

        # aspect_ratio = (self.max_x - self.min_x) / (self.max_y - self.min_y)

        # if (self.rect_area/self.real_area > 1.8):
        #     print("rect area / real area: %s", self.rect_area/self.real_area )
        # elif (aspect_ratio > 1.3 or aspect_ratio < 0.7):
        #     print("aspect ratio: %s", aspect_ratio)
        # else: 
        #     print("rect_area/real_area is < 1.8 and aspect ratio is between 0.7 and 1.3, hence a blob!")
        #     self.type = 'blob'
        
        # return self.type