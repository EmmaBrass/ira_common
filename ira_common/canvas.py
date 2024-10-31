import cv2
import numpy as np
import math

class Canvas:

    def __init__(self):
        self.image = None
        self.image_height = None
        self.image_width = None
        self.transformed_image_x = 2048 
        self.transformed_image_y = None # Will be set based on canvas dimensions later on
        self.transform_matrix = None
        self.transformed_image = None
        self.width_mm = None
        self.height_mm = None
        self.mask = None
        self.color_bgr = None
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.top_line= None
        self.bottom_line = None
        self.left_line = None
        self.right_line = None
        self.top_left_corner = None
        self.top_right_corner = None
        self.bottom_left_corner = None
        self.bottom_right_corner = None

        self.logger_level = 5

    def set_image(self, image):
        self.image = image
        self.image_height = image.shape[0]
        print("image height: %s", self.image_height)
        self.image_width = image.shape[1]
        print("image width: %s", self.image_width)

    def set_real_dimensions(self, width, height):
        """
        Input the real dimensions of the canvas in mm.
        
        :param width: Width of canvas in mm.
        :param height: Height of canvas in mm.
        """
        self.width_mm = width
        self.height_mm = height 
        self.transformed_image_y = int(self.transformed_image_x*(height/width))

    def analyse(self):
        try:
            self.find_mask()
            self.find_lines()
            self.find_color_bgr()
            self.find_min_max_x_y()
            self.find_corners()
            self.transform()
            return True  # All functions succeeded
        except Exception as e:
            print(f"Analysis failed: {e}")
            return False  # An error occurred, analysis failed

    def find_mask(self):
        """Method to pull out the canvas as a mask."""
        # convert image to grayscale
        imgray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # create the bitmap -> white object on black background
        # just standard threshold, adaptive didn't work that well...
        _, thresh = cv2.threshold(imgray,150,255,cv2.THRESH_BINARY)

        # find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the index of the contour with the largest internal area
        max_area = -1
        max_area_idx = -1
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_area_idx = i

        mask = np.zeros(self.image.shape[:2], dtype="uint8")
        # draw contours onto the mask -> thickness = -1 gives a filled-in contour.
        cv2.drawContours(mask, contours, max_area_idx, 255, thickness = -1)
        if self.logger_level <= 10:
            # show the image
            cv2.imshow("mask", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # set the bitmap mask image array with the canvas in white and background in black
        self.mask = mask

        return self.mask

    def find_lines(self):

        if (self.mask).all() == None:
            raise ValueError("The mark mask has not been calculated yet.")

        # Canny edge detection
        dst = cv2.Canny(self.mask, 50, 200, None, 3)

        if self.logger_level <= 10:
            # show the image
            cv2.imshow("Canny", dst)
            cv2.waitKey(0)

        # Find lines
        linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 60, None, 100, 100)

        if self.logger_level <= 10:
            cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
            cdstP = np.copy(cdst)
            if linesP is not None:
                for i in range(0, len(linesP)):
                    l = linesP[i][0]
                    cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
            # show the image
            cv2.imshow("Hough lines", cdstP)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Find the longest line for the top, bottom, left, and right edge
        top_line_len = 0
        bottom_line_len = 0
        left_line_len = 0
        right_line_len = 0
        for line in linesP:
            x1 = line[0][0]
            y1 = line[0][1]
            x2 = line[0][2]
            y2 = line[0][3]
            slope = ((y2 - y1) / (x2 - x1)) if (x2-x1) != 0 else float('inf')

            # Top line
            if (y1 < self.image_height/2 and y2 < self.image_height/2 and y2-y1 < 100 and slope < 1 and slope > -1):
                line_len = math.sqrt((x2-x1)**2+(y2-y1)**2)
                if line_len > top_line_len:
                    top_line_len = line_len
                    top_line = line[0]
            # Bottom line
            if (y1 > self.image_height/2 and y2 > self.image_height/2 and y2-y1 < 100 and slope < 1 and slope > -1):
                line_len = math.sqrt((x2-x1)**2+(y2-y1)**2)
                if line_len > bottom_line_len:
                    bottom_line_len = line_len
                    bottom_line = line[0]
            # Left line
            if (x1 < self.image_width/2 and x2 < self.image_width/2 and x2-x1 < 100 and (slope < -1 or slope > 1 or slope == float('inf'))):
                line_len = math.sqrt((x2-x1)**2+(y2-y1)**2)
                if line_len > left_line_len:
                    left_line_len = line_len
                    left_line = line[0]
            # Right line
            if (x1 > self.image_width/2 and x2 > self.image_width/2 and x2-x1 < 100 and (slope < -1 or slope > 1 or slope == float('inf'))):
                line_len = math.sqrt((x2-x1)**2+(y2-y1)**2)
                if line_len > right_line_len:
                    right_line_len = line_len
                    right_line = line[0]

        linesH = [top_line, bottom_line]
        linesV = [left_line, right_line]

        print("linesH: %s", linesH)
        print("linesV: %s", linesV)

        # for the verticals (left and right), extend the lines to the edges of the image
        # save black images with red lines on them
        for num, line in enumerate(linesV):
            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]
            # calculate slope and intercept of the line
            slope = ((y2 - y1) / (x2 - x1)) if (x2-x1) != 0 else float('inf')
            intercept = (y1 - slope * x1) if slope != float('inf') else None

            # extend the line to the edges of the image
            x1_ext = (0 - intercept) / slope if slope != float('inf') else x1
            x2_ext = ((self.image_height - intercept) / slope) if slope != float('inf') else x1
            y1_ext = 0
            y2_ext = self.image_height

            # draw extended lines on black images
            if num == 0:
                self.left_line = np.zeros(self.image.shape[:2], dtype="uint8")
                cv2.line(self.left_line, (int(x1_ext), int(y1_ext)), (int(x2_ext), int(y2_ext)), 255, 1)
                if self.logger_level <= 10:
                    cv2.imshow("Left line", self.left_line)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            if num == 1:
                self.right_line = np.zeros(self.image.shape[:2], dtype="uint8")
                cv2.line(self.right_line, (int(x1_ext), int(y1_ext)), (int(x2_ext), int(y2_ext)), 255, 1)
                if self.logger_level <= 10:
                    cv2.imshow("Right line", self.right_line)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

        # for the horizontals (top and bottom), extend the lines to the edges of the image
        # save black images with red lines on them
        for num, line in enumerate(linesH):
            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]
            # calculate slope and intercept of the line
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

             # calculate intersection of the line with left and right edges of image
            y1_ext = (slope * 0 + intercept) if slope != 0 else y1
            y2_ext= (slope * self.image_width + intercept) if slope != 0 else y1
            x1_ext = 0
            x2_ext = self.image_width

            # draw extended lines on black images
            if num == 0:
                self.top_line = np.zeros(self.image.shape[:2], dtype="uint8")
                cv2.line(self.top_line, (int(x1_ext), int(y1_ext)), (int(x2_ext), int(y2_ext)), 255, 1)
                if self.logger_level <= 10:
                    cv2.imshow("Top line", self.top_line)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            if num == 1:
                self.bottom_line = np.zeros(self.image.shape[:2], dtype="uint8")
                cv2.line(self.bottom_line, (int(x1_ext), int(y1_ext)), (int(x2_ext), int(y2_ext)), 255, 1)
                if self.logger_level <= 10:
                    cv2.imshow("Bottom line", self.bottom_line)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

    def find_color_bgr(self):

        if (self.mask).all() == None:
            raise ValueError("The mark mask has not been calculated yet.")

        # erode the mask a little to ensure we are getting the right color and no background is included
        mask = cv2.erode(self.mask, None, iterations=13)
        if self.logger_level <= 10:
            cv2.imshow("Eroded mask for finding average canvas color", mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # give mean color of the region in the contour
        canvas_color = cv2.mean(self.image, mask=mask)[:3]
        print("Average canvas color is: %s in BGR", canvas_color)

        self.color_bgr = canvas_color
        
        return self.color_bgr

    def find_min_max_x_y(self):

        if (self.mask).all() == None:
            raise ValueError("The mark mask has not been calculated yet.")

        # look at top line for min y
        y_coords, x_coords = np.where(self.top_line == 255)
        min_y = np.min(y_coords)

        # look at bottom line for max y
        y_coords, x_coords = np.where(self.bottom_line == 255)
        max_y = np.max(y_coords)

        # look at left line for min x
        y_coords, x_coords = np.where(self.left_line == 255)
        min_x = np.min(x_coords)

        # look at right line for max x
        y_coords, x_coords = np.where(self.right_line == 255)
        max_x = np.max(x_coords)

        self.logger.debug("Max x value of canvas area: %s", max_x)
        self.logger.debug("Min x value of canvas area: %s", min_x)
        self.logger.debug("Max y value of canvas area: %s", max_y)
        self.logger.debug("Min y value of canvas area: %s", min_y)

        # Save to the class
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def find_corners(self):
        # bitwise_and on pairs of line images, to get corners

        # top left
        top_left = cv2.bitwise_and(self.left_line, self.top_line)
        if self.logger_level <= 10:
            # Show the image to check/debug
            cv2.imshow("top_left corner", top_left)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        top_left_y_points, top_left_x_points = np.where(top_left == 255)
        top_left_corner = (
            top_left_x_points[0]+(self.image_width/300), 
            top_left_y_points[0]+(self.image_width/300)
        )
        self.top_left_corner = list(top_left_corner)
        print("top left point: %s", self.top_left_corner)

        # top right
        top_right = cv2.bitwise_and(self.right_line, self.top_line)
        if self.logger_level <= 10:
            # Show the image to check/debug
            cv2.imshow("top_right corner", top_right)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        top_right_y_points, top_right_x_points = np.where(top_right == 255)
        top_right_corner = (
            top_right_x_points[0]-(self.image_width/300),
            top_right_y_points[0]+(self.image_width/300), 
        )
        self.top_right_corner = list(top_right_corner)
        print("top right point: %s", self.top_right_corner)

        # bottom left
        bottom_left = cv2.bitwise_and(self.left_line, self.bottom_line)
        if self.logger_level <= 10:
            # Show the image to check/debug
            cv2.imshow("bottom_left corner", bottom_left)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        bottom_left_y_points, bottom_left_x_points = np.where(bottom_left == 255)
        bottom_left_corner = (
            bottom_left_x_points[0]+(self.image_width/300),
            bottom_left_y_points[0]-(self.image_width/300),
        )
        self.bottom_left_corner = list(bottom_left_corner)
        print("bottom left point: %s", self.bottom_left_corner)

        # bottom right
        bottom_right = cv2.bitwise_and(self.right_line, self.bottom_line)
        if self.logger_level <= 10:
            # Show the image to check/debug
            cv2.imshow("bottom_right corner", bottom_right)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        bottom_right_y_points, bottom_right_x_points = np.where(bottom_right == 255)
        bottom_right_corner = (
            bottom_right_x_points[0]-(self.image_width/300),
            bottom_right_y_points[0]-(self.image_width/300)
        )
        self.bottom_right_corner = list(bottom_right_corner)
        print("bottom right point: %s", self.bottom_right_corner)

    def transform(self):
        """Transform the raw image, using the found corners, 
        to get a perfect overhead, filling image.
        """

        # Define the four corners of the original image (in order: top-left, top-right, bottom-right, bottom-left)
        original_corners = np.array(
            [
                self.top_left_corner, 
                self.top_right_corner, 
                self.bottom_right_corner, 
                self.bottom_left_corner
            ],
            dtype=np.float32
        )

        # Define the four corners of the desired output image (a perfect overhead view)
        desired_corners = np.array(
            [
                [0, 0], 
                [self.transformed_image_x, 0], 
                [self.transformed_image_x, self.transformed_image_y], 
                [0, self.transformed_image_y]
            ], 
            dtype=np.float32
        )

        # Compute the perspective transformation matrix
        transformation_matrix = cv2.getPerspectiveTransform(original_corners, desired_corners)
        self.transform_matrix = transformation_matrix

        # Apply the perspective transformation
        result = cv2.warpPerspective(self.image, transformation_matrix, (self.transformed_image_x, self.transformed_image_y))

        # Display the transformed image
        if self.logger_level <= 10:
            cv2.imshow('Transformed Image', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        self.transformed_image = result