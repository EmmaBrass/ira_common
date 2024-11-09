import numpy as np
import math
import random
import cv2

import ira_common.configuration as config

class MarkCreator():
    """
    Create a .svg file for the new mark, based on the user's most recent mark.
    """

    def __init__(self, mark, canvas, colors, prev_id=None, debug=False):
        self.mark = mark
        self.canvas = canvas
        self.colors = colors
        # Value for remembering what mark was made last time.
        self.prev_id = prev_id
        # ID for mark being made in this turn.
        self.id = None
        self.debug = debug

    def check_x(self, value):
        """Checks that the given x value is within 
        the canvas border and changes it if not
        """
        if value < 0:
            value = 0
        if value > self.canvas.transformed_image_x:
            value = self.canvas.transformed_image_x

        return value

    def check_y(self, value):
        """Checks that the given y value is within 
        the canvas border and changes it if not
        """
        if value < 0:
            value = 0
        if value > self.canvas.transformed_image_y:
            value = self.canvas.transformed_image_y

        return value

    def map_pixel_to_canvas(self, x_value, y_value):
        """
        Using the canvas object, map the pixel position to 
        the correct position on the svg.
        """

        mapped_x_value = self.canvas.width_mm-((x_value/self.canvas.transformed_image_x)*self.canvas.width_mm)
        mapped_y_value = (y_value/self.canvas.transformed_image_y)*self.canvas.height_mm

        return [mapped_x_value, mapped_y_value]

    def create(self):
        """
        Create the mark.
        
        :returns output_arrray: 2D array of marks in form [[(xa1,ya1),(xa2,ya2),(xa2,ya3)],[(xb1,yb1),(xb2,yb2)]] 
        """
        if self.mark.type == 'blob':
            print("Choosing reponse mark from blob options.")
            output_array = self.blob_options()
        elif self.mark.type == 'straight':
            print("Choosing reponse mark from straight options.")
            output_array = self.straight_options()
        elif self.mark.type == 'curve':
            print("Choosing reponse mark from curve options.")
            output_array = self.curve_options()
        else: 
            print("ERROR! Mark type not recognised.")
        # Return 2D array in form [[(xa1,ya1),(xa2,ya2),(xa2,ya3)],[(xb1,yb1),(xb2,yb2)]]
        return output_array

    def mark_type_id(self):
        """ Return the id of the type of mark just created. """
        return self.id

    def evaluate_quadratic_bezier(self, P0, P1, P2, t):
        """
        Evaluate a quadratic Bezier curve at a given t value.
        :param P0: Tuple representing the start point (x, y).
        :param P1: Tuple representing the control point (x, y).
        :param P2: Tuple representing the end point (x, y).
        :param t: Parameter value between 0 and 1.
        :return: Tuple (x, y) representing the point on the curve.
        """
        x = (1-t)**2 * P0[0] + 2 * (1-t) * t * P1[0] + t**2 * P2[0]
        y = (1-t)**2 * P0[1] + 2 * (1-t) * t * P1[1] + t**2 * P2[1]
        return (x, y)

    def evaluate_cubic_bezier(self, P0, P1, P2, P3, t):
        """
        Evaluate a cubic Bezier curve at a given t value.
        :param P0: Tuple representing the start point (x, y).
        :param P1: Tuple representing the first control point (x, y).
        :param P2: Tuple representing the second control point (x, y).
        :param P3: Tuple representing the end point (x, y).
        :param t: Parameter value between 0 and 1.
        :return: Tuple (x, y) representing the point on the curve.
        """
        x = (1-t)**3 * P0[0] + 3 * (1-t)**2 * t * P1[0] + 3 * (1-t) * t**2 * P2[0] + t**3 * P3[0]
        y = (1-t)**3 * P0[1] + 3 * (1-t)**2 * t * P1[1] + 3 * (1-t) * t**2 * P2[1] + t**3 * P3[1]
        return (x, y)

    def generate_bezier_points(self, start, control1, end, control2=None, num_points=100):
        """
        Generate a list of points along a quadratic or cubic Bezier curve.
        If `control2` is None, a quadratic curve is generated; otherwise, a cubic curve.
        :param start: Tuple representing the start point (x, y).
        :param control1: Tuple representing the first control point (x, y).
        :param end: Tuple representing the end point (x, y).
        :param control2: Optional; Tuple representing the second control point (x, y) for cubic curves.
        :param num_points: Number of points to generate along the curve.
        :return: List of tuples [(x1, y1), (x2, y2), ..., (xn, yn)].
        """
        points = []
        for i in range(num_points):
            t = i / (num_points - 1)  # Generate t values between 0 and 1
            if control2 is None:  # Quadratic Bezier curve
                point = self.evaluate_quadratic_bezier(start, control1, end, t)
            else:  # Cubic Bezier curve
                point = self.evaluate_cubic_bezier(start, control1, control2, end, t)
            points.append(point)
        return points

    def blob_options(self):
        output_array = []
        ran = random.randint(0,100)
        # Ensure we do not make the same kinds of marks over and over again
        if ran < 20 and self.prev_id == 1:
            ran = random.randint(20,100)
        if ran >= 60 and ran < 75 and self.prev_id == 3:
            ran = random.choice([random.randint(0,59),random.randint(75,100)])
        if ran >= 75 and self.prev_id == 4:
            ran = random.randint(0,74)
        if ran < 20: # blob ID 1  
            self.id = 1
            print("New mark will some dots on the blob")
            # draw some dots on the blob
            # Find max and min of x and y.
            max_x = self.check_x(self.mark.max_x)
            min_x = self.check_x(self.mark.min_x)
            max_y = self.check_y(self.mark.max_y)
            min_y = self.check_y(self.mark.min_y)
            # Choose the dots.
            num_dots = int(self.mark.real_area/1300)
            if num_dots < 5:
                num_dots = 5
            if num_dots > 15:
                num_dots = 15
            dot_len = 3
            for i in range(num_dots):
                path = []
                x = random.randint(min_x,max_x)
                y_start = random.randint(min_y,max_y)
                y_end = self.check_y(y_start+dot_len)
                line_start_svg = self.map_pixel_to_canvas(x, y_start)
                line_end_svg = self.map_pixel_to_canvas(x, y_end)
                path.append((line_start_svg[0], line_start_svg[1]))
                path.append((line_end_svg[0], line_end_svg[1]))
                output_array.append(path)
        if ran >= 20 and ran < 60:
            self.id = 2
            print("New mark will be a curve starting on one side of the blob and ending on the other.")
            # Using cubic bezier curves.
            # Points randomly chosen within the blob bounding box.
            # Find max and min of x and y.
            max_x = self.check_x(self.mark.max_x)
            min_x = self.check_x(self.mark.min_x)
            half_x = ((max_x-min_x)/2)+min_x
            max_y = self.check_y(self.mark.max_y)
            min_y = self.check_y(self.mark.min_y)
            half_y = ((max_y-min_y)/2)+min_y
            # Choose two points to be the control points
            # Range determined by canvas size
            range_x = config.CANVAS_WIDTH/2
            range_y = config.CANVAS_HEIGHT/2
            control1_x = self.check_x(random.randint( min_x-range_x, max_x+range_x))
            control1_y = self.check_y(random.randint( min_y-range_y, max_y+range_y))
            control2_x = self.check_x(random.randint( min_x-range_x, max_x+range_x))
            control2_y = self.check_y(random.randint( min_y-range_y, max_y+range_y))
            # Map pixel coordinates to svg positions.
            # Choose the longest length to draw the curve across (x axis or y axis).
            if (max_x - min_x) > (max_y-min_y):
                end1_svg = self.map_pixel_to_canvas(max_x, half_y)
                end2_svg = self.map_pixel_to_canvas(min_x, half_y)
            else:
                end1_svg = self.map_pixel_to_canvas(half_x, min_y)
                end2_svg = self.map_pixel_to_canvas(half_x, max_y)
            curve_control1_svg = self.map_pixel_to_canvas(control1_x, control1_y)
            curve_control2_svg = self.map_pixel_to_canvas(control2_x, control2_y)
            # Coordinates of start point, end point, and control points
            start = (end1_svg[0], end1_svg[1])
            control1 = (curve_control1_svg[0], curve_control1_svg[1])
            control2 = (curve_control2_svg[0], curve_control2_svg[1])
            end = (end2_svg[0], end2_svg[1])
            # Generate points along the Bezier curve
            bezier_points = self.generate_bezier_points(start, control1, end, control2, num_points=100)
            # Append the points to the output_array
            output_array.append(bezier_points)
        if ran >= 60 and ran < 75:
            self.id = 3
            print("New mark will be a cross through the blob.")
            # draw a big cross through the blob
            max_x = self.check_x(self.mark.max_x)
            min_x = self.check_x(self.mark.min_x)
            half_x = ((max_x-min_x)/2)+min_x
            max_y = self.check_y(self.mark.max_y)
            min_y = self.check_y(self.mark.min_y)
            half_y = ((max_y-min_y)/2)+min_y
            # Map pixel coordinates to svg positions
            vert_start_svg = self.map_pixel_to_canvas(half_x, max_y)
            vert_end_svg = self.map_pixel_to_canvas(half_x, min_y)
            hor_start_svg = self.map_pixel_to_canvas(min_x, half_y)
            hor_end_svg = self.map_pixel_to_canvas(max_x, half_y)
            # Add the line to the output_array
            path = [(vert_start_svg[0],vert_start_svg[1]),(vert_end_svg[0],vert_end_svg[1])]
            output_array.append(path)
            path = [(hor_start_svg[0],hor_start_svg[1]),(hor_end_svg[0],hor_end_svg[1])]
            output_array.append(path)
        if ran >= 75:
            self.id = 4
            print("New mark will be drawing around the outline of the blob.")
            # draw along the outline of the blob... lots of little straight lines.
            # get the contour around the edge
            contours, _ = cv2.findContours(self.mark.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = np.zeros(self.mark.mask.shape[:2], dtype="uint8")
            
            cv2.drawContours(contour_image, contours, 0, 255, 1)

            if self.debug == True:
                cv2.imshow('outline of contour', contour_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # Approximate the contour with a polygon and obtain its convex hull
            epsilon = 0.004 * cv2.arcLength(contours[0], True)
            approx = cv2.approxPolyDP(contours[0], epsilon, True)

            cv2.drawContours(contour_image, [approx], 0, 255, 1)

            if self.debug == True:
                cv2.imshow('with polygon', contour_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            hull = cv2.convexHull(approx)

            # Get the coordinates of the vertices of the convex hull
            hull_points = []
            for i in range(len(hull)):
                hull_points.append((hull[i][0][0], hull[i][0][1]))

            # Traverse the contour in order using the vertices of the convex hull
            contour_points = []
            for i in range(len(hull_points)):
                start_point = hull_points[i]
                end_point = hull_points[(i+1) % len(hull_points)]
                points_on_line = np.linspace(start_point, end_point, int(np.linalg.norm(np.array(end_point) - np.array(start_point)))+1)
                contour_points.extend(points_on_line)

            # Convert the contour points to integers
            contour_points = np.round(contour_points).astype(int)

            if self.debug == True:
                contour_lines = np.zeros(self.mark.mask.shape[:2], dtype="uint8")

                for no, point in enumerate(contour_points):
                    if no < len(contour_points)-1:
                        cv2.line(contour_lines, (point[0], point[1]), (contour_points[no+1][0], contour_points[no+1][1]), 255, thickness=1)

                cv2.imshow('lines!', contour_lines)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # Reduce number of contour points by x25
            contour_points = contour_points[::25]

            # Check if points are within canvas borders
            # Map all the points from pixel coordinates to svg coordinates
            contour_points_svg = []
            for point in contour_points:
                point[0] = self.check_x(point[0])
                point[1] = self.check_y(point[1])
                contour_points_svg.append(self.map_pixel_to_canvas(point[0], point[1]))

            path = []
            for no, point in enumerate(contour_points_svg):
                if no < len(contour_points_svg)-1:
                    path.append((point[0],point[1]))
                else:
                    path.append((contour_points_svg[0][0], contour_points_svg[0][1]))
            
            output_array.append(path)
                
        return output_array
            

    def straight_options(self):
        output_array = []
        ran = random.randint(0, 100)
        # Ensure we do not make the same kinds of marks over and over again
        if ran >= 30 and ran < 50 and self.prev_id == 2:
            ran = random.choice([random.randint(0,29),random.randint(50,100)])
        if ran >= 50 and ran < 75 and self.prev_id == 3:
            ran = random.choice([random.randint(0,49),random.randint(75,100)])
        if ran >= 75 and self.prev_id == 4:
            ran = random.randint(0,74)
        # Other mark ideas:
        # one sine wave along the length of the straight line
        # an ellipse around the straight line
        # a parallel line above and below it, a bit longer/shorter or whatever works to keep it in the canvas
        # make the line into one side of a triangle
        if ran < 30:
            self.id = 1
            print("New mark will be a curve joining the two ends of the straight line.")
            # Curve that joins the ends of the straight line.
            # Using cubic bezier curves.
            # Points randomly chosen within the curve bounding box.
            print("before check: %s, %s, %s, %s", self.mark.min_x, self.mark.max_x, self.mark.min_y, self.mark.max_y)
            # Ensure within canvas
            self.mark.min_x = self.check_x(self.mark.min_x)
            self.mark.max_x = self.check_x(self.mark.max_x)
            self.mark.min_y = self.check_x(self.mark.min_y)
            self.mark.max_y = self.check_x(self.mark.max_y)
            print("after check: %s, %s, %s, %s", self.mark.min_x, self.mark.max_x, self.mark.min_y, self.mark.max_y)
            # End points of the curve
            end1_x = self.mark.skel_end_points[0][1]
            end1_y = self.mark.skel_end_points[0][0]
            end2_x = self.mark.skel_end_points[1][1]
            end2_y = self.mark.skel_end_points[1][0]
            # Randomly choose 1 or 2 points to be on the curve
            no_points_ran = random.randint(0,10) # to allow for random selection from 1 or 2 control points on curve
            curve_1_end = [ random.randint(int(self.mark.min_x), int(self.mark.max_x)),
                              random.randint(int(self.mark.min_y), int(self.mark.max_y)) ]
            curve_2_end = [ random.randint(int(self.mark.min_x), int(self.mark.max_x)),
                            random.randint(int(self.mark.min_y), int(self.mark.max_y)) ]
            # Choose control points, and check them
            curve_1_control1 = [end1_x + random.randint(-300,300),end1_y + random.randint(-300,300)]
            curve_1_control1[0] = self.check_x(curve_1_control1[0])
            curve_1_control1[1] = self.check_y(curve_1_control1[1])
            curve_1_control2 = [curve_1_end[0] + random.randint(-300,300), curve_1_end[1] + random.randint(-300,300)]
            curve_1_control2[0] = self.check_x(curve_1_control2[0])
            curve_1_control2[1] = self.check_y(curve_1_control2[1])
            curve_2_control2 = [curve_2_end[0] + random.randint(-300,300), curve_2_end[1] + random.randint(-300,300)]
            curve_2_control2[0] = self.check_x(curve_2_control2[0])
            curve_2_control2[1] = self.check_y(curve_2_control2[1])
            curve_3_control2 = [end2_x + random.randint(-300,300),end2_y + random.randint(-300,300)]
            curve_3_control2[0] = self.check_x(curve_3_control2[0])
            curve_3_control2[1] = self.check_y(curve_3_control2[1])
            # Map pixel coordinates to svg positions
            end1_svg = self.map_pixel_to_canvas(end1_x, end1_y)
            end2_svg = self.map_pixel_to_canvas(end2_x, end2_y)
            curve1_svg = self.map_pixel_to_canvas(curve_1_end[0], curve_1_end[1])
            curve2_svg = self.map_pixel_to_canvas(curve_2_end[0], curve_2_end[1])
            curve1_control1_svg = self.map_pixel_to_canvas(curve_1_control1[0], curve_1_control1[1])
            curve1_control2_svg = self.map_pixel_to_canvas(curve_1_control2[0], curve_1_control2[1])
            curve2_control2_svg = self.map_pixel_to_canvas(curve_2_control2[0], curve_2_control2[1])
            curve3_control2_svg = self.map_pixel_to_canvas(curve_3_control2[0], curve_3_control2[1])

            # Coordinates of start point, end point, and control points
            start = (end1_svg[0], end1_svg[1])
            control1 = (curve1_control1_svg[0], curve1_control1_svg[1])
            control2 = (curve1_control2_svg[0], curve1_control2_svg[1])
            end = (curve1_svg[0], curve1_svg[1])
            # Generate points along the Bezier curve
            bezier_points = self.generate_bezier_points(start, control1, end, control2, num_points=100)
            # Append the points to the output_array
            output_array.append(bezier_points)

            # Coordinates of start point, end point, and control points
            start = (curve1_svg[0], end1_svg[1])
            control1 = (curve3_control2_svg[0], curve3_control2_svg[1])
            end = (end2_svg[0], end2_svg[0])
            # Generate points along the Bezier curve
            bezier_points = self.generate_bezier_points(start, control1, end, None, num_points=100)
            # Append the points to the output_array
            output_array.append(bezier_points)
        if ran >= 30 and ran < 50:
            self.id = 2
            print("New mark will be perpendicular line through midpoint of original line")
            # Straight line that is about perpendicular to original line and same length
            # Find the endpoints of a new line that is perpendicular to the original
            # graient of lines will be 1/skeleton line gradient
            grad = -1/self.mark.gradient
            # Compute the horizontal and vertical components of the line's direction vector
            dx = 1 / math.sqrt(1 + grad**2)
            dy = grad / math.sqrt(1 + grad**2)
            # Compute the coordinates of the endpoints of the line
            x1 = int(self.mark.skel_midpoint_x + self.mark.length/2 * dx)
            y1 = int(self.mark.skel_midpoint_y + self.mark.length/2 * dy)
            x2 = int(self.mark.skel_midpoint_x - self.mark.length/2 * dx)
            y2 = int(self.mark.skel_midpoint_y - self.mark.length/2 * dy)
            # Ensure the new line will not go off the edge of the canvas by too much
            x1 = self.check_x(x1)
            x2 = self.check_x(x2)
            y1 = self.check_y(y1)
            y2 = self.check_y(y2)
            # Map pixel coordinates to svg positions
            end1_svg = self.map_pixel_to_canvas(x1, y1)
            end2_svg = self.map_pixel_to_canvas(x2, y2)
            # Add to the output_array
            path = [(end1_svg[0],end1_svg[1]),(end2_svg[0],end2_svg[1])]
            output_array.append(path)
        if ran >= 50 and ran < 75:
            self.id = 3
            print("New mark will be rotated rect around the line")
            # graient of top and bottom (perpendicular) lines will be 1/skeleton line gradient
            grad = -1/self.mark.gradient
            # Compute the horizontal and vertical components of the line's direction vector
            dx = 1 / math.sqrt(1 + grad**2)
            dy = grad / math.sqrt(1 + grad**2)
            # Compute the coordinates of the rect corners
            x1 = int(self.mark.skel_end_points[0][1] + 
                min(self.mark.min_rect_area[1][0],self.mark.min_rect_area[1][1]) * dx)
            x2 = int(self.mark.skel_end_points[0][1] - 
                min(self.mark.min_rect_area[1][0],self.mark.min_rect_area[1][1]) * dx)
            x3 = int(self.mark.skel_end_points[1][1] - 
                min(self.mark.min_rect_area[1][0],self.mark.min_rect_area[1][1]) * dx)
            x4 = int(self.mark.skel_end_points[1][1] + 
                min(self.mark.min_rect_area[1][0],self.mark.min_rect_area[1][1]) * dx)
            y1 = int(self.mark.skel_end_points[0][0] + 
                min(self.mark.min_rect_area[1][0],self.mark.min_rect_area[1][1]) * dy)
            y2 = int(self.mark.skel_end_points[0][0] - 
                min(self.mark.min_rect_area[1][0],self.mark.min_rect_area[1][1]) * dy)
            y3 = int(self.mark.skel_end_points[1][0] - 
                min(self.mark.min_rect_area[1][0],self.mark.min_rect_area[1][1]) * dy)
            y4 = int(self.mark.skel_end_points[1][0] + 
                min(self.mark.min_rect_area[1][0],self.mark.min_rect_area[1][1]) * dy)
            # Ensure the new points don't go off the canvas
            x1 = self.check_x(x1)
            x2 = self.check_x(x2)
            x3 = self.check_x(x3)
            x4 = self.check_x(x4)
            y1 = self.check_y(y1)
            y2 = self.check_y(y2)
            y3 = self.check_y(y3)
            y4 = self.check_y(y4)
            # Map pixel coordinates to svg positions
            corner1_svg = self.map_pixel_to_canvas(x1, y1)
            corner2_svg = self.map_pixel_to_canvas(x2, y2)
            corner3_svg = self.map_pixel_to_canvas(x3, y3)
            corner4_svg = self.map_pixel_to_canvas(x4, y4)
            # Add to the output_array
            path = [(corner1_svg[0],corner1_svg[1]),(corner2_svg[0],corner2_svg[1])]
            output_array.append(path)
            path = [(corner2_svg[0],corner2_svg[1]),(corner3_svg[0],corner3_svg[1])]
            output_array.append(path)
            path = [(corner3_svg[0],corner3_svg[1]),(corner4_svg[0],corner4_svg[1])]
            output_array.append(path)
            path = [(corner4_svg[0],corner4_svg[1]),(corner1_svg[0],corner1_svg[1])]
            output_array.append(path)
        if ran >= 75:
            self.id = 4
            print("New mark will make the straight line into a triangle")
            # graient of lines will be 1/skeleton line gradient
            grad = -1/self.mark.gradient
            # Compute the horizontal and vertical components of the line's direction vector
            dx = 1 / math.sqrt(1 + grad**2)
            dy = grad / math.sqrt(1 + grad**2)
            # random variables to determine which side of the line we go on, and how far away
            sign = random.choice((-1, 1))
            len = random.uniform(2,5)
            # Compute the coordinates of the triangle point
            tri_x = int(self.mark.skel_midpoint_x + sign * self.mark.length/len * dx)
            tri_y = int(self.mark.skel_midpoint_y + sign * self.mark.length/len * dy)
            # Get end points of original line
            end1_x = self.mark.skel_end_points[0][1]
            end1_y = self.mark.skel_end_points[0][0]
            end2_x = self.mark.skel_end_points[1][1]
            end2_y = self.mark.skel_end_points[1][0]
            # Ensure the triangle corner is on the canvas
            tri_x = self.check_x(tri_x)
            tri_y = self.check_y(tri_y)
            # Map pixel coordinates to svg positions
            end1_svg = self.map_pixel_to_canvas(end1_x, end1_y)
            end2_svg = self.map_pixel_to_canvas(end2_x, end2_y)
            tri_svg = self.map_pixel_to_canvas(tri_x, tri_y)
            # Add to the output_array
            path = [(end1_svg[0],end1_svg[1]),(tri_svg[0],tri_svg[1])]
            output_array.append(path)
            path = [(tri_svg[0],tri_svg[1]),(end2_svg[0],end2_svg[1])]
            output_array.append(path)

        return output_array

    def curve_options(self):
        output_array = []
        ran = random.randint(0, 100)
        # Ensure we do not make the same kinds of marks over and over again
        if ran < 10 and self.prev_id == 1:
            ran = random.randint(10,100)
        if ran >= 10 and ran < 40 and self.prev_id == 2:
            ran = random.choice([random.randint(0,9),random.randint(40,100)])
        if ran >= 80 and self.prev_id == 4:
            ran = random.randint(0,79)
        # Always do just a straight line between end points or bounding box if line gradient is 0 or inf
        # otherwise these gradients values will mess up the more complex algorithms
        # (this should happen very rarely)
        if self.mark.gradient == 0 or self.mark.gradient == float('inf'):
            print("curve gradient is 0 or inf so limiting response mark options (this should happen rarely).")
            if ran < 50:
                ran = 10
            else:
                ran = 90
        if ran < 10:
            self.id=1
            print("New mark will be straight line joining ends of curve") # TODO if the straight line is too short than do something else.
            # straight line joins the ends of the curve
            end1_x = self.mark.skel_end_points[0][1]
            end1_y = self.mark.skel_end_points[0][0]
            end2_x = self.mark.skel_end_points[1][1]
            end2_y = self.mark.skel_end_points[1][0]
            # Map pixel coordinates to svg positions
            end1_svg = self.map_pixel_to_canvas(end1_x, end1_y)
            end2_svg = self.map_pixel_to_canvas(end2_x, end2_y)
            # Check the distance beween points (in mm)
            # Calculate the Euclidean distance
            distance = math.sqrt((end2_svg[0] - end1_svg[0]) ** 2 + (end2_svg[1] - end1_svg[1]) ** 2)
            if distance < 15:
                ran = 50 # do a curve
            else:
                # Add the line to the drawing as a path
                path = [(end1_svg[0],end1_svg[1]),(end2_svg[0],end2_svg[1])]
                output_array.append(path)
        elif ran >= 10 and ran < 40:
            self.id=2
            print("New mark will be multiple straight lines along the curve")
            # straight lines along the curve
            end1_x = self.mark.skel_end_points[0][1]
            end1_y = self.mark.skel_end_points[0][0]
            end2_x = self.mark.skel_end_points[1][1]
            end2_y = self.mark.skel_end_points[1][0]
            # choose no of lines
            no_lines = int(self.mark.length//50)
            # choose a variable that will determine length of lines
            if self.mark.length < 400:
                len_ran = random.randint(2,5)
            else:
                len_ran = random.randint(5,9)
            # choose length of lines
            line_len = self.mark.length/len_ran
            # graient of lines will be 1/skeleton line gradient
            grad = -1/self.mark.gradient
            # Extract the points from the curve contour
            curve_points = self.mark.contour[:, 0, :]
            # Get a skeleton end point
            end_y, end_x = self.mark.skel_end_points[0]
            # Calculate the Euclidean distance between each point and the skeleton end point
            distances = np.linalg.norm(curve_points - [end_x, end_y], axis=1)
            # Find the index of the point with the minimum distance 
            starting_index = np.argmin(distances) 
            # Reorder the points starting from the chosen index
            ordered_points = np.roll(curve_points, -starting_index, axis=0)
            # Take only the first half of the points #TODO this isn't working properly... chooses too many lines I think
            half_point_count = len(ordered_points) // 2
            first_half_points = ordered_points[:half_point_count]
            # Decide on iterator based on no of lines
            iterator = half_point_count//no_lines
            for l in range(no_lines):
                path = []
                # Iterate over points on the contour
                # Choose the point
                point = first_half_points[iterator*l]
                mid_x = point[0]
                mid_y = point[1]
                # Find line equation constant 
                c = mid_y - (mid_x*grad)
                # Compute the horizontal and vertical components of the line's direction vector
                dx = 1 / math.sqrt(1 + grad**2)
                dy = grad / math.sqrt(1 + grad**2)
                # Compute the coordinates of the endpoints of the line
                x1 = int(mid_x + line_len/2 * dx)
                y1 = int(mid_y + line_len/2 * dy)
                x2 = int(mid_x - line_len/2 * dx)
                y2 = int(mid_y - line_len/2 * dy)
                # Ensure the line doesn't go outside the canvas
                x1 = self.check_x(x1)
                y1 = int((x1*grad)+c)
                x2 = self.check_x(x2)
                y2 = int((x2*grad)+c)
                y1 = self.check_y(y1)
                x1 = int((y1-c)/grad)
                y2 = self.check_y(y2)
                x2 = int((y2-c)/grad)
                # Map pixel coordinates to svg positions
                end1_svg = self.map_pixel_to_canvas(x1, y1)
                end2_svg = self.map_pixel_to_canvas(x2, y2)
                # Add the line to the output_array
                path = [(end1_svg[0],end1_svg[1]),(end2_svg[0],end2_svg[1])]
                output_array.append(path)
        elif ran >= 40  and ran < 80:
            self.id = 3
            print("New mark will be another curve connecting ends of the curve")
            # another curve connecting the ends of the curve together
            # using cubic bezier curves
            # points randomly chosen within the curve bounding box
            print("before check: %s, %s, %s, %s", self.mark.min_x, self.mark.max_x, self.mark.min_y, self.mark.max_y)
            # Ensure within canvas
            self.mark.min_x = self.check_x(self.mark.min_x)
            self.mark.max_x = self.check_x(self.mark.max_x)
            self.mark.min_y = self.check_x(self.mark.min_y)
            self.mark.max_y = self.check_x(self.mark.max_y)
            print("after check: %s, %s, %s, %s", self.mark.min_x, self.mark.max_x, self.mark.min_y, self.mark.max_y)
            # End points of the curve
            end1_x = self.mark.skel_end_points[0][1]
            end1_y = self.mark.skel_end_points[0][0]
            end2_x = self.mark.skel_end_points[1][1]
            end2_y = self.mark.skel_end_points[1][0]
            # Randomly choose 1 or 2 points to be on the curve
            no_points_ran = random.randint(0,10) # to allow for random selection from 1 or 2 control points on curve
            curve_1_end = [ random.randint(int(self.mark.min_x), int(self.mark.max_x)),
                              random.randint(int(self.mark.min_y), int(self.mark.max_y)) ]
            curve_2_end = [ random.randint(int(self.mark.min_x), int(self.mark.max_x)),
                            random.randint(int(self.mark.min_y), int(self.mark.max_y)) ]
            # Choose control points, and check them
            curve_1_control1 = [end1_x + random.randint(-300,300),end1_y + random.randint(-300,300)]
            curve_1_control1[0] = self.check_x(curve_1_control1[0])
            curve_1_control1[1] = self.check_y(curve_1_control1[1])
            curve_1_control2 = [curve_1_end[0] + random.randint(-300,300), curve_1_end[1] + random.randint(-300,300)]
            curve_1_control2[0] = self.check_x(curve_1_control2[0])
            curve_1_control2[1] = self.check_y(curve_1_control2[1])
            curve_2_control2 = [curve_2_end[0] + random.randint(-300,300), curve_2_end[1] + random.randint(-300,300)]
            curve_2_control2[0] = self.check_x(curve_2_control2[0])
            curve_2_control2[1] = self.check_y(curve_2_control2[1])
            curve_3_control2 = [end2_x + random.randint(-300,300),end2_y + random.randint(-300,300)]
            curve_3_control2[0] = self.check_x(curve_3_control2[0])
            curve_3_control2[1] = self.check_y(curve_3_control2[1])
            # Map pixel coordinates to svg positions
            end1_svg = self.map_pixel_to_canvas(end1_x, end1_y)
            end2_svg = self.map_pixel_to_canvas(end2_x, end2_y)
            curve1_svg = self.map_pixel_to_canvas(curve_1_end[0], curve_1_end[1])
            curve2_svg = self.map_pixel_to_canvas(curve_2_end[0], curve_2_end[1])
            curve1_control1_svg = self.map_pixel_to_canvas(curve_1_control1[0], curve_1_control1[1])
            curve1_control2_svg = self.map_pixel_to_canvas(curve_1_control2[0], curve_1_control2[1])
            curve2_control2_svg = self.map_pixel_to_canvas(curve_2_control2[0], curve_2_control2[1])
            curve3_control2_svg = self.map_pixel_to_canvas(curve_3_control2[0], curve_3_control2[1])

            # Coordinates of start point, end point, and control points
            start = (end1_svg[0], end1_svg[1])
            control1 = (curve1_control1_svg[0], curve1_control1_svg[1])
            control2 = (curve1_control2_svg[0], curve1_control2_svg[1])
            end = (curve1_svg[0], curve1_svg[1])
            # Generate points along the Bezier curve
            bezier_points = self.generate_bezier_points(start, control1, end, control2, num_points=100)
            # Append the points to the output_array
            output_array.append(bezier_points)

            # Coordinates of start point, end point, and control points
            start = (curve1_svg[0], end1_svg[1])
            control1 = (curve3_control2_svg[0], curve3_control2_svg[1])
            end = (end2_svg[0], end2_svg[0])
            # Generate points along the Bezier curve
            bezier_points = self.generate_bezier_points(start, control1, end, None, num_points=100)
            # Append the points to the output_array
            output_array.append(bezier_points)

        elif ran >= 80:
            self.id = 4
            print("New mark will be a box around the curve")
            # A box around the curve
            # Ensure within border
            self.mark.min_x = self.check_x(self.mark.min_x) 
            self.mark.max_x = self.check_x(self.mark.max_x)
            self.mark.min_y = self.check_y(self.mark.min_y)
            self.mark.max_y = self.check_y(self.mark.max_y)
            # Find corner coordinates
            top_left = [self.mark.min_x, self.mark.min_y]
            top_right = [self.mark.max_x, self.mark.min_y]
            bottom_left = [self.mark.min_x, self.mark.max_y]
            bottom_right = [self.mark.max_x, self.mark.max_y]
            # Map pixel coordinates to svg positions
            top_left_svg = self.map_pixel_to_canvas(top_left[0],top_left[1])
            top_right_svg = self.map_pixel_to_canvas(top_right[0],top_right[1])
            bottom_left_svg = self.map_pixel_to_canvas(bottom_left[0],bottom_left[1])
            bottom_right_svg = self.map_pixel_to_canvas(bottom_right[0],bottom_right[1])
            # Add the lines to the output_array
            path = [(top_left_svg[0], top_left_svg[1]),(top_right_svg[0],top_right_svg[1])]
            output_array.append(path)
            path = [(top_right_svg[0], top_right_svg[1]),(bottom_right_svg[0],bottom_right_svg[1])]
            output_array.append(path)
            path = [(bottom_right_svg[0], bottom_right_svg[1]),(bottom_left_svg[0],bottom_left_svg[1])]
            output_array.append(path)
            path = [(bottom_left_svg[0], bottom_left_svg[1]),(top_left_svg[0],top_left_svg[1])]
            output_array.append(path)

        return output_array
    
    def choose_color(self):
        """
        Chooses the color that the robot's mark will be.
        For now, chooses a color furthest away from the color of the human user's mark.
        """
        # human_color = self.mark.color_bgr
        # print("Average color for this human mark is: %s", human_color)
        # max_dist = 0
        # max_dist_pot_num = 0
        # for pot_num, robot_color in enumerate(self.colors):
        #     print("Robot color for pot num %s: %s", 
        #         pot_num, robot_color)
        #     r_diff = human_color[2] - robot_color[0]
        #     g_diff = human_color[1] - robot_color[1]
        #     b_diff = human_color[0] - robot_color[2]
        #     overall_dist = math.sqrt((r_diff**2)+(g_diff**2)+(b_diff**2))
        #     print("Overall color distance for pot no %s: %s", 
        #         pot_num, overall_dist)
        #     if overall_dist > max_dist:
        #         max_dist = overall_dist
        #         max_dist_pot_num = pot_num + 1
        # print("Max dist pot num: %s", max_dist_pot_num)

        # just choose a random color por number for now
        choice = random.randint(1,len(self.colors))
        return choice

    # blob -> draw outline of the blob.  Blob if skel line length / area is below a threshold

    # multiple marks -> decide how to react to each one individually
    # in difference image, select ALL the contours with area above a threshold and within canvas bounds, to get ALL the new marks
    # also contours that are enclosing other marks but are not enclosed themselves.