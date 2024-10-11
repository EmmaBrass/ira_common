from xarm.wrapper import XArmAPI #TODO install
import cv2
import math, random
import ira_common.configuration as config
from ira_common.canvas import Canvas
from ira_common.all_marks import AllMarks
from ira_common.mark_creator import MarkCreator

class ArmMovements():
    """
    Arm movement commands for the UFactory XArm 6.
    """
    def __init__(self) -> None:
        self.arm = XArmAPI('192.168.1.200')
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        
        # TODO Turn on collision detection
        # TODO set velocity
        # TODO set_mount_direction()

        # initialise images for ira_collab
        self._initial_image = None
        self._before_image = None
        self._after_image = None
        # initialise variable for holding canvas object
        self.canvas = None
        #values for remembering previous mark types
        self.type_id = {'blob': None, 'straight': None, 'curve': None}


        self.vertical = config.VERTICAL # If true, vertical surface. If False, horizontal

        self.vertical_dist = 40 # Distance in cm of the vertical surface from the robot base
        self.vertical_height = 0 # Distance in cm of the bottom of the vertical surface from the bottom of the robot base
        self.vertical_painting_width = config.VER_PAINT_WIDTH # max reach of robot
        self.vertical_painting_height = config.VER_PAINT_HEIGHT # max reach of robot

        self.horizontal_painting_width = config.HOR_PAINT_WIDTH # max reach of robot
        self.horizontal_painting_height = config.HOR_PAINT_HEIGHT # max reach of robot

        self.light = config.LIGHT # False = using a pen/paintbrush instead (hence need x movement away from canvas between contours)

        # Speed settings
        self.painting_speed = config.PAINT_SPEED # speed when making a mark
        self.between_speed = config.BETWEEN_SPEED # speed when moving between marks/pots

        # For tracking distance travelled - for brush reload
        self.reload_dist = 500 #mm, how much to paint before a reload
        self.travelled_dist = 0

        # For paint reload pot
        self.x_dist_reload = 200 # x distance in mm from the top-left origin corner
        self.y_dist_reload = -50 # y distance in mm from the top-left origin corner

        # For brush lifts
        self.brush_lift = 15 # the amount to lift brush off page between lines, mm
        self.pot_lift = 70 # distance in mm to lift when going to paint pot

        # Start position for HORIZONTAL
        # Set the values here if you want to change start position
        self.hor_x_start = config.HOR_X_START
        self.hor_y_start = config.HOR_Y_START
        self.hor_z_start = config.HOR_Z_START
        self.hor_roll_start = config.HOR_ROLL_START
        self.hor_pitch_start = config.HOR_PITCH_START
        self.hor_yaw_start = config.HOR_YAW_START

        # Start position for VERTICAL
        # Set the values here if you want to change start position
        self.ver_x_start = config.VER_X_START
        self.ver_y_start = config.VER_Y_START
        self.ver_z_start = config.VER_Z_START
        self.ver_roll_start = config.VER_ROLL_START
        self.ver_pitch_start = config.VER_PITCH_START
        self.ver_yaw_start = config.VER_YAW_START

        # Go to initial position
        self.initial_position()

    @initial_image.setter
    def initial_image(self, image):
        self._initial_image = image

    @before_image.setter
    def before_image(self, image):
        self._before_image = image

    @after_image.setter
    def after_image(self, image):
        self._after_image = image

    def initial_position(self):
        """
        Move to initial position.
        """
        self.arm.set_servo_angle(angle=[0, -35, -16, 0, 51, 0], is_radian=False, speed=self.between_speed/2, wait=True)
        print("Done returning to initial position")

    def vert_initial_position(self):
        """
        Move to initial vertical position, 
        to get the arm in a good orientation for vert painitng.
        """
        self.arm.set_servo_angle(angle=[180, -48, -77.3, 0, 10.8, 0], is_radian=False, speed=self.between_speed/2, wait=True)
        print("Done moving to vert initial position")

    def canvas_initialise(self):
        """ 
        Use with ira_collab.
        Initialise the blank canvas.
        """
        if self._initial_image != None:
            # initialise the canvas object
            self.canvas = Canvas()
            self.canvas.set_image(self._initial_image)
            self.canvas.set_real_dimensions(
                config.CANVAS_WIDTH,
                config.CANVAS_HEIGHT
            )
            self.canvas.analyse()
        else:
            print("ERROR! No initial image.")

    def look_at_canvas(self):
        """
        Used with ira_collab.
        Face the eyes downwards to look at the canvas, to take a pic.
        """
        # TODO find good position

    def lift_up(self):
        """
        Use with ira_collab.
        Lift the end effector up a bit higher than initial position.
        To give room for the human to paint unobstructed.
        """
        # TODO find good position.

    def acknowledge(self):
        """
        Use with ira_collab.
        Do a little head twist kind of thing for when the painting is complete.
        """
        # TODO find path.

    def paint_abstract_mark(self, before_image, after_image):
        """
        Use with ira_collab.
        The main meat of the system: this method takes the before and after
        images, find the difference, then chooses how to react to the mark,
        makes the path for the robot, and outputs it.

        :param before_image: Image of the canvas before the human mark.
        :param after_image: Image of the canvas after the human mark.
        """

        if self.canvas == None:
            print("ERROR! No canvas object.")

        if self._before_image != None:
            before_trans = cv2.warpPerspective(
                self._before_image,
                self.canvas.transform_matrix,
                (self.canvas.transformed_image_x, self.canvas.transformed_image_y)
            )
            # diff will be done with before and after image
        else:
            before_trans = cv2.warpPerspective(
                self._initial_image,
                self.canvas.transform_matrix,
                (self.canvas.transformed_image_x, self.canvas.transformed_image_y)
            )
        after_trans = cv2.warpPerspective(
            self._after_image,
            self.canvas.transform_matrix,
            (self.canvas.transformed_image_x, self.canvas.transformed_image_y)
        )
        # Load the transformed images into the AllMarks object, which will find all the new marks
        all_marks = AllMarks(self.canvas)
        all_marks.set_old_image(before_trans)
        all_marks.set_new_image(after_trans)
        # Find all marks, run mark type analysis, color analysis, skeletonisation, etc.
        all_marks.find_all_marks()
        # Get array with all the new marks in it
        marks_array = all_marks.get_all_marks()

        # If more than 1 mark was made by the human, randomly choose a 
        # set number of marks to respond to, otherwise keep the original marks array.
        final_marks_array = []
        if len(marks_array) > config.NUM_MARKS:
            final_marks_array = random.sample(marks_array, config.NUM_MARKS) # Only respond to this many marks
        else:
            final_marks_array = marks_array

        for num, mark in enumerate(final_marks_array):

            #Create a mark based on the user's mark - makes an .svg file of the next mark for the robot to make
            if mark.type == "blob":
                mark_creator = MarkCreator(
                    mark, 
                    canvas, 
                    config.COLORS, 
                    prev_id = self.type_id['blob']
                    )
            elif mark.type == "straight":
                mark_creator = MarkCreator(
                    mark, 
                    canvas, 
                    config.COLORS, 
                    prev_id = self.type_id['straight']
                    )
            elif mark.type == "curve":
                mark_creator = MarkCreator(
                    mark, 
                    canvas, 
                    config.COLORS, 
                    prev_id = self.type_id['curve']
                    )
            output_array = mark_creator.create()
            color_pot = mark_creator.choose_color()
            self.type_id[mark.type] = mark_creator.mark_type_id()
            logger.info("type_id dictionary is now: %s", type_id)


            # MarkCreator class will need to be edited pretty substantially to 
            # output a path for this robot rather than an svg file I think.  
            # TODO will have to play with the reload_brush function to get it to work with multiple colors.

            return output_array, color_pot

    def paint_marks(self, coordinates)
        """
        Paint the marks for ira_collab.

        :param coordinates: 2D array of contours coordinates, in format (x,y)
        """
        prev_x, prev_y = 0, 0
        y_abs = self.hor_y_start
        x_abs = self.hor_x_start
        # Do a first brush load
        self.reload_brush(x_abs, y_abs, False, True)
        for contour in mapped_coordinates:
            if self.travelled_dist >= self.reload_dist:
                self.reload_brush(x_abs, y_abs, False, False)
                self.travelled_dist = 0
            paths = []
            current_path = []
            start_x_abs = -1
            start_y_abs = -1
            for pair in contour:
                x, y = pair
                y_abs = self.hor_y_start+x
                x_abs = self.hor_x_start+y
                if start_x_abs == -1 and start_y_abs == -1:
                    start_x_abs = x_abs
                    start_y_abs = y_abs
                current_path.append([x_abs, y_abs, self.hor_z_start, None, None, None, 50])
                if prev_x != 0 and prev_y != 0:
                    x_change = x - prev_x
                    y_change = y - prev_y
                else:
                    x_change = 0
                    y_change = 0
                prev_x = x
                prev_y = y
                dist_change = math.sqrt(x_change**2+y_change**2)
                self.travelled_dist += dist_change
                if self.travelled_dist >= self.reload_dist:
                    # Add current path to the paths array and start new current path
                    paths.append(current_path)
                    if len(current_path) >= 2:
                        current_path = [current_path[-2], current_path[-1]] # include the last two points from prev path
                    else:
                        current_path = []
                    self.travelled_dist = 0
            # Add the last current_path to paths, if there are coordinates in it
            if len(current_path) > 0:
                paths.append(current_path)
            # Move to the start of the path
            if self.light == False:
                # Lift up the pen
                self.arm.set_position(
                    x=None, 
                    y=None, 
                    z=self.hor_z_start+self.brush_lift, 
                    roll=None, 
                    pitch=None, 
                    yaw=None, 
                    speed=self.between_speed, 
                    relative=False, 
                    wait=True
                )
                # Correct servo 6 angle
                self.arm.set_servo_angle(
                    servo_id=6, 
                    angle=0, 
                    speed=70,
                    is_radian=False, 
                    wait=True
                )
                # Do the movement
                self.arm.set_position(
                    x=start_x_abs, 
                    y=start_y_abs, 
                    z=self.hor_z_start+self.brush_lift, 
                    roll=None, 
                    pitch=None, 
                    yaw=None, 
                    speed=self.between_speed, 
                    relative=False, 
                    wait=True
                )
                # Put the pen down
                self.arm.set_position(
                    x=None, 
                    y=None, 
                    z=self.hor_z_start, 
                    roll=None, 
                    pitch=None, 
                    yaw=None, 
                    speed=self.between_speed, 
                    relative=False, 
                    wait=True
                )
            else:
                # Turn off the light
                # Do the movement to next path
                # Turn on the light
                pass
            
            # Do the movement for the path(s)
            for num, current_path in enumerate(paths):
                self.arm.move_arc_lines(
                    current_path, 
                    is_radian=False, 
                    times=1, 
                    first_pause_time=0.1, 
                    repeat_pause_time=0, 
                    automatic_calibration=True, 
                    speed=self.painting_speed, 
                    mvacc=1500, 
                    wait=True
                )
                # Do a reload between current_path and next one (if more than 1)
                if num < len(paths)-1:
                    self.reload_brush(current_path[-1][0], current_path[-1][1], True, False)
            
        # Return to initial position
        self.initial_position()


    def straight_fw_position(self, set_speed):
        """
        Move to a straight up position facing forward, 
        useful when moving to vertical locations.

        :param set_speed: use from 0-100
        """
        code, value = self.arm.get_servo_angle(servo_id=1, is_radian=False, is_real=False)
        servo_1_angle = int(value)
        code, values = self.arm.get_position()
        if code == 0:
            # Extract the z value, which is the third element in the list
            x_value = values[0]
        else:
            print("error!!!!")
        # If it's currently facing backwards
        if servo_1_angle > 90 or servo_1_angle < -90:
            if x_value < -300:
                # Move away from the wall a bit (+ve x direction)
                self.arm.set_position(x=50, y=None, z=None, roll=None, pitch=None, yaw=None, speed=set_speed, relative=True, wait=True, motion_type=2)
            # Go upright
            print("HERE")
            self.arm.set_servo_angle(angle=[180, -33.1, -126.8, 0, 61.7, 0], is_radian=False, speed=set_speed, relative=False, wait=True)
        else:
            # Move up a bit in Z in case it's on the page (+ve z direction)
            self.arm.set_position(x=None, y=None, z=50, roll=None, pitch=None, yaw=None, speed=set_speed, relative=True, wait=True, motion_type=2)
        # Face straight forwards
        self.arm.set_servo_angle(servo_id=1, angle=0, speed=40, relative=False, wait=True)
        # Go upright
        self.arm.set_servo_angle(angle=[0, -31.5, -117.6, 0, 113.9, 0], is_radian=False, speed=set_speed, wait=True)
        print("Done moving to straight position")

    def straight_bw_position(self, set_speed):
        """
        Move to a straight up position facing backwards, 
        useful when moving to vertical locations.

        :param set_speed: use from 0-100
        """
        # Move up a bit in Z in case it's on the page (+ve z direction)
        code, value = self.arm.get_servo_angle(servo_id=1, is_radian=False, is_real=False)
        servo_1_angle = int(value)
        code, values = self.arm.get_position()
        if code == 0:
            # Extract the z value, which is the third element in the list
            z_value = values[2]
        else:
            print("error!!!!")
        # If it's currently facing forwards
        if servo_1_angle <= 90 and servo_1_angle >= -90:
            if z_value < 300:
                # Move up a bit in Z in case it's on the page (+ve z direction)
                self.arm.set_position(x=None, y=None, z=50, roll=None, pitch=None, yaw=None, speed=set_speed, relative=True, wait=True, motion_type=2)
            # Go upright
            self.arm.set_servo_angle(angle=[0, -31.5, -117.6, 0, 113.9, 0], is_radian=False, speed=set_speed, wait=True)
        # If it's already facing backwards
        else:
            # Move away from the wall a bit (+ve x direction)
            self.arm.set_position(x=50, y=None, z=None, roll=None, pitch=None, yaw=None, speed=set_speed, relative=True, wait=True, motion_type=2)
            # Go upright
            self.arm.set_servo_angle(angle=[180, -33.1, -126.8, 0, 61.7, 0], is_radian=False, speed=set_speed, wait=True)
        # Face straight backwards
        self.arm.set_servo_angle(servo_id=1, angle=180, speed=40, relative=False, wait=True)
        # Go upright
        self.arm.set_servo_angle(angle=[180, -33.1, -126.8, 0, 61.7, 0], is_radian=False, speed=set_speed, wait=True)
        print("Done moving to straight position")

    def scan(self):
        """
        Move to a new random position within the viewing plane.
        """
        self.arm.set_position(
            x=self.hor_x_start+100, 
            y=self.hor_y_start+random.randint(100,400),
            z=self.hor_z_start+random.randint(200,400),
            roll=None, 
            pitch=None, 
            yaw=None, 
            speed=self.between_speed, 
            relative=False, 
            wait=True
        )

    def stop(self, x, y, z):
        """
        Don't move.
        """
        pass

    def resize_and_center_image(self, image_x, image_y, target_width, target_height):
        """
        Resize and centre image for the drawing space
        """
        image_height, image_width = image_y, image_x
        
        # Calculate the scaling factor
        width_ratio = target_width / image_width
        height_ratio = target_height / image_height
        scaling_factor = min(width_ratio, height_ratio)
        
        # Calculate the new size to maintain aspect ratio
        new_width = int(image_width * scaling_factor)
        new_height = int(image_height * scaling_factor)

        # Calculate offsets
        offset_x = (target_width - new_width) // 2
        offset_y = (target_height - new_height) // 2
        
        return offset_x, offset_y, scaling_factor
    
    def map_coordinates(self, coordinates, offset_x, offset_y, scaling_factor):
        """
        Map the original coordinates into the new drawing space.
        """
        new_coordinates = []
        for contour in coordinates:
            new_contour = []
            for x, y in contour:
                new_x = int(x * scaling_factor) + offset_x
                new_y = int(y * scaling_factor) + offset_y
                new_contour.append((new_x, new_y))
            new_coordinates.append(new_contour)
        return new_coordinates
    
    def calculate_distance(self, point1, point2):
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    def reorder_paths_greedy(self, mapped_coordinates):
        """
        Use greedy algo to reorder paths to have the shortest
        travel distances between end of one and start of another.
        """

        if not mapped_coordinates:
            return []
        
        # Initialize the reordered list with the first path
        reordered_paths = [mapped_coordinates.pop(0)]
        
        while mapped_coordinates:
            last_point = reordered_paths[-1][-1]
            closest_index = None
            closest_distance = float('inf')
            
            for i, path in enumerate(mapped_coordinates):
                start_point = path[0]
                distance = self.calculate_distance(last_point, start_point)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_index = i
            
            # Append the closest path to the reordered list and remove it from the original list
            reordered_paths.append(mapped_coordinates.pop(closest_index))
        
        return reordered_paths

    def laser_image(self, coordinates, image_x, image_y):
        """
        'Draw' the image using a laser.
        :param coordinates: 2D array of contours coordinates, in format (x,y)
        :param image_x: x height of original image
        :param image_y: y width of original image
        """
        print("Doing laser drawing of face")
        self.initial_position()

        # Map the contour coordinates into the vertical drawing space, small canvas
        offset_x, offset_y, scaling_factor = self.resize_and_center_image(
            image_x, 
            image_y, 
            config.VER_PAINT_WIDTH,
            config.VER_PAINT_HEIGHT
        )
        small_mapped_coordinates = self.map_coordinates(coordinates, offset_x, offset_y, scaling_factor)
        
        # Map the contour coordinates onto the vertical large canvas.
        offset_x, offset_y, scaling_factor = self.resize_and_center_image(
            image_x, 
            image_y, 
            config.SURFACE_WIDTH, 
            config.SURFACE_HEIGHT
        )
        large_mapped_coordinates = self.map_coordinates(coordinates, offset_x, offset_y, scaling_factor)

        horizontal_translate = int((config.SURFACE_WIDTH - config.VER_PAINT_WIDTH)/2)
        vertical_translate = int((config.SURFACE_HEIGHT - config.VER_PAINT_HEIGHT))
        translated_large_mapped_coordinates = []
        # Now need to change the large_mapped_coordinates to have the same origin point as small_mapped_coordinates
        for contour in large_mapped_coordinates:
            new_contour = []
            for x, y in contour:
                new_x = x - horizontal_translate
                new_y = y - vertical_translate
                new_contour.append((new_x, new_y))
            translated_large_mapped_coordinates.append(new_contour)

        # Now for each contour point in small_mapped_coordinates, it will have a corresponding theta and phi.
        # We will make a new 2D array like: [[(x1a, y1a, theta1a, phi1a), (x1b,y1b,theta1b, phi1b)], [(x2a, y2a, theta2a, phi2a), (x2b,y2b,theta2b, phi2b)]]
        small_mapped_coordinates_with_angles = []
        for small_contour, large_contour in zip(small_mapped_coordinates, translated_large_mapped_coordinates):
            new_contour = []
            for small_pair, large_pair in zip(small_contour, large_contour):
                x_small, y_small = small_pair
                x_large, y_large = large_pair
                delta_x = large_x - small_x
                delta_y = large_y - small_y
                theta = math.degrees(math.atan(delta_x/config.X_DIST)) # in degrees
                phi = math.degrees(math.atam(delta_y/config.X_DIST)) # in degrees
                new_contour.append((x_small, y_small, theta, phi))
            small_mapped_coordinates_with_angles.append(new_contour)

        # Now, how can theta and phi be used in a function for the robot?
        # theta = rotation around Z axis = RZ = Yaw
        # phi = rotation around Y axis = RY = Pitch

        # Now we make the path to give to the move_arc_lines function
        y_abs = config.LASER_Y_START
        z_abs = config.LASER_Z_START
        for contour in small_mapped_coordinates_with_angles:
            path = []
            start_y_abs = -1
            start_z_abs = -1
            for x, y, theta, phi in contour:
                y_abs = config.LASER_Y_START - x
                z_abs = config.LASER_Z_START- y
                if start_y_abs == -1 and start_z_abs == -1:
                    start_y_abs = y_abs
                    start_z_abs = z_abs
                    start_pitch = phi
                    start_yaw = theta
                path.append([config.LASER_X_START, y_abs, z_abs, None, phi, theta, 50]) # TODO what is this 50?

            # Move to the start of the next path

            # TODO Turn off the light
            print("laser turning off")

            # Do the movement to start of next path
            self.arm.set_position(
                x=config.LASER_X_START, 
                y=start_y_abs, 
                z=start_z_abs,
                roll=config.LASER_ROLL_START, 
                pitch=start_pitch, 
                yaw=start_yaw, 
                speed=self.between_speed, 
                relative=False, 
                wait=True
            )

            # TODO Turn on the light
            print("laser turning on")

            # Draw the line
            self.arm.move_arc_lines(
                path, 
                is_radian=False, 
                times=1, 
                first_pause_time=0.1, 
                repeat_pause_time=0, 
                automatic_calibration=True, 
                speed=self.painting_speed, 
                mvacc=1500, 
                wait=True
            )

        # Return to initial position
        self.initial_position()

                
    def paint_image(self, coordinates, image_x, image_y):
        """
        Paint the face image!

        :param coordinates: 2D array of contours coordinates, in format (x,y)
        :param image_x: x height of original image
        :param image_y: y width of original image
        """

        if self.vertical == True:
            print("Doing vertical painting of face")
            self.initial_position()

            # Map the contour coordinates into the vertical drawing space
            offset_x, offset_y, scaling_factor = self.resize_and_center_image(
                image_x, 
                image_y, 
                self.vertical_painting_width, 
                self.vertical_painting_height
            )
            mapped_coordinates = self.map_coordinates(coordinates, offset_x, offset_y, scaling_factor)
            #reordered_paths = self.reorder_paths_greedy(mapped_coordinates)

            print(f"{mapped_coordinates=}")

            prev_x, prev_y = 0, 0
            y_abs = self.ver_y_start
            z_abs = self.ver_z_start
            # Do a first brush load
            self.reload_brush(y_abs, z_abs, False, True)
            for contour in mapped_coordinates:
                if self.travelled_dist >= self.reload_dist:
                    self.reload_brush(y_abs, z_abs, False, False)
                    self.travelled_dist = 0
                paths = []
                current_path = []
                start_y_abs = -1
                start_z_abs = -1
                for pair in contour:
                    x, y = pair
                    y_abs = self.ver_y_start+x
                    z_abs = self.ver_z_start-y
                    if start_y_abs == -1 and start_z_abs == -1:
                        start_y_abs = y_abs
                        start_z_abs = z_abs
                    current_path.append([self.ver_x_start, y_abs, z_abs, None, None, None, 50])
                    if prev_x != 0 and prev_y != 0:
                        x_change = x - prev_x
                        y_change = y - prev_y
                    else:
                        x_change = 0
                        y_change = 0
                    prev_x = x
                    prev_y = y
                    dist_change = math.sqrt(x_change**2+y_change**2)
                    self.travelled_dist += dist_change
                    if self.travelled_dist >= self.reload_dist:
                        # Add current path to the paths array and start new current path
                        paths.append(current_path)
                        if len(current_path) >= 2:
                            current_path = [current_path[-2], current_path[-1]] # include the last two points from prev path
                        else:
                            current_path = []
                        self.travelled_dist = 0
                # Add the last current_path to paths, if there are coordinates in it
                if len(current_path) > 0:
                    paths.append(current_path)
                # Move to the start of the path
                if self.light == False:
                    # Lift up the pen
                    self.arm.set_position(
                        x=self.ver_x_start+self.brush_lift,  
                        y=None, 
                        z=None,
                        roll=None, 
                        pitch=None, 
                        yaw=None, 
                        speed=self.between_speed, 
                        relative=False, 
                        wait=True
                    )
                    # Correct servo 6 angle
                    self.arm.set_servo_angle(
                        servo_id=6, 
                        angle=0, 
                        speed=70,
                        is_radian=False, 
                        wait=True
                    )
                    # Do the movement
                    self.arm.set_position(
                        x=self.ver_x_start+self.brush_lift, 
                        y=start_y_abs, 
                        z=start_z_abs,
                        roll=None, 
                        pitch=None, 
                        yaw=None, 
                        speed=self.between_speed, 
                        relative=False, 
                        wait=True
                    )
                    # Put the pen down
                    self.arm.set_position(
                        x=self.ver_x_start,
                        y=None, 
                        z=None,
                        roll=None, 
                        pitch=None, 
                        yaw=None, 
                        speed=self.between_speed, 
                        relative=False, 
                        wait=True
                    )
                else:
                    # Turn off the light
                    # Do the movement to next path
                    # Turn on the light
                    pass
                
                # Do the movement for the path(s)
                for num, current_path in enumerate(paths):
                    self.arm.move_arc_lines(
                        current_path, 
                        is_radian=False, 
                        times=1, 
                        first_pause_time=0.1, 
                        repeat_pause_time=0, 
                        automatic_calibration=True, 
                        speed=self.painting_speed, 
                        mvacc=1500, 
                        wait=True
                    )
                    # Do a reload between current_path and next one (if more than 1)
                    if num < len(paths)-1:
                        self.reload_brush(current_path[-1][1], current_path[-1][2], True, False)

            # Return to initial position
            self.straight_fw_position(25)
            self.initial_position()
                    
        else:
            print("Doing horizontal painting of face")
            self.initial_position()

            # Map the contour coordinates into the horizontal drawing space
            offset_x, offset_y, scaling_factor = self.resize_and_center_image(
                image_x, 
                image_y, 
                self.horizontal_painting_width, 
                self.horizontal_painting_height
            )
            mapped_coordinates = self.map_coordinates(coordinates, offset_x, offset_y, scaling_factor)

            print(f"{mapped_coordinates=}")

            prev_x, prev_y = 0, 0
            y_abs = self.hor_y_start
            x_abs = self.hor_x_start
            # Do a first brush load
            self.reload_brush(x_abs, y_abs, False, True)
            for contour in mapped_coordinates:
                if self.travelled_dist >= self.reload_dist:
                    self.reload_brush(x_abs, y_abs, False, False)
                    self.travelled_dist = 0
                paths = []
                current_path = []
                start_x_abs = -1
                start_y_abs = -1
                for pair in contour:
                    x, y = pair
                    y_abs = self.hor_y_start+x
                    x_abs = self.hor_x_start+y
                    if start_x_abs == -1 and start_y_abs == -1:
                        start_x_abs = x_abs
                        start_y_abs = y_abs
                    current_path.append([x_abs, y_abs, self.hor_z_start, None, None, None, 50])
                    if prev_x != 0 and prev_y != 0:
                        x_change = x - prev_x
                        y_change = y - prev_y
                    else:
                        x_change = 0
                        y_change = 0
                    prev_x = x
                    prev_y = y
                    dist_change = math.sqrt(x_change**2+y_change**2)
                    self.travelled_dist += dist_change
                    if self.travelled_dist >= self.reload_dist:
                        # Add current path to the paths array and start new current path
                        paths.append(current_path)
                        if len(current_path) >= 2:
                            current_path = [current_path[-2], current_path[-1]] # include the last two points from prev path
                        else:
                            current_path = []
                        self.travelled_dist = 0
                # Add the last current_path to paths, if there are coordinates in it
                if len(current_path) > 0:
                    paths.append(current_path)
                # Move to the start of the path
                if self.light == False:
                    # Lift up the pen
                    self.arm.set_position(
                        x=None, 
                        y=None, 
                        z=self.hor_z_start+self.brush_lift, 
                        roll=None, 
                        pitch=None, 
                        yaw=None, 
                        speed=self.between_speed, 
                        relative=False, 
                        wait=True
                    )
                    # Correct servo 6 angle
                    self.arm.set_servo_angle(
                        servo_id=6, 
                        angle=0, 
                        speed=70,
                        is_radian=False, 
                        wait=True
                    )
                    # Do the movement
                    self.arm.set_position(
                        x=start_x_abs, 
                        y=start_y_abs, 
                        z=self.hor_z_start+self.brush_lift, 
                        roll=None, 
                        pitch=None, 
                        yaw=None, 
                        speed=self.between_speed, 
                        relative=False, 
                        wait=True
                    )
                    # Put the pen down
                    self.arm.set_position(
                        x=None, 
                        y=None, 
                        z=self.hor_z_start, 
                        roll=None, 
                        pitch=None, 
                        yaw=None, 
                        speed=self.between_speed, 
                        relative=False, 
                        wait=True
                    )
                else:
                    # Turn off the light
                    # Do the movement to next path
                    # Turn on the light
                    pass
                
                # Do the movement for the path(s)
                for num, current_path in enumerate(paths):
                    self.arm.move_arc_lines(
                        current_path, 
                        is_radian=False, 
                        times=1, 
                        first_pause_time=0.1, 
                        repeat_pause_time=0, 
                        automatic_calibration=True, 
                        speed=self.painting_speed, 
                        mvacc=1500, 
                        wait=True
                    )
                    # Do a reload between current_path and next one (if more than 1)
                    if num < len(paths)-1:
                        self.reload_brush(current_path[-1][0], current_path[-1][1], True, False)
                
            # Return to initial position
            self.initial_position()


    def reload_brush(self, orig_1, orig_2, low: bool, first:bool):
        """
        Load the brush with more paint from reload pot.

        :param orig_x: The x position to return to after the reload.
        :param orig_y: The y position to return to after the realod.
        :param low: False means stay above the page after, True means lower down so brush is touching the page.
        :param first: Whether or not this is the first brush load.
        """
        print("Reloading brush")
        print("Low after?", low)
        if self.vertical == True:
            if first == False:
                # Go straight up again first if doing vertical painting
                self.straight_fw_position(25)
            # Then go through initial position
            self.initial_position()
            # Move to paint pot
            self.arm.set_position(
                x=self.hor_x_start+self.x_dist_reload, 
                y=self.hor_y_start+self.y_dist_reload, 
                z=self.hor_z_start+self.pot_lift, 
                roll=None, 
                pitch=None, 
                yaw=None, 
                speed=self.between_speed, 
                relative=False, 
                wait=True
            )
            # Put brush down
            self.arm.set_position(
                x=None, 
                y=None, 
                z=self.hor_z_start, 
                roll=None, 
                pitch=None, 
                yaw=None, 
                speed=self.painting_speed, 
                relative=False, 
                wait=True
            ) 
            # Do square in the paint
            self.arm.set_position(x=-15, y=0, z=0, roll=None, pitch=None, yaw=None, speed=self.painting_speed, relative=True, wait=True)
            self.arm.set_position(x=0, y=-15, z=0, roll=None, pitch=None, yaw=None, speed=self.painting_speed, relative=True, wait=True)
            self.arm.set_position(x=15, y=0, z=0, roll=None, pitch=None, yaw=None, speed=self.painting_speed, relative=True, wait=True)
            self.arm.set_position(x=0, y=15, z=0, roll=None, pitch=None, yaw=None, speed=self.painting_speed, relative=True, wait=True)
            # Lift up again
            self.arm.set_position(
                x=None, 
                y=None, 
                z=self.hor_z_start+self.pot_lift, 
                roll=None, 
                pitch=None, 
                yaw=None, 
                speed=self.between_speed, 
                relative=False, 
                wait=True
            )
            # Go through straight up
            self.straight_bw_position(25)
            # Get arm in a good position for painting
            self.vert_initial_position()
            # Move back to prev postion
            self.arm.set_position(
                x=self.ver_x_start+self.brush_lift, 
                y=orig_1,
                z=orig_2,
                roll=self.ver_roll_start, 
                pitch=self.ver_pitch_start, 
                yaw=self.ver_yaw_start, 
                speed=self.between_speed,
                relative=False, 
                wait=False
            )
            if low == True:
                self.arm.set_position(
                    x=self.ver_x_start,
                    y=orig_1,
                    z=orig_2,
                    roll=self.ver_roll_start, 
                    pitch=self.ver_pitch_start, 
                    yaw=self.ver_yaw_start, 
                    speed=self.between_speed,
                    relative=False, 
                    wait=False
                )
        else:
            # Lift up brush
            self.arm.set_position(
                x=None, 
                y=None, 
                z=self.hor_z_start+self.pot_lift, 
                roll=None, 
                pitch=None, 
                yaw=None, 
                speed=self.between_speed, 
                relative=False, 
                wait=True
            )
            # Move to paint pot
            self.arm.set_position(
                x=self.hor_x_start+self.x_dist_reload, 
                y=self.hor_y_start+self.y_dist_reload, 
                z=None, 
                roll=None, 
                pitch=None, 
                yaw=None, 
                speed=self.between_speed, 
                relative=False, 
                wait=True
            )
            # Put brush down
            self.arm.set_position(
                x=None, 
                y=None, 
                z=self.hor_z_start, 
                roll=None, 
                pitch=None, 
                yaw=None, 
                speed=self.painting_speed, 
                relative=False, 
                wait=True
            ) 
            # Do square in the paint
            self.arm.set_position(x=-15, y=0, z=0, roll=None, pitch=None, yaw=None, speed=self.painting_speed, relative=True, wait=True)
            self.arm.set_position(x=0, y=-15, z=0, roll=None, pitch=None, yaw=None, speed=self.painting_speed, relative=True, wait=True)
            self.arm.set_position(x=15, y=0, z=0, roll=None, pitch=None, yaw=None, speed=self.painting_speed, relative=True, wait=True)
            self.arm.set_position(x=0, y=15, z=0, roll=None, pitch=None, yaw=None, speed=self.painting_speed, relative=True, wait=True)
            # Lift up again
            self.arm.set_position(
                x=None, 
                y=None, 
                z=self.hor_z_start+self.pot_lift, 
                roll=None, 
                pitch=None, 
                yaw=None, 
                speed=self.between_speed, 
                relative=False, 
                wait=True
            )
            # Move back to prev postion
            self.arm.set_position(
                x=orig_1,
                y=orig_2,
                z=self.hor_z_start+self.pot_lift, 
                roll=self.hor_roll_start, 
                pitch=self.hor_pitch_start, 
                yaw=self.hor_yaw_start, 
                speed=self.between_speed,
                relative=False, 
                wait=False
            )
            if low == True:
                self.arm.set_position(
                    x=orig_1,
                    y=orig_2,
                    z=self.hor_z_start,
                    roll=self.hor_roll_start, 
                    pitch=self.hor_pitch_start, 
                    yaw=self.hor_yaw_start, 
                    speed=self.between_speed,
                    relative=False, 
                    wait=False
                )

    def wash_brush(self):
        pass

            
