import cv2
import math, random
import ira_common.configuration as config
from ira_common.canvas import Canvas
from ira_common.all_marks import AllMarks
from ira_common.mark_creator import MarkCreator

class CanvasAnalysis():

    def __init__(self):
        self.canvas = None
        # initialise images for ira_collab
        self._initial_image = None
        self._before_image = None
        self._after_image = None

        #values for remembering previous mark types
        self.type_id = {'blob': None, 'straight': None, 'curve': None}

    def canvas_initialise(self, debug=False):
        """ 
        Use with ira_collab.
        Initialise the blank canvas.
        """
        if self._initial_image is not None:
            # initialise the canvas object
            self.canvas = Canvas(debug=debug)
            self.canvas.set_image(self._initial_image)
            self.canvas.set_real_dimensions(
                config.CANVAS_WIDTH,
                config.CANVAS_HEIGHT
            )
            success = self.canvas.analyse()
            return success
        else:
            print("ERROR! No initial image.")
            return False
        
    def paint_abstract_mark(self, debug=False):
        """
        Use with ira_collab.
        The main meat of the system: this method takes the before and after
        images, find the difference, then chooses how to react to the mark,
        makes the path for the robot, and outputs it.

        :param before_image: Image of the canvas before the human mark.
        :param after_image: Image of the canvas after the human mark.
        """

        if self.canvas is None:
            print("ERROR! No canvas object.")

        if self._before_image is not None:
            before_trans = cv2.warpPerspective(
                self._before_image,
                self.canvas.transform_matrix,
                (self.canvas.transformed_image_x, self.canvas.transformed_image_y)
            )
            # diff will be done with before and after image
        else: # before image = initial image for the very first human mark
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
        all_marks = AllMarks(self.canvas, debug)
        all_marks.set_old_image(before_trans)
        all_marks.set_new_image(after_trans)
        # Find all marks, run mark type analysis, color analysis, skeletonisation, etc.
        masked_image = all_marks.find_all_marks()
        # Get array with all the new marks in it
        marks_array = all_marks.get_all_marks()

        # If more than 1 mark was made by the human, randomly choose a 
        # set number of marks to respond to, otherwise keep the original marks array.
        final_marks_array = []
        if len(marks_array) > config.NUM_MARKS:
            final_marks_array = random.sample(marks_array, config.NUM_MARKS) # Only respond to this many marks #TODO respond to the largest area mark only?
        else:
            final_marks_array = marks_array
 
        for num, mark in enumerate(final_marks_array):

            #Create a mark based on the user's mark - makes an .svg file of the next mark for the robot to make
            if mark.type == "blob":
                mark_creator = MarkCreator(
                    mark, 
                    self.canvas, 
                    config.COLORS, 
                    prev_id = self.type_id['blob'],
                    debug=False
                    )
            elif mark.type == "straight":
                mark_creator = MarkCreator(
                    mark, 
                    self.canvas, 
                    config.COLORS, 
                    prev_id = self.type_id['straight'],
                    debug=False
                    )
            elif mark.type == "curve":
                mark_creator = MarkCreator(
                    mark, 
                    self.canvas, 
                    config.COLORS, 
                    prev_id = self.type_id['curve'],
                    debug=False
                    )
            output_array = mark_creator.create() 
            color_pot = mark_creator.choose_color()
            self.type_id[mark.type] = mark_creator.mark_type_id()
            print("type_id dictionary is now: ", self.type_id)

            print("output_array: ", output_array)
            print("color_pot: "), color_pot

            return output_array, color_pot, masked_image