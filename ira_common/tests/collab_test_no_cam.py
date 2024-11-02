

# Need a pic of a canvas before and after human marks

# Then run these with the correct methods in arm_movements (with connection to robot)
# to see how well they work

# Bring in some paper and more brushes and some more paint colors tomorrow.

# I think at the moment the canvas initialisation is failing, but maybe that is just because I haven't put a canvas down.

# Start by writing the test and then testing with no canvas.
# Then test with canvas.

# Figure out how to get failues in this lower-down code to show up in the higher code!

from ira_common.arm_movements import ArmMovements
import cv2


movements = ArmMovements()

# blank canvas
movements.initial_image = cv2.imread("/home/emma/Downloads/test1.jpg")

# before human mark
movements.before_image = cv2.imread("/home/emma/Downloads/test2.jpg")

#after human mark
movements.after_image = cv2.imread("/home/emma/Downloads/test3.jpg")

movements.canvas_initialise(debug=False)

output_coordinates, color_pot = movements.paint_abstract_mark()

movements.paint_marks(output_coordinates)
