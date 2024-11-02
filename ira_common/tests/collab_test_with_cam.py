

# Need a pic of a canvas before and after human marks

# Then run these with the correct methods in arm_movements (with connection to robot)
# to see how well they work

# Bring in some paper and more brushes and some more paint colors tomorrow.

# I think at the moment the canvas initialisation is failing, but maybe that is just because I haven't put a canvas down.

# Start by writing the test and then testing with no canvas.
# Then test with canvas.

# Figure out how to get failues in this lower-down code to show up in the higher code!

from ira_common.arm_movements import ArmMovements
from ira_common.general_gpt import GPT
from ira_common.camera import Camera
import cv2


movements = ArmMovements()
gpt = GPT(collab=True)
cam = Camera(port_num=4)

gpt.add_user_message_and_get_response_and_speak("The command is: <startup_pic>")
# Move arm into position
movements.look_at_canvas()
# Take picture of blank canvas
image = cam.read()
movements.initial_image = image
movements.before_image = image
movements.canvas_initialise(debug=True)
# Move away
movements.lift_up()
# Say it's their turn to paint
gpt.add_user_message_and_get_response_and_speak("The command is: <your_turn>")
# Wait for human input
done = input("Press enter when done with your turn!")
# Look at canvas, take pic, analyse
gpt.add_user_message_and_get_response_and_speak("The command is: <your_turn_pic>")
movements.look_at_canvas()
image = cam.read()
movements.after_image = image
# Do response
movements.initial_position()
output_coordinates, color_pot = movements.paint_abstract_mark(debug=True)   #TODO -> choose only the mark with the LARGEST diff area? // diff on it's own needs more testing.
movements.paint_marks(output_coordinates)

print("Test done!!!")
