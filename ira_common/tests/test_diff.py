from ira_common.arm_movements import ArmMovements
from ira_common.general_gpt import GPT
import cv2, time, os


movements = ArmMovements()
gpt = GPT(collab=True)
time.sleep(2)

# blank canvas
movements.initial_image = cv2.imread("/home/emma/ira_common/ira_common/tests/blank_image.jpg")

# before human mark
movements.before_image = cv2.imread("/home/emma/ira_common/ira_common/tests/image_1.jpg")

#after human mark
movements.after_image = cv2.imread("/home/emma/ira_common/ira_common/tests/image_2.jpg")

movements.canvas_initialise(debug=False)

# See the diff image 
output_coordinates, color_pot, mark_image = movements.paint_abstract_mark(debug=True)

# Save as a file with a path
dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(dir, "mark_image.jpg")
cv2.imwrite(file_path, mark_image)
print("Sending <comment> command to gpt.")

# test sending the diff image to the GPT
response = gpt.add_user_message_and_get_response_and_speak(f"The command is: <comment>. The file path is: {file_path}")
