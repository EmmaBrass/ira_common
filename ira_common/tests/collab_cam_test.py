from ira_common.arm_movements import ArmMovements
from ira_common.general_gpt import GPT
from ira_common.camera import Camera
import cv2, os


movements = ArmMovements()
cam = Camera(port_num=4)

# Move arm into position
movements.look_at_canvas()
# Take picture of blank canvas
image = cam.read()

# Save the image!
file_path = os.path.join(os.path.dirname(__file__), "image_4.jpg")
cv2.imwrite(file_path, image)

#movements.initial_position()

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Test done!!!")
