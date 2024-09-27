
# Take an image and get the outline for drawing as a .svg file.
# use xDOG to get outline
# and then use Potrace to convert to .svg

import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import face_recognition
from scipy.interpolate import splprep, splev
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from skimage.morphology import skeletonize
from scipy.spatial import distance

from ultralytics import YOLO
import cv2
import numpy as np

# TODO what do I want to get from here?
# Need to look into arm commands more
# A series of straight points along the line that 
# the robot can move between in linear mode,
# to make what appears to be a straight line?

class Outline():

    def __init__(self) -> None:
        # Ensure the 'images' directory exists one level up
        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        self.image_dir = os.path.join(parent_dir, "images")
        if not os.path.exists(self.image_dir):
            print(f"Directory {self.image_dir} does not exist. Creating it.")
            try:
                os.makedirs(self.image_dir)
            except Exception as e:
                print(f"Failed to create directory {self.image_dir}: {str(e)}")

        self.beard_model_path = '/home/emma/ira_ws/src/ira/resource/best_hair_117_epoch_v4.pt'
        if not os.path.exists(self.beard_model_path):
            print(f"Beard model path does not exist: {self.beard_model_path}")


    def draw_smooth_curve(self, image, points, closed=False):
        """
        Draw a smooth curve on the image based on given points using Catmull-Rom splines.

        :param image: image to draw the curves on
        :param points: points to draw
        :param closed: whether or not to make a closed shape from the points
        """
        # Separate points into two lists
        x, y = zip(*points)
        x = np.array(x)
        y = np.array(y)

        # Add first x and y value to the end of the list if a closed shape is desired
        if closed == True:
            x = np.append(x, x[0])
            y = np.append(y, y[0])

        # Fit spline to points
        tck, u = splprep([x, y], s=0)
        
        # Evaluate spline over a fine grid (increase 1000 for smoother curve)
        new_points = splev(np.linspace(0, 1, 1000), tck)
        
        # Draw the interpolated curve on the image
        for i in range(len(new_points[0]) - 1):
            cv2.line(
                image, 
                (int(new_points[0][i]), int(new_points[1][i])), 
                (int(new_points[0][i+1]), int(new_points[1][i+1])), 
                color=(255,255,255), 
                thickness=1
            )

    def resize_and_pad(self, image:np.ndarray, height:int = 256, width:int = 256):
        """
        Input preprocessing function, takes input image in np.ndarray format, 
        resizes it to fit specified height and width with preserving aspect ratio 
        and adds padding on top or right side to complete target height x width rectangle.
        
        :param image: input image in np.ndarray format
        :param height (int, *optional*, 256): target height
        :param width (int, *optional*, 256): target width

        :returns padded_img (np.ndarray): processed image
        :returns padding_info (Tuple[int, int]): information about padding size, 
        for postprocessing
        """
        h, w = image.shape[:2]
        if h < w:
            img = cv2.resize(image, (width, np.floor(h / (w / width)).astype(int)))
        else:
            img = cv2.resize(image, (np.floor(w / (h / height)).astype(int), height))
        
        r_h, r_w = img.shape[:2]
        right_padding = width - r_w
        top_padding = height - r_h
        padded_img = cv2.copyMakeBorder(img, top_padding, 0, 0, right_padding, cv2.BORDER_CONSTANT)
        return padded_img, [top_padding, right_padding]
    
    def find_contours_coordinates(self, input_image, testing=False):
        """
        Finds canny edges image with white lines on black background.
        Finds contours using opencv function.
        Returns a list of coordinate arrays, along with image dimensions.
        These can then be turned into paths for robot motion.
        """

        with_beard = self.get_beard_outline(input_image)
        resized_with_beard, padding = self.resize_and_pad(with_beard, 1028, 1028)

        image = cv2.GaussianBlur(input_image, (3, 3), 0)

        # Get Canny edge detection image
        canny_image = self.no_background_canny(image)
        cv2.imwrite(os.path.join(self.image_dir, "CANNY.png"), canny_image)
        use_canny_image = canny_image.copy()
        # Add in facial features from segmentation model
        resized_image, padding = self.resize_and_pad(image, 1028, 1028)
        features_image = self.find_facial_features(resized_image, canny_image)
        # Combine canny with facial features
        pre_final_image = cv2.bitwise_or(use_canny_image, features_image)
        # Combine with beard
        final_image = cv2.bitwise_or(pre_final_image, resized_with_beard)
        # Save the image to the specified path
        cv2.imwrite(os.path.join(self.image_dir, "with_features.png"), final_image)

        # If testing = True, then use image called TEST in the images folder
        if testing == True:
            final_image = cv2.imread('images/TEST.png', cv2.IMREAD_GRAYSCALE)
            final_image, padding = self.resize_and_pad(canny_image, 1028, 1028)

        # Threshold the image to ensure it is binary
        _, binary_image = cv2.threshold(final_image, 127, 255, cv2.THRESH_BINARY)

        # Remove the padding
        # Determine the new dimensions
        height, width = binary_image.shape[:2]  # Get the original dimensions
        new_height = height - padding[0]
        new_width = width - padding[1]

        # Ensure the new dimensions are valid
        if new_height <= 0 or new_width <= 0:
            raise ValueError("The amount to remove is too large, resulting in a non-positive dimension.")

        # Crop the image using array slicing
        cropped_image = binary_image[padding[0]:height, 0:new_width]

        # Skeletonize the image
        skeleton = skeletonize(cropped_image)
        # Convert the skeletonized image to uint8 (0 or 255)
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)
        # Save the image using OpenCV
        cv2.imwrite(os.path.join(self.image_dir, "skeleton.png"), skeleton_uint8)

        # Get the coordinates of the path
        path_coordinates = np.column_stack(np.where(skeleton > 0)) # in format (y,x)

        # Sort path cooridnates
        sorted_paths = []
        current_path = [path_coordinates[0]]
        path_coordinates = np.delete(path_coordinates, 0, 0)
        distance_threshold = 5

        while len(path_coordinates) > 0:
            last_point = current_path[-1]
            distances = distance.cdist([last_point], path_coordinates)
            nearest_index = np.argmin(distances)
            nearest_distance = distances[0, nearest_index]

            if nearest_distance > distance_threshold:
                # If the nearest point is too far, start a new path
                sorted_paths.append(np.array(current_path))
                current_path = [path_coordinates[nearest_index]]
            else:
                # Otherwise, continue adding to the current path
                current_path.append(path_coordinates[nearest_index])
        
            path_coordinates = np.delete(path_coordinates, nearest_index, 0)

        # Append the last path if it's not empty
        if current_path:
            sorted_paths.append(np.array(current_path))

        print("sorted_paths", sorted_paths)

        # Flip the (y,x) values to be (x,y)
        flipped_paths = []
        for path in sorted_paths:
            # Flip each coordinate from (y, x) to (x, y)
            flipped_path = [(x, y) for (y, x) in path]
            flipped_paths.append(np.array(flipped_path))

        # Create a black background image
        paths_image = np.zeros(cropped_image.shape, dtype=np.uint8)
        
        for path in flipped_paths:
            # Draw each path on the image
            for i in range(len(path) - 1):
                # Get start and end points
                start_point = tuple(path[i])
                end_point = tuple(path[i + 1])
                
                # Draw a line between consecutive points
                cv2.line(paths_image, start_point, end_point, color=255, thickness=1) # assumes format (x,y)
        
        # Display the result using OpenCV
        cv2.imshow('paths_image', paths_image)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()

        #  Save the image to the specified path
        cv2.imwrite(os.path.join(self.image_dir, "paths_visualisation.png"), paths_image)

        return flipped_paths, cropped_image.shape[1], cropped_image.shape[0] # image points, image x dimension, image y dimension
    
    def get_beard_outline(self, image):
        """
        Get white outline line for beard, if there is one.
        """

        # Load your custom YOLOv8 model
        model = YOLO(self.beard_model_path)

        if image is None:
            print(f"Error: Unable to open image.")
            exit()

        # Create a black background image of the same size as the original image
        black_background = np.zeros_like(image) 

        # Perform inference
        results = model(image)

        # Process the results
        for result in results:
            # Print out the detected objects and their confidence scores
            print(f"Detected {len(result.boxes)} objects")

            print("Result: ", result)

            # Check if masks are available
            if result.masks:
                for i, mask in enumerate(result.masks.data):
                    class_id = int(result.boxes.cls[i].item())  # Ensure the class ID is an integer
                    class_name = result.names[class_id]

                    # Only process masks for the 'beard' class
                    if class_name == 'beard':
                        # Convert mask to a NumPy array if it's not already
                        mask = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask

                        # Resize mask to fit the original image dimensions
                        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

                        # Convert mask to binary
                        _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)

                        # Convert binary mask to uint8 type
                        binary_mask = (binary_mask * 255).astype(np.uint8)

                        # Find contours from the binary mask
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        # Draw contours on the original image with a white color and a thickness of 1 pixel
                        cv2.drawContours(black_background, contours, -1, (255, 255, 255), 1)

        # Convert the image to grayscale (black and white)
        gray_image = cv2.cvtColor(black_background, cv2.COLOR_BGR2GRAY)

        # Save or display the labeled image
        cv2.imwrite(os.path.join(self.image_dir, "beard_outline.png"), gray_image)

        return gray_image
    
    def no_background_canny(self, image):
        """
        Removes background using segmentation model.
        Fills in details using Canny edge dection.
        Adds in jaw line if no beard.
        """
        # Height and width that will be used by the model
        DESIRED_HEIGHT = 256
        DESIRED_WIDTH = 256
        # Perform image resizing with padding
        resized_image, _ = self.resize_and_pad(image, DESIRED_HEIGHT, DESIRED_WIDTH)
        cv2.imwrite(os.path.join(self.image_dir, "resized_image.png"), resized_image) 
        
        small_face_mask = self.find_face_mask(resized_image)
        large_resized_face_mask, _ = self.resize_and_pad(small_face_mask, 1028, 1028)

        small_clothes_mask = self.find_clothes_mask(resized_image)
        large_resized_clothes_mask, _ = self.resize_and_pad(small_clothes_mask, 1028, 1028)

        small_hair_mask = self.find_hair_mask(resized_image)
        large_resized_hair_mask, _ = self.resize_and_pad(small_hair_mask, 1028, 1028)

        small_body_mask = self.find_body_mask(resized_image)
        large_resized_body_mask, _ = self.resize_and_pad(small_body_mask, 1028, 1028)

        small_others_mask = self.find_others_mask(resized_image)
        large_resized_others_mask, _ = self.resize_and_pad(small_others_mask, 1028, 1028)

        small_no_background_mask = self.find_no_background_mask(resized_image)
        large_resized_no_background_mask, _ = self.resize_and_pad(small_no_background_mask, 1028, 1028)

        # small_not_face_mask = self.find_not_face_mask(resized_image)
        # large_resized_not_face_mask, _ = self.resize_and_pad(small_not_face_mask, 1028, 1028)

        large_resized_original, _ = self.resize_and_pad(image, 1028, 1028)

        image_path = os.path.join(self.image_dir, "large_resized_face_mask.png")
        cv2.imwrite(image_path, large_resized_face_mask)
        image_path = os.path.join(self.image_dir, "large_resized_no_background_mask.png")
        cv2.imwrite(image_path, large_resized_no_background_mask)  
        image_path = os.path.join(self.image_dir, "large_resized_original.png")
        cv2.imwrite(image_path, large_resized_original) 

        # ##### GET MASK FOR NO BACKGROUND #####

        # # Turn no background mask to greyscale
        # if len(large_resized_no_background_mask.shape) == 3:
        #     gray_no_background_mask = cv2.cvtColor(large_resized_no_background_mask, cv2.COLOR_BGR2GRAY)
        #     print("here")
        # else:
        #     gray_no_background_mask = large_resized_no_background_mask

        # # Ensure the mask is a binary mask with values 0 and 255
        # _, binary_mask = cv2.threshold(gray_no_background_mask, 127, 255, cv2.THRESH_BINARY)
        # image_path = os.path.join(self.image_dir, "binary_mask.png")
        # cv2.imwrite(image_path, binary_mask) 

        # # Apply Gaussian blur to smooth the edges
        # blurred_mask = cv2.GaussianBlur(binary_mask, (15, 15), 0)
        # image_path = os.path.join(self.image_dir, "blurred_mask.png")
        # cv2.imwrite(image_path, blurred_mask) 

        # # Re-apply thresholding to keep the mask binary
        # _, smoothed_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

        # # Apply morphological closing to further refine the edges
        # kernel = np.ones((5, 5), np.uint8)
        # closed_mask = cv2.morphologyEx(smoothed_mask, cv2.MORPH_CLOSE, kernel)
        # cv2.imwrite(os.path.join(self.image_dir, "closed_mask.png"), closed_mask) 

        # # Turn original image to greyscale
        # if len(large_resized_original.shape) == 3:
        #     gray = cv2.cvtColor(large_resized_original, cv2.COLOR_BGR2GRAY)
        #     print("here")
        # else:
        #     gray = large_resized_original

        # # Apply the mask to the image
        # no_background = cv2.bitwise_and(gray, closed_mask)
        # cv2.imwrite(os.path.join(self.image_dir, "no_background.png"), no_background) 

        ##### GET MASK FOR FACE ONLY #####

        # Turn face mask to greyscale
        if len(large_resized_face_mask.shape) == 3:
            gray_face_mask = cv2.cvtColor(large_resized_face_mask, cv2.COLOR_BGR2GRAY)
            print("here")
        else:
            gray_face_mask = large_resized_face_mask

        # Ensure face mask is binary
        face_mask = cv2.threshold(gray_face_mask, 127, 255, cv2.THRESH_BINARY)[1]

        ##### GET MASK FOR CLOTHES ONLY #####

        # Turn clothes mask to greyscale
        if len(large_resized_clothes_mask.shape) == 3:
            gray_clothes_mask = cv2.cvtColor(large_resized_clothes_mask, cv2.COLOR_BGR2GRAY)
            print("here")
        else:
            gray_clothes_mask = large_resized_clothes_mask

        # Ensure clothes mask is binary
        clothes_mask = cv2.threshold(gray_clothes_mask, 127, 255, cv2.THRESH_BINARY)[1]

        ##### GET MASK FOR HAIR ONLY #####

        # Turn hair mask to greyscale
        if len(large_resized_hair_mask.shape) == 3:
            gray_hair_mask = cv2.cvtColor(large_resized_hair_mask, cv2.COLOR_BGR2GRAY)
            print("here")
        else:
            gray_hair_mask = large_resized_hair_mask

        # Ensure hair mask is binary
        hair_mask = cv2.threshold(gray_hair_mask, 127, 255, cv2.THRESH_BINARY)[1]

        ##### GET MASK FOR BODY ONLY #####

        # Turn body mask to greyscale
        if len(large_resized_body_mask.shape) == 3:
            gray_body_mask = cv2.cvtColor(large_resized_body_mask, cv2.COLOR_BGR2GRAY)
            print("here")
        else:
            gray_body_mask = large_resized_body_mask

        # Ensure body mask is binary
        body_mask = cv2.threshold(gray_body_mask, 127, 255, cv2.THRESH_BINARY)[1]

        ##### GET MASK FOR OTHERS ONLY #####

        # Turn others mask to greyscale
        if len(large_resized_others_mask.shape) == 3:
            gray_others_mask = cv2.cvtColor(large_resized_others_mask, cv2.COLOR_BGR2GRAY)
            print("here")
        else:
            gray_others_mask = large_resized_others_mask

        # Ensure others mask is binary
        others_mask = cv2.threshold(gray_others_mask, 127, 255, cv2.THRESH_BINARY)[1]

        # print("face_mask shape:", face_mask.shape)
        # print("gray shape:", gray.shape)

        # # Extract the pixel values where the mask is white
        # face_masked_pixels = gray[face_mask == 255]

        # Apply a blur of (9,9) only to the not_face parts of the image!
        # Turn not_face mask to greyscale
        # if len(large_resized_not_face_mask.shape) == 3:
        #     gray_not_face_mask = cv2.cvtColor(large_resized_not_face_mask, cv2.COLOR_BGR2GRAY)
        #     print("here")
        # else:
        #     gray_not_face_mask = large_resized_not_face_mask

        # # Ensure the mask is a binary mask with values 0 and 255
        # _, binary_not_face_mask = cv2.threshold(gray_not_face_mask, 127, 255, cv2.THRESH_BINARY)
        # cv2.imwrite(os.path.join(self.image_dir, "not_face_mask.png"), binary_not_face_mask) 

        # inverse_binary_not_face_mask = cv2.bitwise_not(binary_not_face_mask)

        # # Apply a blur to the whole (no background) image
        # blurred = cv2.GaussianBlur(no_background, (5,5), 0)

        # # Apply extreme Gaussian blur to the entire image
        # very_blurred = cv2.GaussianBlur(no_background, (13, 13), 0)

        # # Combine the blurred image and the original image using the masks
        # blurred_part = cv2.bitwise_and(very_blurred, binary_not_face_mask) # everything but the face (hair, clothes) very blurred
        # original_part = cv2.bitwise_and(blurred, inverse_binary_not_face_mask) # face a bit blurred
        # result = cv2.add(blurred_part, original_part)
        # cv2.imwrite(os.path.join(self.image_dir, "blurred_not_face.png"), blurred_part) 
        # cv2.imwrite(os.path.join(self.image_dir, "is_face.png"), original_part) 
        # cv2.imwrite(os.path.join(self.image_dir, "combined_blurred.png"), result) 

        # Setting canny edge detection parameter values based on the face mask pixels
        sigma = 0.45
        t_lower = 0 #int(max(0, (1.0-sigma)*v))
        t_upper = 40 #int(min(255, (sigma*v))) #TODO needs adjusting?

        # result_blurred_no_face = cv2.Canny(blurred_part, t_lower, t_upper)
        # result_blurred = cv2.Canny(result, t_lower, t_upper)
        # result_no_blur = cv2.Canny(no_background, t_lower, t_upper)

        canny_face_mask = cv2.Canny(face_mask, t_lower, t_upper)
        canny_body = cv2.Canny(body_mask, t_lower, t_upper)
        canny_clothes_mask = cv2.Canny(clothes_mask, t_lower, t_upper)
        canny_hair_mask = cv2.Canny(hair_mask, t_lower, t_upper)
        canny_others_mask = cv2.Canny(others_mask, t_lower, t_upper)
        combined_1 = cv2.add(canny_body, canny_face_mask)
        combined_2 = cv2.add(combined_1, canny_clothes_mask)
        combined_3 = cv2.add(combined_2, canny_others_mask)
        combined = cv2.add(combined_3, canny_hair_mask)

        cv2.imwrite(os.path.join(self.image_dir, "canny_face_mask.png"), canny_face_mask) 
        # cv2.imwrite(os.path.join(self.image_dir, "canny_image_no_blur.png"), result_no_blur) 
        # cv2.imwrite(os.path.join(self.image_dir, "canny_image_blurred.png"), result_blurred) 

        return combined # previously used result_blurred
    
    def find_mask_general(
        self, 
        image, 
        bg_color, 
        hair_color,
        body_skin_color,
        face_skin_color,
        clothes_color,
        others_color
    ):

        BaseOptions = mp.tasks.BaseOptions
        ImageSegmenter = mp.tasks.vision.ImageSegmenter
        ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Colors
        BG_COLOR = bg_color
        HAIR_COLOR = hair_color
        BODY_SKIN_COLOR = body_skin_color
        FACE_SKIN_COLOR = face_skin_color
        CLOTHES_COLOR = clothes_color
        OTHERS_COLOR = others_color

        # Create the options that will be used for ImageSegmenter
        model_path = "/home/emma/models/selfie_multiclass_256x256.tflite"

        base_options = BaseOptions(model_asset_path=model_path)
        options = ImageSegmenterOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.IMAGE,
            output_confidence_masks=False,
            output_category_mask=True
        )

        # Create the image segmenter
        with ImageSegmenter.create_from_options(options) as segmenter:

            # Create the MediaPipe image file that will be segmented
            #image = mp.Image.create_from_file(resized_image)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            # Retrieve the masks for the segmented image
            segmentation_result = segmenter.segment(image)
            print(segmentation_result)
            category_mask = segmentation_result.category_mask

            print(category_mask)
            print(type(category_mask))

            # Generate solid color images for showing the output segmentation mask.
            image_data = image.numpy_view()
            bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            hair_image = np.zeros(image_data.shape, dtype=np.uint8)
            hair_image[:] = HAIR_COLOR
            body_image = np.zeros(image_data.shape, dtype=np.uint8)
            body_image[:] = BODY_SKIN_COLOR
            face_image = np.zeros(image_data.shape, dtype=np.uint8)
            face_image[:] = FACE_SKIN_COLOR
            clothes_image = np.zeros(image_data.shape, dtype=np.uint8)
            clothes_image[:] = CLOTHES_COLOR
            others_image = np.zeros(image_data.shape, dtype=np.uint8)
            others_image[:] = OTHERS_COLOR

            output_image = np.zeros(image_data.shape, dtype=np.uint8)

            images = [bg_image, hair_image, body_image, face_image, clothes_image, others_image]

            # Iterate over each category and apply the mask to the output image
            for category in range(6):
                # Create a binary mask for the current category
                condition = (category_mask.numpy_view() == category)
                # Apply the mask to the output image
                output_image = np.where(condition[..., None], images[category], output_image)

        return output_image 

    def find_body_mask(self, image):
        """
        Segmentation of image to return a mask of body.
        """

        output_image = self.find_mask_general(
            image,
            bg_color = (0, 0, 0),
            hair_color = (0, 0, 0),
            body_skin_color = (255, 255, 255),
            face_skin_color = (0, 0, 0),
            clothes_color = (0, 0, 0),
            others_color = (0, 0, 0)
        )

        cv2.imwrite(os.path.join(self.image_dir, "body_mask.png"), output_image) 

        return output_image 
    
    def find_others_mask(self, image):
        """
        Segmentation of image to return a mask of others.
        """

        output_image = self.find_mask_general(
            image,
            bg_color = (0, 0, 0),
            hair_color = (0, 0, 0),
            body_skin_color = (0, 0, 0),
            face_skin_color = (0, 0, 0),
            clothes_color = (0, 0, 0),
            others_color = (255, 255, 255)
        )

        cv2.imwrite(os.path.join(self.image_dir, "others_mask.png"), output_image) 

        return output_image 

    
    def find_hair_mask(self, image):
        """
        Segmentation of image to return a mask of hair.
        """

        output_image = self.find_mask_general(
            image,
            bg_color = (0, 0, 0),
            hair_color = (255, 255, 255),
            body_skin_color = (0, 0, 0),
            face_skin_color = (0, 0, 0),
            clothes_color = (0, 0, 0),
            others_color = (0, 0, 0)
        )

        cv2.imwrite(os.path.join(self.image_dir, "hair_mask.png"), output_image) 

        return output_image 


    def find_clothes_mask(self, image):
        """
        Segmentation of image to return a mask of not face (and not background).
        """

        output_image = self.find_mask_general(
            image,
            bg_color = (0, 0, 0),
            hair_color = (0, 0, 0),
            body_skin_color = (0, 0, 0),
            face_skin_color = (0, 0, 0),
            clothes_color = (255, 255, 255),
            others_color = (0, 0, 0)
        )

        cv2.imwrite(os.path.join(self.image_dir, "clothes_mask.png"), output_image) 

        return output_image 

    def find_not_face_mask(self, image):
        """
        Segmentation of image to return a mask of not face (and not background).
        """

        output_image = self.find_mask_general(
            image,
            bg_color = (0, 0, 0),
            hair_color = (255, 255, 255),
            body_skin_color = (255, 255, 255),
            face_skin_color = (0, 0, 0),
            clothes_color = (255, 255, 255),
            others_color = (0, 0, 0)
        )

        cv2.imwrite(os.path.join(self.image_dir, "not_face_mask.png"), output_image) 

        return output_image 
        

    def find_no_background_mask(self,image):
        """
        Segmentation of image to return a mask of everything 
        but the background.
        """

        output_image = self.find_mask_general(
            image,
            bg_color = (0, 0, 0),
            hair_color = (255, 255, 255),
            body_skin_color = (255, 255, 255),
            face_skin_color = (255, 255, 255),
            clothes_color = (255, 255, 255),
            others_color = (255, 255, 255)
        )

        cv2.imwrite(os.path.join(self.image_dir, "no_background_mask.png"), output_image) 

        return output_image 

    def find_face_mask(self, image):
        """
        Segmentation of the image and then returns a mask consisting of the 
        face skin only, in white, on a black background.
        """

        output_image = self.find_mask_general(
            image,
            bg_color = (0, 0, 0),
            hair_color = (0, 0, 0),
            body_skin_color = (0, 0, 0),
            face_skin_color = (255, 255, 255),
            clothes_color = (0, 0, 0),
            others_color = (0, 0, 0)
        )

        cv2.imwrite(os.path.join(self.image_dir, "face_mask.png"), output_image) 

        return output_image 

    def find_facial_features(self, image, output_image):
        """
        Using facial landmark detection from MediaPipe for outline of 
        key facial features: eyebrows, eyes, nose, mouth.
        """

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh

        # For static image:
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        jaw_idx = [172, 136, 150, 149, 176, 148, 152, 
        377, 400, 378, 379, 365, 397]
        # Full list of jaw points: [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 
        # 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
        left_eye_idx = [133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 
        163, 144, 145, 153, 154, 155]
        right_eye_idx = [362, 398, 384, 385, 386, 387, 388, 466, 263, 
        249, 390, 373, 374, 380, 381, 382]
        left_eyebrow_idx = [107, 66, 105, 63, 70, 156, 46, 53, 52, 65, 55]
        right_eyebrow_idx = [336, 296, 334, 293, 300, 383, 276, 283, 282, 295, 285]
        nose_idx = [49, 64, 240, 75, 79, 237, 141, 94, 370, 
        457, 309, 305, 460, 294, 279]
        # full list of nose points: [209, 49, 64, 240, 75, 79, 237, 141, 94, 370, 
        #457, 309, 305, 460, 294, 279, 429]
        top_lip_idx = [13, 312, 311, 310, 415, 306, 291, 409, 270, 269, 267, 
        0, 37, 39, 40, 185, 61, 78, 191, 80, 81, 82]
        bottom_lip_idx = [14, 317, 402, 318, 324, 306, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 78, 95, 88, 178, 87]
        left_iris_idx = [470, 469, 472,471]
        right_iris_idx = [475, 474, 477, 476]
        nose_bridge_left_idx = [245, 188,174, 236]
        nose_bridge_right_idx = [465, 412, 399, 456]
        image_hight, image_width, _ = image.shape
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            ) as face_mesh:
            results = face_mesh.process(image)
            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                print("Error!")
            for face_landmarks in results.multi_face_landmarks: # https://stackoverflow.com/questions/67141844/python-how-to-get-face-mesh-landmarks-coordinates-in-mediapipe
                jaw = []
                left_eye = []
                right_eye = []
                left_iris = []
                right_iris = []
                left_eyebrow = []
                right_eyebrow = []
                nose = []
                top_lip = []
                bottom_lip = []
                nose_bridge_left = []
                nose_bridge_right = []
                for i in jaw_idx:
                    x_coodinate = int(face_landmarks.landmark[i].x * image_width)
                    y_coodinate = int(face_landmarks.landmark[i].y * image_hight)
                    jaw.append([x_coodinate,y_coodinate])
                    #cv2.circle(output_image, (x_coodinate, y_coodinate), 2, (0, 0, 255), -1)
                for i in left_eye_idx:
                    x_coodinate = int(face_landmarks.landmark[i].x * image_width)
                    y_coodinate = int(face_landmarks.landmark[i].y * image_hight)
                    left_eye.append([x_coodinate,y_coodinate])
                    #cv2.circle(output_image, (x_coodinate, y_coodinate), 2, (0, 0, 255), -1)
                for i in right_eye_idx:
                    x_coodinate = int(face_landmarks.landmark[i].x * image_width)
                    y_coodinate = int(face_landmarks.landmark[i].y * image_hight)
                    right_eye.append([x_coodinate,y_coodinate])
                for i in left_iris_idx:
                    x_coodinate = int(face_landmarks.landmark[i].x * image_width)
                    y_coodinate = int(face_landmarks.landmark[i].y * image_hight)
                    left_iris.append([x_coodinate,y_coodinate])
                    #cv2.circle(output_image, (x_coodinate, y_coodinate), 2, (0, 0, 255), -1)
                for i in right_iris_idx:
                    x_coodinate = int(face_landmarks.landmark[i].x * image_width)
                    y_coodinate = int(face_landmarks.landmark[i].y * image_hight)
                    right_iris.append([x_coodinate,y_coodinate])
                    #cv2.circle(output_image, (x_coodinate, y_coodinate), 2, (0, 0, 255), -1)
                for i in left_eyebrow_idx:
                    x_coodinate = int(face_landmarks.landmark[i].x * image_width)
                    y_coodinate = int(face_landmarks.landmark[i].y * image_hight)
                    left_eyebrow.append([x_coodinate,y_coodinate])
                    #cv2.circle(output_image, (x_coodinate, y_coodinate), 2, (0, 0, 255), -1)
                for i in right_eyebrow_idx:
                    x_coodinate = int(face_landmarks.landmark[i].x * image_width)
                    y_coodinate = int(face_landmarks.landmark[i].y * image_hight)
                    right_eyebrow.append([x_coodinate,y_coodinate])
                    #cv2.circle(output_image, (x_coodinate, y_coodinate), 2, (0, 0, 255), -1)
                for i in nose_idx:
                    x_coodinate = int(face_landmarks.landmark[i].x * image_width)
                    y_coodinate = int(face_landmarks.landmark[i].y * image_hight)
                    nose.append([x_coodinate,y_coodinate])
                    #cv2.circle(output_image, (x_coodinate, y_coodinate), 2, (0, 0, 255), -1)
                for i in top_lip_idx:
                    x_coodinate = int(face_landmarks.landmark[i].x * image_width)
                    y_coodinate = int(face_landmarks.landmark[i].y * image_hight)
                    top_lip.append([x_coodinate,y_coodinate])
                    #cv2.circle(output_image, (x_coodinate, y_coodinate), 2, (0, 0, 255), -1)
                for i in bottom_lip_idx:
                    x_coodinate = int(face_landmarks.landmark[i].x * image_width)
                    y_coodinate = int(face_landmarks.landmark[i].y * image_hight)
                    bottom_lip.append([x_coodinate,y_coodinate])
                    #cv2.circle(output_image, (x_coodinate, y_coodinate), 2, (0, 0, 255), -1)
                for i in nose_bridge_left_idx:
                    x_coodinate = int(face_landmarks.landmark[i].x * image_width)
                    y_coodinate = int(face_landmarks.landmark[i].y * image_hight)
                    nose_bridge_left.append([x_coodinate,y_coodinate])
                    #cv2.circle(output_image, (x_coodinate, y_coodinate), 2, (0, 0, 255), -1)
                for i in nose_bridge_right_idx:
                    x_coodinate = int(face_landmarks.landmark[i].x * image_width)
                    y_coodinate = int(face_landmarks.landmark[i].y * image_hight)
                    nose_bridge_right.append([x_coodinate,y_coodinate])
                    #cv2.circle(output_image, (x_coodinate, y_coodinate), 2, (0, 0, 255), -1)
                #self.draw_smooth_curve(output_image, jaw, closed=False)

                # Draw eyes and irises and then remove any iris parts outside the eyes
                eye_image = self.draw_eyes(output_image, left_eye, right_eye, left_iris, right_iris)

                # Calculate side for nose bridge and put it in.
                nose_side = self.choose_nose_bridge(image, nose_bridge_left, nose_bridge_right)
                if nose_side == 'right':
                    self.draw_smooth_curve(eye_image, nose_bridge_right)
                if nose_side == 'left':
                    self.draw_smooth_curve(eye_image, nose_bridge_left)

                self.draw_smooth_curve(eye_image, left_eyebrow, closed=True)
                self.draw_smooth_curve(eye_image, right_eyebrow, closed=True)
                self.draw_smooth_curve(eye_image, nose, closed=False)
                self.draw_smooth_curve(eye_image, top_lip, closed=True)
                self.draw_smooth_curve(eye_image, bottom_lip, closed=True)
                
            # Display Image
            cv2.imwrite(os.path.join(self.image_dir, "annotated.png"), eye_image) 

        return eye_image
    
    def choose_nose_bridge(self, image, nose_bridge_left, nose_bridge_right):
        """
        Decide which side to put in a nose bridge based on pixel darkness.
        """

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # left_intensities = []
        # for [x,y] in nose_bridge_left:
        #     left_intensities.append(gray_image[y, x])
        # mean_left = np.mean(left_intensities)
    
        # right_intensities = []
        # for [x,y] in nose_bridge_right:
        #     right_intensities.append(gray_image[y, x])
        # mean_right = np.mean(right_intensities)

        # Function to get the standard deviation of pixel intensity in a region of interest around the points
        roi_size = 10
        half_size = roi_size // 2

        left_intensities = []
        for [x,y] in nose_bridge_left:
            roi = gray_image[max(0, y-half_size):min(image.shape[0], y+half_size+1),
                            max(0, x-half_size):min(image.shape[1], x+half_size+1)]
            left_intensities.extend(roi.flatten())
        left_stdev = np.std(left_intensities)

        right_intensities = []
        for [x,y] in nose_bridge_right:
            roi = gray_image[max(0, y-half_size):min(image.shape[0], y+half_size+1),
                            max(0, x-half_size):min(image.shape[1], x+half_size+1)]
            right_intensities.extend(roi.flatten())
        right_stdev = np.std(right_intensities)

        # Choose the side with the darker pixels (lower average intensity)
        if right_stdev < left_stdev:
            side = 'left'
        else:
            side = 'right'

        # Output the result
        print(f"The side of the nose bridge is: {side}")

        return side
    
    def draw_eyes(self, output_image, left_eye, right_eye, left_iris, right_iris):
        """ 
        Draw on the eyes and irises and remove any white bits outside the eyes.
        """
        
        # Do irises

        self.draw_smooth_curve(output_image, left_iris, closed=True)
        self.draw_smooth_curve(output_image, right_iris, closed=True)

        ##### LEFT EYE #####

        # Separate points into two lists
        x_left, y_left = zip(*left_eye)
        x_left = np.array(x_left)
        y_left = np.array(y_left)

        # Add first x_left and y value to the end of the list if a closed shape is desired
        x_left = np.append(x_left, x_left[0])
        y_left = np.append(y_left, y_left[0])

        # Fit spline to points
        tck, u = splprep([x_left, y_left], s=0)
        
        # Evaluate spline over a fine grid (increase 1000 for smoother curve)
        new_points_left = splev(np.linspace(0, 1, 1000), tck)
        
        # Draw the interpolated curve on the image
        for i in range(len(new_points_left[0]) - 1):
            cv2.line(
                output_image, 
                (int(new_points_left[0][i]), int(new_points_left[1][i])), 
                (int(new_points_left[0][i+1]), int(new_points_left[1][i+1])), 
                color=(255,255,255), 
                thickness=1
            )

        ##### RIGHT EYE #####

        # Separate points into two lists
        x_right, y_right = zip(*right_eye)
        x_right = np.array(x_right)
        y_right = np.array(y_right)

        # Add first x_right and y value to the end of the list if a closed shape is desired
        x_right = np.append(x_right, x_right[0])
        y_right = np.append(y_right, y_right[0])

        # Fit spline to points
        tck, u = splprep([x_right, y_right], s=0)
        
        # Evaluate spline over a fine grid (increase 1000 for smoother curve)
        new_points_right = splev(np.linspace(0, 1, 1000), tck)
        
        # Draw the interpolated curve on the image
        for i in range(len(new_points_right[0]) - 1):
            cv2.line(
                output_image, 
                (int(new_points_right[0][i]), int(new_points_right[1][i])), 
                (int(new_points_right[0][i+1]), int(new_points_right[1][i+1])), 
                color=(255,255,255), 
                thickness=1
            )

        ##### CREATE MASK #####

        # Convert points to suitable format for OpenCV
        points_left  = np.array([new_points_left[0], new_points_left[1]])
        points_left_cv = np.column_stack((points_left[0], points_left[1])).astype(np.int32)
        points_right  = np.array([new_points_right[0], new_points_right[1]])
        points_right_cv = np.column_stack((points_right[0], points_right[1])).astype(np.int32)

        # Create a black image, and a mask for the eyes
        mask = np.zeros_like(output_image)
        cv2.fillPoly(mask, [points_left_cv], color=255)
        cv2.fillPoly(mask, [points_right_cv], color=255)

        # Remove white areas outside the larger shape
        result = cv2.bitwise_and(output_image, mask)

        return result

