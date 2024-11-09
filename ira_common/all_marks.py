import cv2
import numpy as np
import logging
from ira_common.mark import Mark
from skimage.metrics import structural_similarity
import skimage.morphology

class AllMarks:

    def __init__(self, canvas, debug: bool):
        self.canvas = canvas
        self.marks_array = []
        self.all_marks_mask = None
        self.no_marks = 0
        self.old_image = None
        self.new_image = None
        self.debug = debug

    def set_old_image(self, image):
        self.old_image = image

    def set_new_image(self, image):
        self.new_image = image

    def get_all_marks(self):
        """
        Return a list with all the mark objects in it.
        """
        return self.marks_array
    
    def has_large_white_area(self, mask, min_area):
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Check if any contour meets the minimum area requirement
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                return True
        return False

    def find_all_marks(self):
        """
        The main fucntion to call to find and analyse all new marks in the image.
        Method to pull out all the marks as a mask, 
        by looking at the difference between an image taken 
        before and after the mark was made.
        Both images should be in full color, and exactly the same dimensions.
        """

        # Compute SSIM between the two images

        # Increase saturation
        # Convert the image to HSV color space
        old_hsv_image = cv2.cvtColor(self.old_image, cv2.COLOR_BGR2HSV)
        new_hsv_image = cv2.cvtColor(self.new_image, cv2.COLOR_BGR2HSV)

        # Split out h,s,v
        h1, s1, b1 = cv2.split(old_hsv_image)
        h2, s2, b2 = cv2.split(new_hsv_image)

        # Calculate the absolute difference for each channel
        hue_diff = cv2.absdiff(h1, h2)
        sat_diff = cv2.absdiff(s1, s2)
        bright_diff = cv2.absdiff(b1, b2)

        # Define a threshold for the hue difference
        h_thresh = 50  # Adjust this value as needed  
        s_thresh = 30
        b_thresh = 20

        # Create a binary mask where differences greater than the threshold are white
        _, hue_mask = cv2.threshold(hue_diff, h_thresh, 255, cv2.THRESH_BINARY)
        _, sat_mask = cv2.threshold(sat_diff, s_thresh, 255, cv2.THRESH_BINARY)
        _, bright_mask = cv2.threshold(bright_diff, b_thresh, 255, cv2.THRESH_BINARY)

        # Define the area threshold as a percentage of the total image area
        percentage_threshold = 0.4  # For example, 5%
        image_area = old_hsv_image.shape[0] * old_hsv_image.shape[1]
        min_contour_area = (percentage_threshold / 100) * image_area
        print("min_contour_area: ", min_contour_area)

        # Step 1: Check the saturation mask
        if self.has_large_white_area(sat_mask, min_contour_area):
            final_mask = sat_mask
            print("Using the saturation mask")
        # Step 2: If no valid area in saturation, check the hue mask
        elif self.has_large_white_area(hue_mask, min_contour_area):
            final_mask = hue_mask
            print("Using the hue mask")
        # Step 3: If no valid area in hue, use the brightness mask
        elif self.has_large_white_area(bright_mask, min_contour_area):
            final_mask = bright_mask
            print("Using the brightness mask")
        # Step 4: If none of the masks meet the condition, default to an empty mask
        else:
            final_mask = np.zeros_like(sat_mask)
            print("No significant white areas found in any mask")

        # Find contours of the white regions in the combined mask
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create a new mask to store only the large regions
        filtered_mask = np.zeros_like(final_mask)
        # Loop through contours and filter by area
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            print("contour_area: ", contour_area)
            if contour_area >= min_contour_area:
                # Draw the contour on the filtered mask
                cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

        print("h1: ", h1)
        print("h2: ", h1)
        print("s1: ", h1)
        print("s2: ", h1)
        print("b1: ", h1)
        print("b2: ", h1)

        if self.debug == True:
            # Show the image to check/debug
            cv2.imshow("h1", h1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if self.debug == True:
            # Show the image to check/debug
            cv2.imshow("h2", h2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if self.debug == True:
            # Show the image to check/debug
            cv2.imshow("hue_mask", hue_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if self.debug == True:
            # Show the image to check/debug
            cv2.imshow("s1", s1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if self.debug == True:
            # Show the image to check/debug
            cv2.imshow("s2", s2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if self.debug == True:
            # Show the image to check/debug
            cv2.imshow("sat_mask", sat_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if self.debug == True:
            # Show the image to check/debug
            cv2.imshow("b1", b1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if self.debug == True:
            # Show the image to check/debug
            cv2.imshow("b2", b2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if self.debug == True:
            # Show the image to check/debug
            cv2.imshow("bright_mask", bright_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if self.debug == True:
            # Show the image to check/debug
            cv2.imshow("final_mask", final_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if self.debug == True:
            # Save or display the filtered mask
            cv2.imwrite("filtered_mask.jpg", filtered_mask)
            cv2.imshow("Filtered Mask", filtered_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        # (score, diff) = structural_similarity(h1, h2, full=True)
        # print("Image Similarity: {:.4f}%".format(score * 100))
        # TODO account for small position differences of camera...
        # find the difference mask for multiple pairs, where 2nd image is translated slightly in each one, relative to original image
        # if a difference area size changes significantly with shaking, it is probably not actually a difference


        # The diff image contains the actual image differences between the two images
        # and is represented as a floating point data type in the range [0,1] 
        # so we must convert the array to 8-bit unsigned integers in the range
        # [0,255] before we can use it with OpenCV
        # diff = (diff * 255).astype("uint8")
        # #diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # if self.debug == True:
        #     # Show the image to check/debug
        #     cv2.imshow("diff not gray", diff)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # # if self.debug == True:
        # #     # Show the image to check/debug
        # #     cv2.imshow("diff_gray", diff_gray)
        # #     cv2.waitKey(0)
        # #     cv2.destroyAllWindows()

        # # Threshold the difference image, followed by finding contours to
        # # obtain the regions of the two input images that differ
        # _, thresh = cv2.threshold(diff, 150, 255, cv2.THRESH_BINARY_INV)  # Higher number for 2nd param = more marks seen.

        # if self.debug == True:
        #     # Show the image to check/debug
        #     cv2.imshow("thresh_after", thresh)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        

        ######## Get the masked_image with just the new marks showing ############

        # Ensure final_mask is in the correct shape and type (single channel, binary)
        final_mask = cv2.threshold(filtered_mask, 127, 255, cv2.THRESH_BINARY)[1]

        # Create a white background (same size as original image)
        output_image = np.ones_like(self.new_image) * 255

        # Copy the masked area from the original image onto the output image
        output_image[final_mask == 255] = self.new_image[final_mask == 255]

        # Save or display the result
        if self.debug == True:
            cv2.imwrite("masked_output_image.jpg", output_image)
            cv2.imshow("Masked Output Image", output_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find all the marks in the image with areas above a given threshold
        mask_all = np.zeros(self.old_image.shape[:2], dtype="uint8")
        area_threshold = (self.canvas.transformed_image_x/15)*(self.canvas.transformed_image_y/15) 
        print("Min area threshold is: %s", area_threshold)
        contours_id = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > area_threshold:
                # It's a mark!
                # Run all the creation work for this mark.
                mask = np.zeros(self.old_image.shape[:2], dtype="uint8")
                cv2.drawContours(mask, contours, i, 255, -1)
                cv2.drawContours(mask_all, contours, i, 255, -1)
                if self.debug == True:
                    cv2.imshow(f"Mask for contour {i}", mask)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                contours_id.append(i)
                mark = Mark(self.debug)
                mark.real_area = area
                print("Area of the actual mark: %s", mark.real_area)
                mark.min_rect_area = cv2.minAreaRect(contour)
                print("Min rect area: %s", mark.min_rect_area)
                mark.mask = mask
                mark.analyse(self.new_image)
                self.marks_array.append(mark)
                self.no_marks += 1

        self.all_marks_mask = mask_all

        # Return the image with just the new marks
        return output_image
