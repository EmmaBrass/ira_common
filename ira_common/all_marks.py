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
        self.debug=debug

    def set_old_image(self, image):
        self.old_image = image

    def set_new_image(self, image):
        self.new_image = image

    def get_all_marks(self):
        """
        Return a list with all the mark objects in it.
        """
        return self.marks_array

    def find_all_marks(self):
        """
        The main fucntion to call to find and analyse all new marks in the image.
        Method to pull out all the marks as a mask, 
        by looking at the difference between an image taken 
        before and after the mark was made.
        Both images should be in full color, and exactly the same dimensions.
        """

        # Compute SSIM between the two images
        (score, diff) = structural_similarity(self.old_image, self.new_image, channel_axis=2 ,full=True)
        print("Image Similarity: {:.4f}%".format(score * 100))
        # TODO this needs to be made better... does NOT always work well...

        # The diff image contains the actual image differences between the two images
        # and is represented as a floating point data type in the range [0,1] 
        # so we must convert the array to 8-bit unsigned integers in the range
        # [0,255] before we can use it with OpenCV
        diff = (diff * 255).astype("uint8")
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        _, thresh = cv2.threshold(diff_gray, 150, 255, cv2.THRESH_BINARY_INV)

        if self.debug == True:
            # Show the image to check/debug
            cv2.imshow("thresh_after", thresh)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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