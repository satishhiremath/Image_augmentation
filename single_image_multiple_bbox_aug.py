import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage, BoundingBox


image = cv2.imread('input_image.jpg', 1)
bbs = BoundingBoxesOnImage([
    BoundingBox(x1=125.5, y1=485.5, x2=400.5, y2=680.5),
    BoundingBox(x1=400.5, y1=750.5, x2=700.5, y2=800.5)
], shape=image.shape)


image_before = bbs.draw_on_image(image, size=2)
cv2.imwrite("before_augmentation.jpg", image_before)

seq = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=0.05*255),
    iaa.Fliplr(0.9)
])

image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

# print coordinates before/after augmentation (see below)
# use .x1_int, .y_int, ... to get integer coordinates
for i in range(len(bbs.bounding_boxes)):
    before = bbs.bounding_boxes[i]
    after = bbs_aug.bounding_boxes[i]
    print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
        i,
        before.x1, before.y1, before.x2, before.y2,
        after.x1, after.y1, after.x2, after.y2)
    )

image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])
cv2.imwrite("augmented_image.jpg", image_after)
