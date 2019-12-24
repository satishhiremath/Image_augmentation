import imgaug as ia
import imageio
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage


ia.seed(1)

image = imageio.imread("input.jpg")

x1=125.5
y1=485.5
x2=400.5
y2=680.5

kps = KeypointsOnImage([
    Keypoint(x=x1, y=y1),
    Keypoint(x=x2, y=y1),
    Keypoint(x=x2, y=y2),
    Keypoint(x=x1, y=y2)
], shape=image.shape)

seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect keypoints
    iaa.Affine(
        rotate=45
    )
])

# Augment keypoints and images.
image_aug, kps_aug = seq(image=image, keypoints=kps)

# print coordinates before/after augmentation (see below)
# use after.x_int and after.y_int to get rounded integer coordinates
for i in range(len(kps.keypoints)):
    before = kps.keypoints[i]
    after = kps_aug.keypoints[i]
    print("Keypoint %d: (%.8f, %.8f) -> (%.8f, %.8f)" % (
        i, before.x, before.y, after.x, after.y)
    )

# image with keypoints before/after augmentation (shown below)
image_before = kps.draw_on_image(image, size=7)
image_after = kps_aug.draw_on_image(image_aug, size=7)

ia.imshow(image_before)
ia.imshow(image_after)
