import numpy as np
import cv2 as cv

# Load two images
img1 = cv.imread('sample1.png')
img2 = cv.imread('sample2.png')
img3 = cv.imread('sample3.png')
img4 = cv.imread('sample4.png')
assert (img1 is not None) and (img2 is not None) and (img3 is not None) and (img4 is not None), 'Cannot read the given images'
imgs = [img1, img2, img3, img4]

# Retrieve matching points
fdetector = cv.BRISK_create()
keypoints1, descriptors1 = fdetector.detectAndCompute(img1, None)
keypoints2, descriptors2 = fdetector.detectAndCompute(img2, None)
keypoints3, descriptors3 = fdetector.detectAndCompute(img3, None)
keypoints4, descriptors4 = fdetector.detectAndCompute(img4, None)

fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
match12 = fmatcher.match(descriptors1, descriptors2)
match23 = fmatcher.match(descriptors2, descriptors3)
match34 = fmatcher.match(descriptors3, descriptors4)
match41 = fmatcher.match(descriptors4, descriptors1)

# Calculate planar homography and merge two images
pts1, pts2 = [], []
for i in range(len(match12)):
    pts1.append(keypoints1[match12[i].queryIdx].pt)
    pts2.append(keypoints2[match12[i].trainIdx].pt)
pts1 = np.array(pts1, dtype=np.float32)
pts2 = np.array(pts2, dtype=np.float32)

H, inlier_mask = cv.findHomography(pts2, pts1, cv.RANSAC)

# img1의 네 모서리 좌표 (좌상, 좌하, 우하, 우상)
h, w = img1.shape[:2]
corners = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype='float32').reshape(-1, 1, 2)
# H를 이용해 이 모서리들이 img2 좌표계에서 어디에 찍히는지 계산 (Perspective Transform)
transformed_corners = cv.perspectiveTransform(corners, H)
# x, y좌표 구분
x_coords = transformed_corners[:, 0, 0]
y_coords = transformed_corners[:, 0, 1]

min_x, max_x = np.min(x_coords), np.max(x_coords)
min_y, max_y = np.min(y_coords), np.max(y_coords)


dx = 0
dy = 0

if min_x < 0:
    dx = int(abs(min_x))
if min_y < 0:
    dy = int(abs(min_y))
# 이동 행렬 생성
T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float32)
H = T @ H

if dx is 0:
    dx = int(img2.shape[1]-img1.shape[1] + max_x if img2.shape[1] > img1.shape[1] else 0)
if dy is 0:
    dy = int(img2.shape[0]-img1.shape[0] + max_y if img2.shape[0] > img1.shape[0] else 0)


img_merged = cv.warpPerspective(img2, H, (img1.shape[1] + dx + 1, img1.shape[0] + dy + 1)) # Perspective Transform
img_merged[dy:img1.shape[0]+dy,dx:img1.shape[1]+dx] = img1 # Copy

# Show the merged image
img_matched = cv.drawMatches(img1, keypoints1, img2, keypoints2, match12, None, None, None,
                             matchesMask=inlier_mask.ravel().tolist()) # Remove `matchesMask` if you want to show all putative matches
merge = [img1, img2, img_matched, img_merged]
for i, result in enumerate(merge):
    cv.imshow(f'Planar Image Stitching {i+1}', result)
cv.waitKey(0)
cv.destroyAllWindows()