import numpy as np
import cv2 as cv

# Load images
TargetPX=900
imgs=[]
for j in range(1, 5):
    img = cv.imread(f'img/{j}.jpg')
    assert (img is not None), 'Cannot read the given images'
    # 가로를 TargetPX로 고정하고 세로 비율 유지
    target_width = TargetPX
    aspect_ratio = img.shape[0] / img.shape[1] # 세로/가로 비율
    target_height = int(target_width * aspect_ratio)
    dim = (target_width, target_height)
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)

    imgs.append(img)

# Retrieve matching points
fdetector = cv.BRISK_create()
fmatcher = cv.DescriptorMatcher_create('BruteForce-Hamming')
for i, img in enumerate(imgs):
    # 투명도 부여 초기값 255(불투명)
    b, g, r = cv.split(img)
    alpha = np.ones(b.shape, dtype=b.dtype) * 255
    img = cv.merge((b, g, r, alpha))
    if i == 0:
        img_merged = img
    else:
        keypoints_merged, descriptors_merged = fdetector.detectAndCompute(img_merged, None)
        keypoints, descriptors = fdetector.detectAndCompute(img, None)
        match = fmatcher.match(descriptors_merged, descriptors)

        # Calculate planar homography and merge two images
        pts1, pts2 = [], []
        for i in range(len(match)):
            pts1.append(keypoints_merged[match[i].queryIdx].pt)
            pts2.append(keypoints[match[i].trainIdx].pt)
        pts1 = np.array(pts1, dtype=np.float32)
        pts2 = np.array(pts2, dtype=np.float32)

        H, inlier_mask = cv.findHomography(pts2, pts1, cv.RANSAC)

        # img의 네 모서리 좌표 (좌상, 좌하, 우하, 우상)
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype='float32').reshape(-1, 1, 2)
        # H를 이용해 이 모서리들이 img2 좌표계에서 어디에 찍히는지 계산 (Perspective Transform)
        transformed_corners = cv.perspectiveTransform(corners, H)
        # x, y좌표 구분
        x_coords = transformed_corners[:, 0, 0]
        y_coords = transformed_corners[:, 0, 1]

        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        # 붙일 이미지가 왼쪽에 있을 경우 오른쪽으로 평행이동해서 정상적으로 붙이기
        dx = int(max(0, -min_x))
        dy = int(max(0, -min_y))
        # 합쳐진 이미지의 추정 사이즈
        dist_x = int(max(img_merged.shape[1], max_x) + dx)
        dist_y = int(max(img_merged.shape[0], max_y) + dy)

        # 이동 행렬 생성
        T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float32)
        H = T @ H

        result_img = cv.warpPerspective(img, H, (dist_x, dist_y)) # Perspective Transform
        # 기존 이미지를 새 캔버스에 배치 (덮어쓰기)
        #result_img[dy:img_merged.shape[0]+dy,dx:img_merged.shape[1]+dx] = img_merged
        
        # 1. 기존 이미지가 들어갈 영역(ROI)을 잘라냅니다.
        roi = result_img[dy:img_merged.shape[0]+dy, dx:img_merged.shape[1]+dx]

        # 기존 이미지(img_merged)에서 검은색이 아닌 부분(실제 그림) 마스크 생성
        img_merged_gray = cv.cvtColor(img_merged, cv.COLOR_BGRA2GRAY)
        _, mask = cv.threshold(img_merged_gray, 1, 255, cv.THRESH_BINARY)
        mask_inv = cv.bitwise_not(mask)

        # 마스크를 이용해 합성 (배경 제외)
        img_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
        img_fg = cv.bitwise_and(img_merged, img_merged, mask=mask)

        # 두 영역을 합쳐서 다시 넣습니다.
        result_img[dy:img_merged.shape[0]+dy, dx:img_merged.shape[1]+dx] = cv.add(img_bg, img_fg)
        
        img_merged = result_img

        # 검은 배경 투명하게
        gray = cv.cvtColor(img_merged, cv.COLOR_BGR2GRAY)
        is_black = (gray < 1) 
        img_merged[is_black, 3] = 0

cv.imwrite('stitched_result.png', img_merged)
cv.imshow("result", img_merged)
cv.waitKey(0)
cv.destroyAllWindows()