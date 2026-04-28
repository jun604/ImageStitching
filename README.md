# Image Stitching
OpenCV를 이용하여 여러 장의 이미지를 하나의 이미지로 정합(Stitching)하는 프로그램</br>
BRISK 특징점 검출기와 RANSAC 기반의 호모그래피(Homography) 계산을 조합하여 구현</br>
이미지 투명도 처리 및 마스크 기반 합성을 통해 경계선 잘림 없는 자연스러운 정합 지원

## 기능
1. 이미지 전처리 및 해상도 최적화
   - 입력받은 이미지들을 가로 900px(TargetPX) 기준으로 비율을 유지하며 리사이즈
   - 특징점 검출의 효율성을 높이고 전체 정합 결과물의 해상도 밸런스 유지
2. 특징점 기반 매칭 및 강건한 호모그래피 산출
   - BRISK(Binary Robust Invariant Scalable Keypoints) 알고리즘을 통한 특징점 및 기술자 추출
   - BruteForce-Hamming 매칭을 사용하여 이미지 간 대응점 탐색
   - RANSAC(Random Sample Consensus)을 적용하여 Outlier를 제거하고 신뢰도 높은 평면 투영 행렬(H) 산출
3. 동적 캔버스 확장 및 좌표계 보정
   - Perspective Transform을 통해 각 모서리의 변환 좌표를 계산하여 정합에 필요한 최적의 캔버스 크기(dist_x, dist_y) 산출
   - 이미지가 캔버스 좌측/상단 밖으로 나가는 경우를 방지하기 위해 평행이동 행렬(T)을 연산하여 전체 이미지 좌표계 보정
4. 마스크 기반 정밀 합성 (Image Blending)
   - 단순 덮어쓰기 방식의 한계를 극복하기 위해 bitwise_and 연산과 이진화 마스크(Thresholding)를 활용
   - 기존 결과물(img_merged)과 새로운 이미지(result_img)가 겹치는 영역에서 유효한 데이터만 추출하여 합성함으로써 이미지 잘림 현상 원천 차단
5. 자동 배경 투명화 및 무손실 저장
   - 정합 완료 후 발생하는 기하학적 형태의 검은색 배경을 탐색하여 알파 채널(Alpha Channel) 값 0으로 처리
   - 배경 투명도가 유지된 상태로 무손실 압축 방식인 PNG 형식으로 자동 저장 지원

### 실행 결과
- stitched_result.png
  + 배경 투명화가 적용된 최종 정합 이미지
