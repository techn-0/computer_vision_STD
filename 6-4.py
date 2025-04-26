import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys
import winsound

class TrafficWeak(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('영화찾기')
        self.setGeometry(300, 300, 800, 300)

        # 버튼 및 라벨 생성
        signButton = QPushButton('포스터 등록', self)
        roadButton = QPushButton('타겟 불러오기', self)
        recognitionButton = QPushButton('인식', self)
        quitButton = QPushButton('종료', self)
        self.label = QLabel('welcome', self)

        # 버튼 및 라벨 위치 설정
        signButton.setGeometry(10, 10, 100, 30)
        roadButton.setGeometry(110, 10, 100, 30)
        recognitionButton.setGeometry(210, 10, 100, 30)
        quitButton.setGeometry(510, 10, 100, 30)
        self.label.setGeometry(10, 50, 780, 250)

        # 버튼 클릭 이벤트 연결
        signButton.clicked.connect(self.signFunction)
        roadButton.clicked.connect(self.roadFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

        # 포스터 파일 경로 및 이름
        self.signFiles = [['토토로.jpg', '토토로'], ['로보캅.jpg', '로보캅'], ['메트로폴리스.jpg', '메트로폴리스']]
        self.signImgs = []

    def signFunction(self):
        # 포스터 등록 함수
        self.label.clear()
        self.label.setText('포스터 등록')

        standard_size = (240, 320)  # 표준 포스터 크기

        for fname, _ in self.signFiles:
            img = cv.imread(fname)
            resized = cv.resize(img, standard_size)  # 크기 통일
            self.signImgs.append(resized)
            cv.imshow(fname, resized)

    def roadFunction(self):
        # 영화관 장면 이미지 불러오기
        if self.signImgs == []:
            self.label.setText('먼저 포스터 등록!')
        else:
            fname = QFileDialog.getOpenFileName(self, 'Open file', './')
            self.roadImg = cv.imread(fname[0])
            if self.roadImg is None:
                sys.exit('파일 찾을 수 없음')
            cv.imshow('Road scene', self.roadImg)

    def recognitionFunction(self):
        # 포스터 인식 함수
        if self.roadImg is None:
            self.label.setText('타겟 입력!')
            return

        sift = cv.SIFT_create()  # SIFT 객체 생성

        KD = []  # 등록 포스터들의 키포인트와 디스크립터 저장
        for img in self.signImgs:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            KD.append(sift.detectAndCompute(gray, None))

        # 장면 이미지의 키포인트 및 디스크립터 추출
        grayRoad = cv.cvtColor(self.roadImg, cv.COLOR_BGR2GRAY)
        road_kp, road_des = sift.detectAndCompute(grayRoad, None)

        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)

        display_img = self.roadImg.copy()  # 원본 훼손 방지용 복사본

        for i in range(len(KD)):
            sign_kp, sign_des = KD[i]
            knn_matches = matcher.knnMatch(sign_des, road_des, 2)

            # Lowe's ratio test 적용
            good_match = []
            for m, n in knn_matches:
                if m.distance < 0.4 * n.distance:
                    good_match.append(m)

            if len(good_match) < 10:
                continue  # 좋은 매칭이 충분하지 않으면 넘어감

            # 상위 30개 좋은 매칭만 사용
            good_match = sorted(good_match, key=lambda x: x.distance)[:30]

            # 매칭된 포인트 좌표 추출
            points1 = np.float32([sign_kp[m.queryIdx].pt for m in good_match])
            points2 = np.float32([road_kp[m.trainIdx].pt for m in good_match])

            # Homography 계산
            H, mask = cv.findHomography(points1, points2, cv.RANSAC, 3.0)
            if H is None or mask is None or np.sum(mask) < 10:
                continue  # Homography 실패나 신뢰성 낮으면 무시

            # 포스터 외곽 박스 변환
            h1, w1 = self.signImgs[i].shape[:2]
            box = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            box_transformed = cv.perspectiveTransform(box, H)

            # 초록색 박스 그리기
            display_img = cv.polylines(display_img, [np.int32(box_transformed)], True, (0, 255, 0), 4)

            # 매칭 결과(drawMatches) 이미지 생성
            img_match = np.empty(
                (max(h1, display_img.shape[0]), w1 + display_img.shape[1], 3),
                dtype=np.uint8
            )

            cv.drawMatches(
                self.signImgs[i], sign_kp,
                display_img, road_kp,
                good_match, img_match,
                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            cv.imshow(f"Matches: {self.signFiles[i][1]}", img_match)

            # 인식 성공 표시 및 소리 알림
            self.label.setText(self.signFiles[i][1] + ' 영화 감지됨!')
            winsound.Beep(2000, 300)


    def quitFunction(self):
        # 창 종료 함수
        cv.destroyAllWindows()
        self.close()

# 프로그램 실행
app = QApplication(sys.argv)
window = TrafficWeak()
window.show()
app.exec_()
