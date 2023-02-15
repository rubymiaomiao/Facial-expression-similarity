import sys,os,cv2,random,numpy as np
from datetime import datetime 
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from untitled import Ui_MainWindow # 導入圖形按鈕關聯介面
import dlib
from imutils import face_utils
from numpy.linalg import pinv
from scipy.spatial import distance
import statistics



datFile =  "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def opencvimg_2_pixmap(srcMat):
        cv2.cvtColor(srcMat, cv2.COLOR_BGR2RGB,srcMat)
        height, width, bytesPerComponent= srcMat.shape
        bytesPerLine = bytesPerComponent* width
        srcQImage= QImage(srcMat.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return QPixmap.fromImage(srcQImage)

class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        #繼承(QMainWindow,Ui_MainWindow)父類的屬性
        super(MainWindow,self).__init__()
        # 初始化介面組件
        self.setupUi(self)
        self.label_11.setPixmap(QPixmap(r"static\no_signal.jpg").scaled(677,647))#左框
        self.label_11.setScaledContents(True)
        self.label_14.setScaledContents(True)#右框
        self.label_3.setScaledContents(True)#左框
        self.label_2.setScaledContents(True)#右框
        self.last_similarity_0 = 0
        self.last_similarity_1 = 0
        self.last_similarity_2 = 0
        self.face_key_points = 0

        #初始化頁面設置
        self.stackedWidget.setCurrentIndex(0)

        #設置定時器
        self.time = QTimer()
        #更新畫面和分數
        self.time.timeout.connect(self.refrsh)


        #两个玩家模式
        self.time1 = QTimer()
        self.time1.timeout.connect(self.refrsh1)

        ########################## 按鈕事件 #####################################

        #開始按鈕
        self.pushButton.clicked.connect(self.startMain)
        
        ##########選擇模式

        #單人模式
        self.pushButton_6.clicked.connect(lambda :self.stackedWidget.setCurrentIndex(2))

        #多人模式
        self.pushButton_5.clicked.connect(self.getCamra1)

        #if 單人
        # level 1
        self.pushButton_2.clicked.connect(self.getNum1)
        # level 2
        self.pushButton_3.clicked.connect(self.getNum2)
        # level 3
        self.pushButton_4.clicked.connect(self.getNum3)

        self.status = None  # 判断状态
        self.cap = None
        self.cap1 = None

        # 暫停音樂
        self.pushButton_17.clicked.connect(self.stopMusic)
        # 播放音樂
        self.pushButton_9.clicked.connect(self.startMusic)
        # 重玩
        self.pushButton_10.clicked.connect(self.backMain)

        #結束遊戲
        self.pushButton_11.clicked.connect(self.overGame1)
        self.verticalSlider.valueChanged.connect(self.valChange)#相似度显示




        #多人模式
        self.status1 = None
        self.cap = None
        self.cap1 = None
        
        self.pushButton_18.clicked.connect(self.stopMusic1)
        self.pushButton_14.clicked.connect(self.startMusic1)
        self.pushButton_15.clicked.connect(self.backMain)
        self.pushButton_16.clicked.connect(self.overGame1)
        self.verticalSlider_2.valueChanged.connect(self.valChange1)
        
        #返回主界面
        self.pushButton_12.clicked.connect(lambda :self.stackedWidget.setCurrentIndex(0))
        #結束遊戲
        self.pushButton_13.clicked.connect(self.close)

     #雙人

    def stopMusic1(self):
        self.media_player.pause()

    def startMusic1(self):
        self.media_player.play()

    #相似度數字
    def valChange1(self):
        self.label_13.setNum(self.verticalSlider_2.value()) 

    #單人

    def stopMusic(self):
        self.media_player.pause()

    def startMusic(self):
        self.media_player.play()

    def valChange(self):
        self.label_7.setNum(self.verticalSlider.value()) 


    #開始初始化
    def startMain(self):
        # 單人模式初始化
        self.label.clear()
        self.label_6.clear()
        self.label_7.clear()
        self.verticalSlider.setValue(0)

        #多人模式初始化
        self.label_13.clear()
        self.label_16.clear()
        self.label_18.clear()
        self.verticalSlider_2.setValue(0)
   
        #分數
        self.score = 0

        self.media_player = QMediaPlayer(self)
        self.stackedWidget.setCurrentIndex(1)

    
    #回主頁面
    def backMain(self):
        self.media_player.stop()
        self.stackedWidget.setCurrentIndex(0)

    #結束顯示分數 
    def overGame1(self):
        if self.time1.isActive():
            self.time1.stop()
            self.cap.release()
            self.cap1.release()
        self.media_player.stop()
        self.stackedWidget.setCurrentIndex(5)
        self.label_8.setText(f"分數：{self.score}")




    #多人模式
    #獲取攝像頭
    def getCamra1(self):
        self.stackedWidget.setCurrentIndex(4)
        abs_path = os.path.abspath(r"static\music\background.mp3")
        url = QUrl.fromLocalFile(abs_path)
        c = QMediaContent(url)
        self.media_player.setMedia(c)
        self.media_player.play()

        if self.time1.isActive():
            self.cap.release()
            self.cap1.release()
            self.time1.stop()

        self.cap = cv2.VideoCapture(0)
        
        try:
            self.cap1 = cv2.VideoCapture(1)
        except:
            print("no signal")
            self.label_11.setPixmap(QPixmap(r"static\no_signal.jpg").scaled(677,647))
        self.localTimeDouble = datetime.now()
        
        self.status1 = 0
        self.time1.start()
    
    def refrsh1(self):
        if self.status1 == 0:
            print("status1 == 0")
            now = datetime.now()
            dateGet = (now - self.localTimeDouble).seconds
            self.label_16.setText(str(15-dateGet))  
            
            #右邊的玩家
            ret, frame = self.cap.read()
            

            if ret:
                #從攝像頭讀取畫面
                after_frame =cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
                after_frame =cv2.flip(after_frame, 1)
                #畫人臉特徵
                draw_frame = self.draw_rectangle(after_frame)

                print("draw_frame 2people right")

                self.face_key_points_right = self.face_key_points


                # 格式轉變，BGRtoRGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 進行檢測
                r_image = np.array(frame)
                #畫面長度
                showImage = QImage(r_image, r_image.shape[1], r_image.shape[0], 3 * r_image.shape[1], QImage.Format_RGB888).scaled(677,677)
                #顯示右邊畫面
                self.label_14.setPixmap(QPixmap.fromImage(showImage))
                self.label_14.setPixmap(opencvimg_2_pixmap(draw_frame))
            
            try:
                ret1, frame1 = self.cap1.read()
                if ret1:
                    after_frame =cv2.resize(frame1, (640, 480), interpolation=cv2.INTER_CUBIC)
                    after_frame =cv2.flip(after_frame, 1)
                    self.draw_frame_1 = self.draw_rectangle(after_frame)
                    self.face_key_points_left = self.face_key_points
                    print("draw_frame 2p left")

                    similarity = self.compare_face(self.face_key_points_left, self.face_key_points_right)
                    self.verticalSlider_2.setValue(similarity)
                    
                    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                    r_image = np.array(frame)
                    showImage_1 = QImage(r_image, r_image.shape[1], r_image.shape[0], 3 * r_image.shape[1], QImage.Format_RGB888).scaled(677,677)
                    self.label_11.setPixmap(QPixmap.fromImage(showImage_1))    
                    self.label_11.setPixmap(opencvimg_2_pixmap(self.draw_frame_1))

                    if dateGet == 15:
                        num = self.count_score(similarity)#奖励
                        self.score += num
                        self.label_18.setText(str(int(self.score)))
                        self.verticalSlider.setValue(similarity)
                        self.localTimeDouble = now   
            except:
                pass
        
                



    
    

    #單人模式
    def getNum1(self):
        abs_path = os.path.abspath(r"static\music\background.mp3")
        url = QUrl.fromLocalFile(abs_path)
        c = QMediaContent(url)
        self.media_player.setMedia(c)
        self.media_player.play()
        
        #時間和圖片張數限制
        self.resetTime = 10
        self.picTotal = 14

        self.index = 0
        self.stackedWidget.setCurrentIndex(3)
        if self.time.isActive():
            self.cap.release()
            self.time.stop()
        self.localTimeDouble = datetime.now()
        self.cap = cv2.VideoCapture(0)

        self.load_next_reference_image()
        self.time.start()


    def getNum2(self):
        abs_path = os.path.abspath(r"static\music\background.mp3")
        url = QUrl.fromLocalFile(abs_path)
        c = QMediaContent(url)
        self.media_player.setMedia(c)
        self.media_player.play()
        

        #限制時間7s
        self.resetTime = 7      
        #圖片數量
        self.picTotal = 14
        self.index = 0
        #跳轉頁面
        self.stackedWidget.setCurrentIndex(3)

        if self.time.isActive():
            self.cap.release()
            self.time.stop()
        self.localTimeDouble = datetime.now()
        self.cap = cv2.VideoCapture(0)

        self.load_next_reference_image()
        self.time.start()


    def getNum3(self):
        abs_path = os.path.abspath(r"static\music\background.mp3")
        url = QUrl.fromLocalFile(abs_path)
        c = QMediaContent(url)
        self.media_player.setMedia(c)
        self.media_player.play()
        
        self.resetTime = 3
        #圖片數量
        self.picTotal = 25
        self.index = 0
        self.stackedWidget.setCurrentIndex(3)
        #防止報死的，判斷定時器是否開啓，防止重復開攝像頭，浪費資源
        if self.time.isActive():
            self.cap.release()
            self.time.stop()
        self.localTimeDouble = datetime.now()
        self.cap = cv2.VideoCapture(0)
        self.load_next_reference_image()
        
        self.time.start()

    #刷新頁面；更新分數和頁面
    def refrsh(self):
        ret, frame = self.cap.read()
        if ret:
            after_frame =cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
            after_frame =cv2.flip(after_frame, 1)
            draw_frame = self.draw_rectangle(after_frame)

            similarity = self.compare_face(self.face_key_points_reference, self.face_key_points)
            self.verticalSlider.setValue(similarity)
            #print(similarity)
            

            #計算用了多少分鐘
            now = datetime.now()
            dateGet = (now - self.localTimeDouble).seconds
            self.label.setText(str(self.resetTime - dateGet))
            #print("剩餘時間：",str(self.resetTime - dateGet))

            #當時間到：
            if dateGet == self.resetTime:
                self.index += 1
                print("change picture")

                #如果圖片放完
                if self.index == self.picTotal:
                    self.time.stop()
                    self.cap.release()
                    self.media_player.stop()
                    self.stackedWidget.setCurrentIndex(5)
                    self.label_8.setText(f"分数：{self.score}")

                self.localTimeDouble = now

                #刷新表情包
                self.load_next_reference_image()
                #刷新分數        
                num = self.count_score(similarity)#奖励
                self.score += num
                self.label_6.setText(str(self.score))           
                #相似度顯示
                self.verticalSlider.setValue(similarity)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r_image = np.array(frame)
            showImage = QImage(r_image, r_image.shape[1], r_image.shape[0], 3 * r_image.shape[1], QImage.Format_RGB888).scaled(584,345)
            self.label_2.setPixmap(QPixmap.fromImage(showImage))
            self.label_2.setPixmap(opencvimg_2_pixmap(draw_frame))
    
    
    
    #框出人臉+得到人臉關鍵點
    def draw_rectangle(self, frame):
        res_reference = cv2.resize(frame,(640, 480), interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(res_reference, cv2.COLOR_BGR2GRAY)

        # 在灰度圖中檢測人臉
        self.faces = detector(gray, 0)

        for face in self.faces:
            # 確定面部特徵
            # 将特徵 (x, y) 座標轉換為 NumPy 數組
            shape = predictor(gray, face)

            self.face_key_points = face_utils.shape_to_np(shape)
 
            (x, y, w, h) = face_utils.rect_to_bb(face)
            #cv2.rectangle(res_reference, (x, y), (x + w, y + h), (255, 0, 0), 3)
            #注释cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度)
            
            for (x, y) in self.face_key_points:
                cv2.circle(res_reference, (x, y), 3, (0, 0, 255), 1)
                
                #print((x,y))
            # 圖像、圓心、半徑、顏色、第五個參數正數為線的粗細，負數則為填滿
        
        return res_reference
    

    #計算分數
    def count_score(self,similarity):

        if similarity >= 85:
            return 6
        
        elif similarity >=60 and similarity < 85 :
            return 4
        
        elif similarity >=40 and similarity < 60 :
            return 2

        else :
            return 0




    
    #加載下一張圖片
    def load_next_reference_image(self):
        #從文件中獲取圖片
        self.data = []
        path = "img"
        for i in os.listdir(path):
            img_reference_file_name = os.path.join(path, i)
            self.data.append(img_reference_file_name)
        self.next_emoji= random.choice(self.data)
        #opencv 
        img_reference = cv2.imread(self.next_emoji)
        res_reference = cv2.resize(img_reference,(640, 480), interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(res_reference, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces:
            shape = predictor(gray, face)
            
            self.face_key_points_reference = face_utils.shape_to_np(shape)
 
            (x, y, w, h) = face_utils.rect_to_bb(face)
            #cv2.rectangle(res_reference, (x, y), (x + w, y + h), (255, 0, 0), 2)

            for (x, y) in self.face_key_points_reference:
                cv2.circle(res_reference, (x, y), 4, (0, 0, 255), -1)
                #打印关键点
                #print((x,y))
            # 圖像、圓心、半徑、顏色、第五個參數正數為線的粗細，負數則為填滿

        self.label_3.setPixmap(QPixmap(self.next_emoji).scaled(640, 480))
        self.label_3.setPixmap(opencvimg_2_pixmap(res_reference))


        
    ###################### compare face #################

    #計算相似度
    def compare_face(self, face_key_points_1, face_key_points_2):
        
        #1是表情包的特徵點，2是攝像頭抓捕到的人臉特徵

        #檢查是否檢測到臉部
        if len(self.faces) == 0:
            return 0
        face_key_points_1 = np.insert(face_key_points_1, 2, values=1, axis=1)
        face_key_points_2 = np.insert(face_key_points_2, 2, values=1, axis=1)
        


        # alignement_1 對齊1
        #縮放
        a = 1/(distance.euclidean(face_key_points_1[30, :2], face_key_points_1[8, :2]))
        T_1 = np.array([[a,0,0],[0,a,0],[0,0,1]])
        face_key_points_1_transformed = np.transpose(np.dot(T_1, np.transpose(face_key_points_1)))
        
        # alignement_2 
        location_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                36, 39, 42, 45,
                                31, 32, 33, 34, 35,
                                27, 28, 29, 30]
        
        I = np.transpose(face_key_points_2[location_indexes, :])

        I_p = np.transpose(face_key_points_1_transformed[location_indexes, :])
        
        #pinv(I)返回矩陣 I 的 Moore-Penrose 為你矩陣
        #最小二乘法
        T_2 = np.dot(I_p, pinv(I)) 





        # vectors
        signature_indexes = [17, 18, 19, 20, 21,                                # left eye brow 左眼眉
                                22, 23, 24, 25, 26,                             # right eye brow
                                36, 37, 38, 39, 40, 41,                         # left eye
                                42, 43, 44, 45, 46, 47,                         # right eye
                                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, # outside mouth
                                60, 61, 62, 63, 64, 65, 66, 67]               # inside mouth                                         # pupil
        center_index = 30
        # vectors_1 
        face_key_points_1_signature_aligned = np.transpose(np.dot(T_1, np.transpose(face_key_points_1[signature_indexes, :])))
        face_key_points_1_center_aligned = np.transpose(np.dot(T_1, np.transpose(face_key_points_1[center_index, :])))
        vectors_1 = face_key_points_1_signature_aligned[:,:2]-face_key_points_1_center_aligned[:2]
        # vectors_2
        face_key_points_2_signature_aligned = np.transpose(np.dot(T_2, np.transpose(face_key_points_2[signature_indexes, :])))
        face_key_points_2_center_aligned = np.transpose(np.dot(T_2, np.transpose(face_key_points_2[center_index, :])))
        vectors_2 = face_key_points_2_signature_aligned[:,:2]-face_key_points_2_center_aligned[:2]



        # squared Euclidean distance 
        squared_euclidean_distance = 0
        for i in range(len(signature_indexes)):
            squared_euclidean_distance += (distance.euclidean(vectors_1[i, :], vectors_2[i, :]))**2
        #print("squared_euclidean_distance:",squared_euclidean_distance)
        distance_when_similarity_is_90 = 0.2
        distance_when_similarity_is_0 = 0.9
        squared_euclidean_similarity = min(1, max(0, (distance_when_similarity_is_0-squared_euclidean_distance)/(distance_when_similarity_is_0-distance_when_similarity_is_90)))

        # cos distance 
        cos_distance = 0
        for i in range(len(signature_indexes)):
            cos_distance += distance.cosine(vectors_1[i, :], vectors_2[i, :])
        distance_when_similarity_is_90 = 0.2
        distance_when_similarity_is_0 = 1.0
        cos_similarity = min(1, max(0, (distance_when_similarity_is_0-cos_distance)/(distance_when_similarity_is_0-distance_when_similarity_is_90)))
        similarity = statistics.mean([squared_euclidean_similarity, cos_similarity])
        #print("s:",similarity*100)

        # Gauss filter
        gauss_filter = [7/74, 26/74, 41/74]
        self.last_similarity_2 = self.last_similarity_1
        self.last_similarity_1 = self.last_similarity_0
        self.last_similarity_0 = similarity
        similarity = np.dot(gauss_filter, [self.last_similarity_2, self.last_similarity_1, self.last_similarity_0])

        #print('similarity={}\r'.format(similarity), end='')
        return ((similarity*100)**0.5)*10    
    
        
    def closeEvent(self, a0: QCloseEvent) -> None:
        if self.time.isActive():
            self.cap.release()
            self.time.stop()

if __name__ == "__main__":
    #創建QApplication 固定寫法
    app = QApplication(sys.argv)
    # 實例化介面
    window = MainWindow()
    #顯示介面
    window.show()
    #阻塞
    sys.exit(app.exec_())