# Runestones recognition in Tower of Saviors
以CNN架構實現神魔之塔中符石的屬性辨識 (使用Keras + OpenCV)</br></br> 
Runestones recognition in Tower of Saviors via CNN neural network implemented in Keras + OpenCV

# Contents

- [**new_Tower of Saviors finalproject.py**](https://github.com/s90210jacklen/Runestones-recognition-for-the-Tower-of-Saviors/blob/master/new_Tower%20of%20Saviors_final%20project.py) : The main script launcher. This file contains all the code for UI options and OpenCV code to capture camera contents.</br> 
啟動此專案的檔案，其中包含了OpenCV的程式碼來讀取影像

- [**new_keras_cnn_神魔之塔-optimized-restore-model.ipynb**](https://github.com/s90210jacklen/Runestones-recognition-for-the-Tower-of-Saviors/blob/master/new_keras_cnn_%E7%A5%9E%E9%AD%94%E4%B9%8B%E5%A1%94-optimized-restore-model.ipynb) : This script file holds all the CNN specific code to create CNN model, load the weight file.</br> 
先前用來處理原始的資料，並以CNN的架構訓練模型

- [**final_model_weights_new.h5**](https://github.com/s90210jacklen/Runestones-recognition-for-the-Tower-of-Saviors/blob/master/final_model_weights_new.h5) : This is pretrained file.</br> 
先前訓練好的模型權重

- [**final_model_new.json**](https://github.com/s90210jacklen/Runestones-recognition-for-the-Tower-of-Saviors/blob/master/final_model_new.json) : This is pretrained file.</br>
先前所使用的模型架構

# Usage

**On Windows**
```bash
eg: With Tensorflow as backend
> python new_Tower of Saviors_final project.py 
```
- **Step1:**
Click the left button twice on left upper, upper right, lower left, and lower right corners to complete the coordinate record and press 's' key to save it in txt file.</br>
(紀錄四個座標點:依序點擊左上, 右上, 左下, 右下角各兩下完成座標的紀錄, 並按下's'鍵將其存入txt檔)

- **Step2:**
Press the 't' key to start execution.</br> 
(按下't'鍵開始執行)

- **Step3:**
Spin the runestones and back to Step2 or proceed to the next step. </br> 
(開始轉珠並回到步驟2,或者繼續到下一步)

- **Step4:**
Press the 'q' key to exit. </br> 
(按下'q'結束程式)

# Features
This application comes with CNN model to recognize 6 attributes of pretrained Runestones </br>
利用CNN的模型來辨識先前訓練過的6種屬性的符石
- Fire 火 → 1
- Water 水 → 2
- Earth 木 → 3
- Light 光 → 4
- Dark 暗 → 5
- Heart 心 → 6

This application provides following functionality:
- Prediction : Which allows the app to guess the Runestones against pretrained Runestones image. App can dump the prediction data to the console terminal or to a json file directly.


# Demo 
- The middle window is the predicted image of the CNN model, and the right window is the correct answer.</br> 
中間的視窗為CNN模型所預測的畫面,右邊視窗則為正確答案

![](https://github.com/s90210jacklen/Runestones-recognition-for-the-Tower-of-Saviors/blob/master/test_video.gif)


# Runestones Input
I used OpenCV to capture the image of the Runestones. In order to simplify the processing of images, I recorded four coordinate points to highlight contours & edges. Finally, use perspective transform & grayscale & thresholding for images.

- **Record four coordinate points** : Click the left button twice on left upper, upper right, lower left, and lower right corners to complete the coordinate record and press 's' key to save it in txt file.</br>
紀錄四個座標點:依序點擊左上, 右上, 左下, 右下角各兩下完成座標的紀錄, 並按下's'鍵將其存入txt檔
- **Perspective transform** : Perspective Transformation is the projecting of a picture into a new Viewing Plane, also known as Projective Mapping.</br>
透視變換是將圖片投影到一個新的視平面(Viewing Plane)，也稱作投影映射(Projective Mapping)

- **Grayscale** : Converting Image to Grayscale.</br>
將影像轉成灰階

- **Thresholding** : Thresholding is the simplest method of image segmentation. From a grayscale image, thresholding can be used to create binary images.</br>
二值化是圖像分割的一種最簡單的方法，可以把灰度圖像轉換成二值圖像。</br></br>

**Record four coordinate points** </br>
紀錄4個座標點
```python
def get_point(event,x,y,flags,param):
    global img2,index,x0,x1,x2,x3,y0,y1,y2,y3
    if event == cv2.EVENT_LBUTTONDBLCLK:#是不是滑鼠左鍵
        cv2.circle(img2,(x,y),3,(255,255,255),-1)
        cv2.imshow('image_mouse',img2)
        print("x:%d,y:%d"%(x,y))
        if index==0:
            x0 = x
            y0 = y
        elif index==1:
            x1 = x
            y1 = y
        elif index==2:
            x2 = x
            y2 = y
        elif index==3:
            x3 = x
            y3 = y    
        index = index +1    
```

![recorded coordinate points](https://github.com/s90210jacklen/Runestones-recognition-for-the-Tower-of-Saviors/blob/master/rocord_point.png)

**Press 's' key to save it** </br>
按下's'鍵存入txt檔

```python
 if … 
 … 
 … 
 
 elif k ==ord('s'):
        with open('c://test/pos2.txt','w') as f:
            f.write(str(x0)+"\n")
            f.write(str(y0)+"\n")
            f.write(str(x1)+"\n")
            f.write(str(y1)+"\n")
            f.write(str(x2)+"\n")
            f.write(str(y2)+"\n")
            f.write(str(x3)+"\n")
            f.write(str(y3)+"\n")
```
![save](https://github.com/s90210jacklen/Runestones-recognition-for-the-Tower-of-Saviors/blob/master/save.png)

**Perspective transform**
```python
pts1 = np.float32([[x0,y0],[x1,y1],[x2,y2],[x3,y3]])
pts2 = np.float32([[0,0],[300,0],[0,250],[300,250]])
                
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(300,250))
```

![Perspective transform](https://github.com/s90210jacklen/Runestones-recognition-for-the-Tower-of-Saviors/blob/master/Perspective_transform.png)


**Grayscale**
```python
# color to grayscale 轉灰階
dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
```

![Grayscale](https://github.com/s90210jacklen/Runestones-recognition-for-the-Tower-of-Saviors/blob/master/grayscale.png)


**Thresholding**</br>
二值化中有兩種演算法
- Mean shresholding (均值法)
- Gaussian shresholding (高斯法)

```python
#均值法
th2 = cv2.adaptiveThreshold(dst_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2) 
```

![Mean](https://github.com/s90210jacklen/Runestones-recognition-for-the-Tower-of-Saviors/blob/master/mean.png)


```python
#高斯法
th3 = cv2.adaptiveThreshold(dst_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) 
```
![Gaussian](https://github.com/s90210jacklen/Runestones-recognition-for-the-Tower-of-Saviors/blob/master/gaussian.png)

**The above two algorithms show that the image processed by the Mean shresholding has less noise and is more suitable as training data.**</br>
(以上兩種演算法可看出，均值法處理完的影像noise較少，較適合當作training data)

## CNN Model used
**Using the Keras CNN Sequential model with 4 Convolution Layer**</br>
(使用Keras的Sequential來建立4層Convolution的模型)

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
```
This model has 4 Convolutional Layer -
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 28, 28, 32)        320       
_________________________________________________________________
activation_1 (Activation)    (None, 28, 28, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 26, 26, 32)        9248      
_________________________________________________________________
activation_2 (Activation)    (None, 26, 26, 32)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 13, 13, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 13, 13, 64)        18496     
_________________________________________________________________
activation_3 (Activation)    (None, 13, 13, 64)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 11, 11, 64)        36928     
_________________________________________________________________
activation_4 (Activation)    (None, 11, 11, 64)        0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 64)          0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 5, 5, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1600)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 512)               819712    
_________________________________________________________________
activation_5 (Activation)    (None, 512)               0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 7)                 3591      
_________________________________________________________________
activation_6 (Activation)    (None, 7)                 0         
=================================================================
```
Total params: 888,295
Trainable params: 888,295
Non-trainable params: 0

**Test loss:** 0.0864271933833758</br> 
**Test accuracy:** 0.9666666746139526

# Training
I have used 840 images for training and 60 for testing and trained the model for 15 epochs.</br>
使用了840影像作訓練，60張影像作為測試，並設定epoch為15次

- Accuracy & Loss

![Accuracy](https://github.com/s90210jacklen/Runestones-recognition-for-the-Tower-of-Saviors/blob/master/Acc.png)

![Loss](https://github.com/s90210jacklen/Runestones-recognition-for-the-Tower-of-Saviors/blob/master/Loss.png)

# Confusion matrix </br>
混淆矩陣 </br></br>
**The diagonal is correct for prediction and the non-diagonal is for prediction error**</br>
對角線為預測正確，非對角線為預測錯誤
```python
import pandas as pd
prediction = model.predict_classes(x_test)
pd.crosstab(y_test_categories, prediction, rownames=['label'], colnames=['predict'])
```

![confusion matrix](https://github.com/s90210jacklen/Runestones-recognition-for-the-Tower-of-Saviors/blob/master/confusion%20matrix.png)


# Show the Runestone that were predicted incorrectly
顯示預測錯誤的符石
```python
df = pd.DataFrame( {'label':y_test_categories, 'predict':prediction} )
print(df.shape)
#df[:2]
df[(df.label=='1')&(df.predict==4)]
```

![label_predict](https://github.com/s90210jacklen/Runestones-recognition-for-the-Tower-of-Saviors/blob/master/label_predict.png)

```python
from matplotlib import pyplot as plt
# Plot inline
get_ipython().magic('matplotlib inline')
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()

plot_image(x_test_copy[29]) 
plot_image(x_test_copy[59])  
```

![label29](https://github.com/s90210jacklen/Runestones-recognition-for-the-Tower-of-Saviors/blob/master/label29.png)
![label59](https://github.com/s90210jacklen/Runestones-recognition-for-the-Tower-of-Saviors/blob/master/label59.png)


- It can be known that the attribute is a fire Runestone prediction error</br>
 可得知火屬性的符石預測錯誤
