
# coding: utf-8

# In[22]:


import cv2
import numpy as np
import csv
import os


# In[23]:


from __future__ import print_function
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from scipy import misc
from numpy import genfromtxt
import csv


# In[24]:


from keras.models import model_from_json
json_file = open('c://test/final_model_new.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("c://test/final_model_weights_new.h5")
print("Loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[25]:


x0=0
x1=0
x2=0
x3=0
y0=0
y1=0
y2=0
y3=0
index=0
folder_index =0
font = cv2.FONT_HERSHEY_SIMPLEX
#             "火","水","綠","土","月","心"
look_table = ["20","13","31","33","00","45"]


def FileOpen(fn):
    try:
      f=open(fn, "r")
      global x0,y0,x1,y1,x2,y2,x3,y3
      x0 = f.readline()
      y0 = f.readline()
      x1 = f.readline()
      y1 = f.readline()
      x2 = f.readline()
      y2 = f.readline()
      x3 = f.readline()
      y3 = f.readline()
      return 1
    except IOError:
      print("Error: File does not appear to exist.")
      return 0

ret = FileOpen('c:/test/pos2.txt')

    

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


# In[26]:


cam = cv2.VideoCapture(0) 
 
cv2.namedWindow('image')
cv2.setMouseCallback('image',get_point)
ret_val, img2 = cam.read()
#img2 = cv2.imread('c://test/src.jpg')
#img = cv2.imread('c://test/src.jpg')
#cv2.imshow('image',img)

while(1):
    ret_val, img = cam.read()
    #img = cv2.imread('c://test/src.jpg')
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('t'):
        
        directory = 'c:/test/img_out/'+str(folder_index)#尋找目前index的folder
        if not os.path.exists(directory):#沒有的化就create一個folder
            os.makedirs(directory)
        
        
        
        pts1 = np.float32([[x0,y0],[x1,y1],[x2,y2],[x3,y3]])
        pts2 = np.float32([[0,0],[300,0],[0,250],[300,250]])
        
        
        
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,M,(300,250))
        dst2 = dst.copy()
        
        dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        
        cv2.imshow('dst image',dst)
        
        th2 = cv2.adaptiveThreshold(dst_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(dst_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        
        #cv2.imshow('ADAPTIVE_THRESH_MEAN_C',th2)
        #cv2.imshow('ADAPTIVE_THRESH_GAUSSIAN_C',th3)
        
        grid_h = 5
        grid_w = 6
        grid_pixel = 50
        of = 10;
        arr  = np.zeros(shape=(grid_h,grid_w), dtype = int)
        arr_cnn  = np.zeros(shape=(grid_h,grid_w), dtype = int)
        out_str =""
        out_str_cnn =""
        # write jpeg
        #for yy in range(0,grid_h,1):
        #  for xx in range(0,grid_w,1):
        #      img_src = th2[yy*grid_pixel+of:(yy+1)*grid_pixel-of,xx*grid_pixel+of:(xx+1)*grid_pixel-of]
        #      resized_image = cv2.resize(img_src, (28, 28))   
        #      cv2.imwrite(directory+'/'+str(yy)+str(xx)+'.jpg',resized_image)
            
            
        # compare #這裡只是為了跟CNN做比較,所以才又做
        for yy in range(0,grid_h,1):
          for xx in range(0,grid_w,1):
              img_src = th2[yy*grid_pixel:(yy+1)*grid_pixel,xx*grid_pixel:(xx+1)*grid_pixel]
              gmax_val =0
              for i in range(0,6,1): # total 1.jpg ~ 6.jpg
                    #print "./template_jpg/"+look_table[i] + ".jpg"
                    template = cv2.imread("c://test/template/"+look_table[i] + ".jpg",0)
                    #print(img_src.shape)
                    #print(template.shape)
                    res = cv2.matchTemplate(img_src,template,cv2.TM_CCOEFF_NORMED)
                    #max_val = np.amax(res)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    if(max_val > gmax_val):
                             gmax_val = max_val
                             index = i+1

              arr[yy][xx] = index
              out_str +=str(index)
        # Cnn   
        for yy in range(0,grid_h,1): #拿# write jpeg的IMG做測試
          for xx in range(0,grid_w,1):
              img_src = th2[yy*grid_pixel+of:(yy+1)*grid_pixel-of,xx*grid_pixel+of:(xx+1)*grid_pixel-of]#取出img resource
              resized_image = cv2.resize(img_src, (28, 28))#resized成28*28 ->拿img去做測試
              resized_image = resized_image.astype('float32')
              resized_image /= 255 #正規化
              resized_image = resized_image.reshape(1,28, 28, 1)#resized成這樣才能呼叫loaded_model.predict_classes
              index= loaded_model.predict_classes(resized_image)#resized_image就是圖片陣列 但只代表一張圖 藉由它來算出1個index
              arr_cnn[yy][xx] = index#把index存起來
              out_str_cnn +=str(index)
        
       
        
        #print(out_str) SHOW 2張圖  CNN會有誤差 可能要撐加Train patten來提高準確率
        for yy in range(0,grid_h,1):
           for xx in range(0,grid_w,1):
             cv2.putText(dst,str( arr[yy][xx]) ,( int((xx+0.5)*grid_pixel),int((yy+0.5)*grid_pixel)), font, 1,(0,0,0),2,cv2.LINE_AA)
     
        cv2.imshow('detect',dst)
        
        for yy in range(0,grid_h,1):
           for xx in range(0,grid_w,1):
             cv2.putText(dst2,str( arr_cnn[yy][xx]) ,( int((xx+0.5)*grid_pixel),int((yy+0.5)*grid_pixel)), font, 1,(0,0,0),2,cv2.LINE_AA)
     
        cv2.imshow('cnn_detect',dst2)
        
        #cv2.imwrite(directory+'/detect.jpg',dst)
        
        # write csv out
        #print(arr.tolist())
        #myFile = open(directory+'/label.csv', 'w')  
        #with myFile:  
        #   writer = csv.writer(myFile,lineterminator='\n')
        #   writer.writerows(arr.tolist())
            
        folder_index = folder_index +1    
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
    elif k == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()

