import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd

from imageUtil import clustering_image, remove_light_color, Canny_Detect

#Defining hàm để chia nhỏ bức ảnh 200x200 pixels thành từng phần nhỏ 40x40 pixels, và tính trung bình, độ lệch chuẩn của từng phần.
#Function INPUT: 1 ảnh kích cỡ 200x200
#Function OUTPUT: mảng feature bao gồm trung bình cộng và độ lệch chuẩn của 25 phần (40x40 pixel)
def features_section(img):
    features = np.array([], dtype='uint8')
    section = 1
    
    for y in range(0, img.shape[0], 40):
        for x in range(0, img.shape[1], 40):
            
            # crop ảnh về các section 40 * 40
            section_img = img[y:y+40, x:x+40]
            
            # tính trung bình và độ lệch chuẩn của pixels trong từng section 
            section_mean = np.mean(section_img)
            section_std = np.std(section_img)
            
            # tạo features array
            features = np.append(features, [section_mean, section_std])
    
    return features


path = 'img_data'
name_fruits = ['cachua', 'cam']
# name_fruits = ['cachua', 'cam', 'chuoi', 'duahau', 'le', 'nho', 'quyt', 'tao', 'thom', 'xoai']
dataset_train = {}
dataset_test = {}
test_size = 0.25
number = 1
dim = (200, 200)

for name in name_fruits:
  index = 0
  images_train = []
  images_test = []
  while index < 250:
    img_path = path+'/'+name+'/'+name+'_'+str(index)+'.jpg'
    
    #print(img_path)
    fruit_img = cv2.imread(img_path)
    #resize ảnh về 200 x 200
    fruit_img = cv2.resize(fruit_img, dim)
    fruit_img = cv2.cvtColor(fruit_img,cv2.COLOR_BGR2RGB)

    cluster, center = clustering_image(fruit_img )
    image = remove_light_color(cluster, center)
    
    edge = Canny_Detect(image)

    if len(images_test) < test_size*(len(os.listdir(path+'/'+name))):
      images_test.append(edge)
    else:
      images_train.append(edge)
    if index == 249:
      print('done {} image'.format(index + 1))
      figure_size = 6
      plt.figure(figsize=(figure_size,figure_size))
      plt.subplot(1,2,1),plt.imshow(fruit_img)
      plt.title('Anh Goc'), plt.xticks([]), plt.yticks([])
      plt.subplot(1,2,2),plt.imshow(edge)
      plt.title(name), plt.xticks([]), plt.yticks([])
    
      plt.show()
    index+=1
  dataset_train[name] = images_train
  dataset_test[name] = images_test
  
  
# Tạo array gồm các giá trị của featured
#Khởi tạo mảng numpy để lưu trữ feature
all_imgs = np.zeros((1, 50), dtype='uint8')
progress_counter = 0
name = []
x = []
section_std = []
for fruits in dataset_train: 
    for img in dataset_train[fruits]:


        img_features = features_section(img)
        #Chuyển array feature 1 hàng 50 cột thành 1 cột 50 hàng
        img_features = img_features.reshape(1, img_features.shape[0])

        all_imgs = np.append(all_imgs, img_features, axis=0)
        progress_counter += 1
        name.append (fruits)

all_imgs = all_imgs[1:]


# Tạo array tên đặc trưng
feature_name = []
section = 1
for y in range(0, 200, 40):
    for x in range(0, 200, 40):
        feature_name.append(f"sec{section}_mean")
        feature_name.append(f"sec{section}_std")
        section += 1
        
        
ftb_train = pd.DataFrame(all_imgs, columns= feature_name)
print (ftb_train.shape)
name_col = name
ftb_train.insert(loc = 0, column = "name", value= name_col)
ftb_train.head(200)


ftb_train.to_csv("edge+cluster_valid_features.csv", index = False)