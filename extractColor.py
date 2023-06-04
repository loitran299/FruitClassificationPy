import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd

from imageUtil import clustering_image, remove_light_color

def create_df_with_many_images(dataset):
    # columns_name = ['name', 'red_mean', 'red_std', 'green_mean', 'green_std', 'blue_mean', 'blue_std']
    red_means = []
    red_stds = []
    green_means = []
    green_stds = []
    blue_means = []
    blue_stds = []
    names = []
    for fruit in dataset:
      for img in dataset[fruit]:
            
        red_channel = img[:,:,0]
        green_channel = img[:,:,1]
        blue_channel = img[:,:,2]

        red_mean = np.mean(red_channel)
        green_mean = np.mean(green_channel)
        blue_mean = np.mean(blue_channel)

        red_std = np.std(red_channel)
        green_std = np.std(green_channel)
        blue_std = np.std(blue_channel)

        red_means.append(red_mean)
        red_stds.append(red_std)
        green_means.append(green_mean)
        green_stds.append(green_std)
        blue_means.append(blue_mean)
        blue_stds.append(blue_std)
        names.append(fruit)

    data = {'name': names, 'red_mean': red_means, 'red_std': red_stds, 'green_mean': green_means, 
            'green_std': green_stds, 'blue_mean': blue_means, 'blue_std': blue_stds}
    df = pd.DataFrame(data)
    return df



path = 'img_data'#@param {type:"string"}
name_fruits = ['cachua', 'cam', 'chanh', 'cherry', 'dua luoi', 'man', 'mo', 'nho', 'quyt', 'tao']
# name_fruits = ['cachua', 'cam', 'duahau', 'le', 'nho', 'quyt', 'tao', 'thom', 'xoai']
dataset_train = {}
dataset_test = {}
test_size = 0.20 #@param {type:"number"}
number = 1
dim = (200, 200)
for name in name_fruits:
  index = 0
  images_train = []
  images_test = []
  while index < 250:
    img_path = path+'/'+name+'/'+name+' ('+str(index+1)+').jpg'
    index+=1
    #print(img_path)
    fruit_img = cv2.imread(img_path)
    fruit_img = cv2.resize(fruit_img, dim)
    fruit_img = cv2.cvtColor(fruit_img,cv2.COLOR_BGR2RGB)

    cluster_img, center_color= clustering_image(fruit_img)

    cluster_img = remove_light_color(cluster_img, center_color)
    
    if len(images_test) < test_size*(len(os.listdir(path+'/'+name))):
      images_test.append(cluster_img)
    else:
      images_train.append(cluster_img)
   
    #Hết 1 loại quả
    if number%250==0:
      print('done {} image'.format(number))
      # figure_size = 6
      # plt.figure(figsize=(figure_size,figure_size))

      # plt.subplot(1,2,1),plt.imshow(fruit_img)
      # plt.title('Ảnh gốc'), plt.xticks([]), plt.yticks([])

      # plt.subplot(1,2,2),plt.imshow(cluster_img)
      # plt.title(name), plt.xticks([]), plt.yticks([])

     
      # plt.show()

    number+=1
  dataset_train[name] = images_train
  dataset_test[name] = images_test
  
  
df_train = create_df_with_many_images(dataset_train)
df_test = create_df_with_many_images(dataset_test)

df_train.to_csv("color_features_train.csv", index=False)
# df_test.to_csv("color_features_test.csv", index=False)