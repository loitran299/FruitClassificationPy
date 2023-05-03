import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd

from imageUtil import clustering_image, remove_light_color

simple_path = 'img_data/cachua'
for name in os.listdir(simple_path):
  img = cv2.imread(simple_path+'/'+name)
  resized = cv2.resize(img, (200, 200))
  resized = cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)

  cluster_img, center_color= clustering_image(resized)

  figure_size = 12
  plt.figure(figsize=(figure_size,figure_size))

  plt.subplot(1,3,1),plt.imshow(resized)
  plt.title('original image'), plt.xticks([]), plt.yticks([])

  plt.subplot(1,3,2),plt.imshow(cluster_img)
  plt.title('cluster image'), plt.xticks([]), plt.yticks([])

  final_img = remove_light_color(cluster_img, center_color)

  plt.subplot(1,3,3),plt.imshow(final_img)
  plt.title('image after remove color'), plt.xticks([]), plt.yticks([])
 
  plt.show()


#@title read and pre-processing image
# path = 'img_data'#@param {type:"string"}
# name_fruits = ['cachua', 'cam', 'duahau', 'le', 'nho', 'quyt', 'tao', 'thom', 'xoai']
# dataset_train = {}
# dataset_test = {}
# test_size = 0.20 #@param {type:"number"}
# number = 1
# dim = (200, 200)
# for name in name_fruits:
#   index = 0
#   images_train = []
#   images_test = []
#   while index < 250:
#     img_path = path+'/'+name+'/'+name+'_'+str(index)+'.jpg'
#     index+=1
#     #print(img_path)
#     fruit_img = cv2.imread(img_path)
#     fruit_img = cv2.resize(fruit_img, dim)
#     fruit_img = cv2.cvtColor(fruit_img,cv2.COLOR_BGR2RGB)

#     cluster_img, center_color= clustering_image(fruit_img)

#     cluster_img = remove_light_color(cluster_img, center_color)
    
#     if len(images_test) < test_size*(len(os.listdir(path+'/'+name))):
#       images_test.append(cluster_img)
#     else:
#       images_train.append(cluster_img)
   
#     if number%250==0:
#       print('done {} image'.format(number))
#       figure_size = 6
#       plt.figure(figsize=(figure_size,figure_size))

#       plt.subplot(1,2,1),plt.imshow(fruit_img)
#       plt.title('Original Image'), plt.xticks([]), plt.yticks([])

#       plt.subplot(1,2,2),plt.imshow(cluster_img)
#       plt.title(name), plt.xticks([]), plt.yticks([])

     
#       plt.show()

#     number+=1
#   dataset_train[name] = images_train
#   dataset_test[name] = images_test