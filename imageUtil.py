import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd


# **Clustering ảnh**
# Function INPUT: ảnh 200x200
# Function OUTPUT: ảnh chỉ có 2 màu sau khi được cluster
#hàm này thực hiện phân đoạn ảnh, chia một hình ảnh thành hai cụm dựa trên sự tương đồng của giá trị pixel. Nó có thể được sử dụng cho các nhiệm vụ như nhận dạng đối tượng
def clustering_image(img):
    # Hình ảnh được chuyển đổi thành một vector có kích thước (num_pixels, 3), trong đó num_pixels là tổng số pixel trong hình ảnh, và mỗi pixel được đại diện bởi một vector 3 chiều (giá trị R, G, B).
    vectorized = img.reshape((-1,3))
    #Vector hóa ảnh được chuyển đổi sang kiểu dữ liệu float32.
    vectorized = np.float32(vectorized)
    
    #Đây là một tuple được sử dụng để thiết lập các tiêu chí dừng cho giải thuật phân cụm K-means.
    #Các giá trị còn lại của tuple là 10 và 1.0, lần lượt là số lần lặp tối đa và ngưỡng chấp nhận được cho sự thay đổi giữa trung tâm cụm hiện tại và trung tâm cụm trước đó. Trong trường hợp này, giải thuật sẽ dừng lại sau 10 lần lặp hoặc nếu sự thay đổi giữa trung tâm cụm hiện tại và trung tâm cụm trước đó nhỏ hơn 1.0.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # phân 2 cụm tương ứng với xác định đầu ra ảnh chỉ có 2 màu
    K = 2
    
    #Tham số attempts chỉ định số lần giải thuật được chạy với các giá trị trọng tâm khởi tạo khác nhau.
    attempts=10
    
    #ret là tổng khoảng cách bình phương từ các điểm dữ liệu đến trung tâm của cụm tương ứng.
    #label là danh sách các nhãn cho mỗi điểm dữ liệu.
    #Mảng center chứa các giá trị RGB của trọng tâm của hai cụm.
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    #Mảng center được chuyển đổi sang kiểu dữ liệu uint8.
    center = np.uint8(center)
    #Mảng res là một phiên bản vector hóa của hình ảnh đầu vào, trong đó mỗi pixel được gán giá trị RGB của trọng tâm được gán cho nó.
    res = center[label.flatten()]
    #Mảng res được chuyển đổi lại thành một hình ảnh có cùng kích thước với hình ảnh đầu vào.
    result_image = res.reshape((img.shape))
    return result_image, center


# **Loại bỏ màu sáng**
# Function INPUT: ảnh có 2 màu được cluster, 2 màu của ảnh đó
# Function OUTPUT: ảnh chỉ còn 1 màu (màu tối hơn)
def remove_light_color(img, center_color):
    #Trong hàm này, màu sáng nhất được xác định bằng cách tính tổng giá trị màu của hai trung tâm cụm.
    # Sau đó, hàm sẽ duyệt qua tất cả các điểm ảnh của ảnh đầu vào và nếu tổng giá trị màu của một điểm ảnh bằng với giá trị màu sáng nhất,
    # thì giá trị màu của điểm ảnh đó sẽ được đặt bằng 0 (đen). Cuối cùng, hàm trả về ảnh đã được xử lý.
    light_color = max(sum(center_color[0]), sum(center_color[1]))
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if sum(img[x][y]) == light_color:
                img[x][y][0]=0
                img[x][y][1]=0
                img[x][y][2]=0
    return img


#Defining hàm để chia nhỏ bức ảnh 200x200 pixels thành từng phần nhỏ 40x40 pixels, và tính trung bình, độ lệch chuẩn của từng phần.
#Function INPUT: 1 cảnh kích cỡ 200x200
#Function OUTPUT: mảng feature bao gồm trung bình cộng và độ lệch chuẩn của 25 phần (40x40 pixel)
def features_grid(img):
    features = np.array([], dtype='uint8')
    section = 1
    
    for y in range(0, img.shape[0], 40):
        for x in range(0, img.shape[1], 40):

            # Cropping the image into a section.
            section_img = img[y:y+40, x:x+40]
            
            # Claculating the mean and stdev of the sectioned image.
            section_mean = np.mean(section_img)
            section_std = np.std(section_img)
            
            # Appending the above calculated values into features array.
            features = np.append(features, [section_mean, section_std])
    
    # Returning the features array.
    return features

# Tìm cạnh
def Canny_Detect(img):
    edge = cv2.Canny (img, 50, 70)
    return edge

def extract_edge_with_one_image(img):
    feature_name = []
    section = 1
    for y in range(0, 200, 40):
        for x in range(0, 200, 40):
            feature_name.append(f"sec{section}_mean")
            feature_name.append(f"sec{section}_std")
            section += 1
    
    all_imgs = np.zeros((1, 50), dtype='uint8')

    name = []
    section_mean = []
    section_std = []

    img_features = features_grid(img)
    img_features = img_features.reshape(1, img_features.shape[0])

    all_imgs = np.append(all_imgs, img_features, axis=0)
    all_imgs = all_imgs[1:]

    df = pd.DataFrame(all_imgs, columns= feature_name)
    return df


def extract_color_with_one_image(img):
    columns_name = ['red_mean', 'red_std', 'green_mean', 'green_std', 'blue_mean', 'blue_std']
    red_means = []
    red_stds = []
    green_means = []
    green_stds = []
    blue_means = []
    blue_stds = []

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


    data = {'red_mean': red_means, 'red_std': red_stds, 'green_mean': green_means, 
            'green_std': green_stds, 'blue_mean': blue_means, 'blue_std': blue_stds}
    df = pd.DataFrame(data)
    return df