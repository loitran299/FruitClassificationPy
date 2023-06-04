import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import pickle
from imageUtil import clustering_image, remove_light_color, features_grid, extract_color_with_one_image, extract_edge_with_one_image
import tkinter as tk
from tkinter import *
import PIL.Image, PIL.ImageTk
import cv2
from tkinter.filedialog import askopenfilename

# Load model
filename_rfc = "rfc_final_model"
rfc_model = pickle.load(open(filename_rfc, 'rb'))

class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        
        self.result = Label(master, bg="#E2F1E9")
        self.result.pack(anchor="ne", expand=YES)
        
    def filepicker(self):
        filename = askopenfilename()
        img = PIL.Image.open(filename)
        filename = img.filename.split('/')[-1]
        img.save(f"img/{filename}")
        imgShow = img.resize((200, 200))
        imgShow = PIL.ImageTk.PhotoImage(imgShow)
        self.result.configure(image=imgShow, justify="center")
        self.result.image = imgShow
        self.result.place(x=200, rely=0.1)
        
        path = "img/"+filename
        image = io.imread(path)
        
        fruitName = self.fruit_classification(image)[0]
        self.text = Label(self.master, text=fruitName, font=('Courier', 30))
        self.text.place(x=200, y=10)
        print(self.fruit_classification(image))
        

    def create_widgets(self):
        self.result = Label(self.master, bg="#B0BEC5", width=50, height=20)
        self.result.pack(anchor="ne", expand=YES)
        self.result.place(x=200, rely=0.1)
        
        self.btn_filepicker=Button(self.master, text="Chọn ảnh", width=10,bg="blue",fg="white",font="bold" , command=self.filepicker)
        self.btn_filepicker.place(relx=0.4, rely=0.7)
        
    def fruit_classification(self, img):
        resized = cv2.resize(img, (200, 200))
        clus, center = clustering_image(resized)
        img = remove_light_color(clus, center)
        edge = cv2.Canny(img, 50, 70)
        # tạo dataframe từ ảnh test
        data = extract_edge_with_one_image(edge)
        data2 = extract_color_with_one_image(img)
        
        final_data = pd.concat([data2, data],axis=1)
        
        return rfc_model.predict(final_data)
    
root = tk.Tk()
root.geometry("800x600")
app = App(master=root)
app.mainloop()