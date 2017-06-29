from keras.datasets import cifar10
import numpy as np 
(x_image_train,y_label_train),(x_image_test,y_label_test)=cifar10.load_data()

load_dict={0:"airplane", 1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}

import matplotlib.pyplot as plt
def plot_image_labels_prediction(images,labels,prediction,idx,num=10):
    fig=plt.gcf()
    fig.set_size_inches(12,14)
    if num>25: num=25
    for i in range(0,num):
        ax=plt.subplot(5,5,1+i)
        ax.imshow(images[idx],cmap='binary')
        title=str(i)+','+load_dict[labels[i][0]]
        if(len(prediction))>0:
            title+='=>'+load_dict[prediction[i]]
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()
plot_image_labels_prediction(x_image_train,y_label_train,[],0)