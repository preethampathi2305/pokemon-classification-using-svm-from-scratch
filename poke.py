#%%
import numpy as np
import os
from pathlib import Path
from keras_preprocessing import image
# %%
import pandas as pd
import tensorflow
train=pd.read_csv("train.csv")
trainfiles=train["ImageId"]
label_list=train["NameOfPokemon"]
train.head()

#%%
np.unique(label_list)
# %%
P=Path("Images/")

label_dict={"Pikachu":0,"Charmander":1,"Bulbasaur":2}
image_data=[]
for i in  range(len(trainfiles)):
    img=image.load_img("Images/"+trainfiles[i],target_size=(64,64))
    img_array=image.img_to_array(img)
    image_data.append(img_array)
    label_list[i]=(label_dict[label_list[i]])
# %%
len(image_data),len(label_list)
# %%
image_data=np.array(image_data,dtype='float32')/255.0
label_list=np.array(label_list)
print(image_data.shape,label_list.shape)
import random
combined=list(zip(image_data,label_list))
random.shuffle(combined)
image_data[:],label_list[:]=zip(*combined)
def drawimage(img):
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.show()
    return
drawimage(image_data[1])
# %%
class SVM:
    def __init__(self,c=1.0):
        self.c=c
        self.b=0
        self.w=0
    def hingeLoss(self,w,b,x,y):
        loss=0.0
        loss+=0.5*np.dot(w,w.T)
        n=x.shape[0]
        for i in range(n):
            ti=y[i]*(np.dot(w,x[i].T)+b)
            loss+=self.c*max(0,(1-ti))
        return loss
    def fit(self,x,y,batch_size=100,learning_rate=0.001,n_epochs=5):
        nfeatures=x.shape[1]
        nsamples=x.shape[0]
        c=self.c
        weights=np.zeros((1,nfeatures))
        bias=0
        #training loop:
        loss_list=[]
        for i in range(n_epochs):
            l=self.hingeLoss(weights,bias,x,y)
            loss_list.append(int(l))
            ids=np.arange(nsamples)
            np.random.shuffle(ids)
            #batch gradient descent:
            for batch_start in range(0,nsamples,batch_size):
                gradw=0
                gradb=0
                for j in range(batch_start,batch_start+batch_size):
                    if j<nsamples:#for samples which are not in multiples of batch size
                        i=ids[j]
                        ti=y[i]*(np.dot(weights,x[i].T)+bias)

                        if ti>1:
                            gradw+=0
                            gradb+=0
                        else:
                            gradw+=c*y[i]*x[i]
                            gradb+= c*y[i]
            weights=weights-learning_rate*weights+learning_rate*gradw
            bias=bias+learning_rate*gradb

        return weights,bias,loss_list
# %%
image_data=image_data.reshape(image_data.shape[0],-1)
image_data.shape,label_list.shape
# %%
def classWiseData(x,y):
    data={}
    for i in np.unique(label_list):
        data[i]=[]
    for i in range(x.shape[0]):
        data[y[i]].append(x[i])
    for k in data.keys():
        data[k]=np.array(data[k])
    return data
data=classWiseData(image_data,label_list)
data[0].shape
# %%
def preprocess(image_data,label_list):
    nsamples=len(image_data)
    nfeatures=image_data.shape[1]
    im=[]
    lb=[]
    for i in np.unique(label_list):
        data_pair=np.zeros((nsamples,nfeatures))
        data_labels=np.zeros((nsamples,))
        l1=len(np.where(label_list==i)[0])
        #l2=len(np.where(label_list!=i))
        data_pair[:l1]=image_data[np.where(label_list==i)]
        data_pair[l1:]=image_data[np.where(label_list!=i)]
        data_labels[:l1]=-1
        data_labels[l1:]=+1
        im.append(data_pair)
        lb.append(data_labels)
    return im,lb
im,lb=preprocess(image_data,label_list)
len(im),len(lb)
# %%
mysvm=SVM()
nclasses=len(np.unique(label_list))
def trainSVM1toall(x,y):
    svm_classifiers=[]
    for i in range(nclasses):
        weights,bias,loss=mysvm.fit(im[i],lb[i],learning_rate=0.0001,n_epochs=1000)
        svm_classifiers.append([weights,bias])
    return svm_classifiers
svm_classifiers=trainSVM1toall(image_data,label_list)
# %%
svm_classifiers
# %%
def binaryPredict(x,w,b):
    z=np.dot(x,w.T)+b
    if z>=0:
        return 1
    else:
        return -1
def predict(x):
    count=np.zeros((nclasses,))
    for i in range(nclasses):
        w,b=svm_classifiers[i]
        z=binaryPredict(x,w,b)
        if z==1:
            return i
    return np.argmax(count)
def accuracy(x,y):
    count=0
    for i in range(x.shape[0]):
        pred=predict(x[i])
        if pred==y[i]:
            count+=1
    return count/x.shape[0]
# %%
accuracy(image_data,label_list)
# %%
from sklearn import svm
svc=svm.SVC()
label_list=label_list.astype("int")
svc.fit(image_data,label_list)
svc.score(image_data,label_list)
# %%
from sklearn.neighbors import KNeighborsClassifier
lr=KNeighborsClassifier()
lr.fit(image_data,label_list)
lr.score(image_data,label_list)
# %%
from tensorflow.keras import  layers,models
model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(64,64,3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3))
# %%
import tensorflow
model.compile(optimizer='adam',
              loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(image_data[:200], label_list[:200], epochs=10, 
                    validation_data=(image_data[200:], label_list[200:]))

# %%
image_data[0][0][0][0]
# %%
import tensorflow as tf
label_list = tf.convert_to_tensor(label_list, dtype=tf.int64) 
image_data = tf.convert_to_tensor(image_data, dtype=tf.int64) 

# %%
train=pd.read_csv("test.csv")
trainfiles=train["ImageId"]
train.head()
image_data=[]
for i in  range(len(trainfiles)):
    img=image.load_img("Images/"+trainfiles[i],target_size=(64,64))
    img_array=image.img_to_array(img)
    image_data.append(img_array)
# %%
len(image_data)
image_data=np.array(image_data,dtype='float32')/255.0
for i in image_data[:10]:
    drawimage(i)
# %%

image_data=image_data.reshape(image_data.shape[0],-1)
image_data.shape
pre=[]
for i in image_data:
    pre.append(predict(i))
print(pre)
# %%
# %%
for i in range(len(pre)):
    if pre[i]==0:
        pre[i]="Pikachu"
    elif pre[i]==1:
        pre[i]="Charmander"
    elif pre[i]==2:
        pre[i]="Bulbasaur"
out=pd.DataFrame(trainfiles,columns=["ImageId"])
out["NameOfPokemon"]=pre
out.to_csv("out.csv",index=False)
# %%
type(pre[0])

# %%
