
import pandas as pd
import numpy as np
import cv2
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

x_train_class_1 = pd.read_excel('Training_Task_Alex_201904/BP3_Control_Raw_Data_Double_Clicks.xlsx',sheet_name='Gyro_x')
y_train_class_1 = pd.read_excel('Training_Task_Alex_201904/BP3_Control_Raw_Data_Double_Clicks.xlsx',sheet_name='Gyro_y')
z_train_class_1 = pd.read_excel('Training_Task_Alex_201904/BP3_Control_Raw_Data_Double_Clicks.xlsx',sheet_name='Gyro_z')
train_class_1_label = pd.read_excel('Training_Task_Alex_201904/BP3_Control_Raw_Data_Double_Clicks.xlsx',sheet_name='Model_2_Label')

train_class_1_label = train_class_1_label.dropna(subset=['Start'])
idx = pd.Index(train_class_1_label['No.'])

label = []

r1 = []
count_1 = 0
for i in idx - 1:
    s = train_class_1_label.loc[i,'Start'].astype(int)
    e = train_class_1_label.loc[i,'End'].astype(int)

    bondary = pd.Series(range(int(s),int(e)))
    x_record = x_train_class_1.loc[i,bondary]
    y_record = y_train_class_1.loc[i,bondary]
    z_record = z_train_class_1.loc[i,bondary]
    record = np.dstack((x_record,y_record,z_record))
    record = np.resize(record,(1,24,3))
    label.append(1)
    if count_1 == 0 :
        r1 = record
    else:
        r1 = np.concatenate((r1, record))
    count_1 += 1


x_train_class_2 = pd.read_excel('Training_Task_Alex_201904/BP3_Raw_Data_TripleClicks.xlsx',sheet_name='Gyro_X')
y_train_class_2 = pd.read_excel('Training_Task_Alex_201904/BP3_Raw_Data_TripleClicks.xlsx',sheet_name='Gyro_Y')
z_train_class_2 = pd.read_excel('Training_Task_Alex_201904/BP3_Raw_Data_TripleClicks.xlsx',sheet_name='Gyro_Z')
train_class_2_label = pd.read_excel('Training_Task_Alex_201904/BP3_Raw_Data_TripleClicks.xlsx',sheet_name='model_Label_2')

train_class_2_label = train_class_2_label.dropna(subset=['Start Frame'])
idx = pd.Index(train_class_2_label['No.'])

r2 = []
count = 0
for i in idx - 1:
    s = train_class_2_label.loc[i,'Start Frame'].astype(int)
    e = train_class_2_label.loc[i,'End Frame'].astype(int)
    boundary = pd.Series(range(s,e))
    x_record = x_train_class_2.loc[i,bondary]
    y_record = y_train_class_2.loc[i,bondary]
    z_record = z_train_class_2.loc[i,bondary]
    record = np.dstack((x_record,y_record,z_record))
    record = np.resize(record,(1,24,3))
    label.append(0)
    if count == 0 :
        r2 = record
    else:
        r2 = np.concatenate((r2, record))
    count += 1

train_input = np.concatenate((r1, r2))
label=np.array(label)
print(train_input.shape)
print(label.shape)


# split into input (X) and output (Y) variables
#X = dataset[:,0:8]
#Y = dataset[:,8]

#normalize the data
#scaler = Normalizer().fit(X)
#X = scaler.transform(X)

#label = to_categorical(label)
traindata, testdata, trainlabel, testlabel = train_test_split(train_input,label, test_size=0.33, random_state=42)
expected = testlabel

