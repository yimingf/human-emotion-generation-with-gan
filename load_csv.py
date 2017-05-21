import csv
import numpy as np

def load_csv():
    L=35887 #length of data
    y_dim=7
    labels=np.zeros(L)
    images_reshape=np.zeros((L,2304))
    with open("./data/fer2013/fer2013.csv","rb") as csvfile:
        reader = csv.DictReader(csvfile)
        count=0
        for row in reader:
            labels[count]=row['emotion']
            images_reshape[count]=(row['pixels'].split())
            count=count+1

    X=images_reshape.reshape((L,48,48,1)).astype(np.float)/255.0
    Y_=labels.astype(np.int)

    y_vec = np.zeros((len(Y_), y_dim), dtype=np.float)

    y_vec[np.arange(L), Y_] = 1.0

    return X,y_vec