#reading data from csv and convert to one hot encoding and normalizing data
import pandas as pd
import csv
train = pd.read_csv(args.train)
test = pd.read_csv(args.test)
val = pd.read_csv(args.val)
#change train_x,train_y and below columns according to your dataset
train_x = np.array(train.drop(columns=["id", "label"], axis=1))#remove columns according to your dataset. here trainx doesnot include 2 columns
train_y = np.array(train["label"]).reshape(55000, 1)#keeping trainy as label column
val_x = np.array(val.drop(columns=["id", "label"], axis=1))
val_y = np.array(val["label"]).reshape(5000, 1)
test_x = np.array(test.drop(columns=["id"], axis=1))
#convert data to onehot encoding
train_y_target = train_y.reshape(-1)
train_y_onehot = np.eye(10)[train_y_target]
val_y_target = val_y.reshape(-1)
val_y_onehot = np.eye(10)[val_y_target]
#normalize data
def normalize(x):
    a = 0
    b = 1
    x_max = 255
    x_min = np.amin(x)
    return ((x - x_min) * (b - a)) / (x_max - x_min)


