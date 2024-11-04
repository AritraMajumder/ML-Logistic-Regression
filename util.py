import numpy as np

def confusion_matrix(data,actual,model):
    pred = model.predict(data)
    bins = np.array([0,0.5,1])
    cm = np.histogram2d(actual, pred, bins=bins)[0]
    accuracy = (cm[0,0]+cm[1,1])/cm.sum()
    return cm, accuracy