from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def confusion1():
    data_set_true = pd.read_csv("ids-sst-dev.csv", sep ='\t')
    data_set_pred = pd.read_csv("sst-dev-output-best-copy.csv", sep='\t')
    data_set_pred[['id', 'sentiment']] = data_set_pred['Unnamed: 0'].str.split(',', expand=True)
    y_true = data_set_true["sentiment"]
    y_true = y_true.astype(int)
    y_pred = data_set_pred["sentiment"]
    y_pred = y_pred.astype(int)
    cm = confusion_matrix(y_true, y_pred, normalize = 'true')
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.savefig('confusion_matrix_sst.png')
    #print(y_pred.head(5))
    






def main():
    confusion1()
    


if __name__ == "__main__":
    main()