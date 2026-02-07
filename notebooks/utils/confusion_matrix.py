import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_test, y_pred):
    """
    Genera y muestra una matriz de confusión personalizada
    con la disposición [[TP, FN], [FP, TN]].
    """

    cm = confusion_matrix(y_test, y_pred)

    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]

    cm_custom = np.array([
        [TP, FN],
        [FP, TN]
    ])

    cm_df = pd.DataFrame(
        cm_custom,
        index=['Pred Positivo', 'Pred Negativo'],
        columns=['Real Positivo', 'Real Negativo']
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_custom,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Real +', 'Real -'],
        yticklabels=['Pred +', 'Pred -']
    )

    plt.xlabel('Clase real')
    plt.ylabel('Clase predicha')
    plt.title('Matriz de Confusión (TP FN / FP TN)')
    plt.tight_layout()
    plt.show()