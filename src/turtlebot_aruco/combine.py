#!/usr/bin/env python3

from matplotlib import pyplot as plt
import cv2
import numpy as np

def c(name):
    rows = 5
    columns = 5

    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(50,50))
    fig.tight_layout()

    i = 1

    for c in np.linspace(0, 2, num=5):
        for t in np.linspace(0, 1, num=5):
            p = f"output/C{-c}_T{t}_{name}.png"
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            fig.add_subplot(rows, columns, i)
            i += 1

            plt.imshow(img)
            plt.axis('off')
            plt.title(f"CHECK-IN COST: {c} | TRANSITION ERROR: {t}")

    fig.savefig(f"output/_COMBINED_{name}.png")

def combine():
    c("LEN")
    c("INDIFF")
    
            
if __name__=="__main__":
    combine()