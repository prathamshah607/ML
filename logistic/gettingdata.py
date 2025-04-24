import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading the training and testing data
original = pd.read_csv("Gotem Pumpkins.csv")
test_original = pd.read_csv("Freyja_Pumpkins.csv")

test_df = test_original.copy()
df = original.copy()


def map_p(type):
    if type == "Çerçevelik":
        return 1
    else: return 0


deets = [list(value) for value in test_df.drop("Class", axis=1).values] 
classes = [val for val in map(map_p, list(test_df['Class']))]

print(deets)
