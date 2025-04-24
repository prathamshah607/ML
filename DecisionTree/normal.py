import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

original = pd.read_csv("Threats (1).csv")
data = original.copy()

unique_ids = {}

def unique_item_code (item):
    if item != "":
        sum_of_ords = sum([ord(val) for val in item])
        length = len(item)
        ord_first = ord(item[0])
        ord_last = ord(item[-1])
        unique_id = sum_of_ords+length+ord_first*2+ord_last
    else: unique_id = 1

    if unique_id not in unique_ids.keys():
        unique_ids[unique_id] = item

    return unique_id

def frequency_encoding(columns):
    global data
    for col in columns:
        data[col + '_freq'] = [unique_item_code(item) for item in data[col].values]
        data.drop(col, axis=1, inplace=True)

frequency_encoding(["proto", "service", "state", "attack_cat"])
titles = list(data)[1:-1]

kf = KFold(n_splits=10, shuffle=False)

accuracy_list = []
X= data[titles].values
y= data['attack_cat_freq'].values

clf = DecisionTreeClassifier(min_samples_leaf=int(len(data)/300))
bag = BaggingClassifier(clf)
m = RandomForestClassifier(oob_score=True)
m.fit(X, y)
rfscore = m.score(X, y)

bag.fit(X, y)
bagscore = bag.score(X, y)

print(rfscore)
print(bagscore)
