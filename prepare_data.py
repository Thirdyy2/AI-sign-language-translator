# prepare_data.py
import glob, pandas as pd, numpy as np, pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

files = glob.glob("landmark_data/*.csv")
df_list = []
for f in files:
    df = pd.read_csv(f)
    df_list.append(df)
big = pd.concat(df_list, ignore_index=True)
print("Total samples:", len(big))

# last col is label
X = big.iloc[:, :-1].values.astype('float32')
y = big.iloc[:, -1].values
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Classes:", list(le.classes_))

# split train/val
X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)

# save arrays & label encoder
np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)
pickle.dump(le, open("label_encoder.pkl","wb"))
print("Saved X_train.npy etc. shapes:", X_train.shape, X_val.shape)
