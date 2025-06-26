import numpy as np
import pandas as pd

X_csv_path = r"csv"
y_csv_path = r"csv"


df_X = pd.read_csv(X_csv_path)
df_y = pd.read_csv(y_csv_path)


X_array = df_X.values
y_array = df_y.iloc[:, 0].values


np.save( r".npy", X_array)
np.save( r".npy", y_array)


