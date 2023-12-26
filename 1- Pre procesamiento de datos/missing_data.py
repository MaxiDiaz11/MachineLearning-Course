# Plantilla de Pre Procesado
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Data.csv')

# variables dependientes X
# [:, :-1] -> todas las filas, todas las columnas menos la ultima
x = dataset.iloc[:, :-1].values

# variables independientes Y
y = dataset.iloc[:, 3].values  # [:, 3] -> todas las filas, columna 3

# Tratamiento de los NAs

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

imputer = imputer.fit(x[:, 1:3])  # [:, 1:3] -> todas las filas, columnas 1 y 2

x[:, 1:3] = imputer.transform(x[:, 1:3])
