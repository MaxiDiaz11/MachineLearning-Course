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

# Codificar datos categoricos

labelencoder_x = LabelEncoder()

x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

# Crear variables dummy

# One hot encoding X
ct = ColumnTransformer(
    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
    # Leave the rest of the columns untouched
    remainder='passthrough'
)

x = np.array(ct.fit_transform(x))

# One hot encoding Y

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

# Dividir el data set en conjunto de entrenamiento y conjunto de testing


# test_size = 0.2 -> 20% de los datos para testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

# Escalado de variables

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)
