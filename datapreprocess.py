import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
data=pd.read_csv(url)
data.describe().T
data.head()	
data.info()
train_data, val_data = train_test_split(data, test_size=0.2, random_state=32)
imputer = SimpleImputer(fill_value=np.nan, startegy='mean')
X = imputer.fit_transform(data)
missing_values = data.isnull().sum()
data['column_name'].fillna(data['column_name'].mean(), inplace=True)

categorical_features = data.select_dtypes(include=['object']).columns
encoded_data = pd.get_dummies(data, columns=categorical_features)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

normalized_data = scaler.fit_transform(data)