from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv('data/raw/data.csv')
le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'].values)
train_data, val_data = train_test_split(data, test_size=0.2,
 stratify=data['Species'], random_state=42)
train_data.to_csv('data/processed/train.csv', index=False)
val_data.to_csv('data/processed/val.csv', index=False)
