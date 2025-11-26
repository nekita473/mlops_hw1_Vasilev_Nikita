from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import yaml


def main():
    data = pd.read_csv('data/raw/data.csv')
    with open('params.yaml', 'r') as file:
        loaded_data = yaml.safe_load(file)
    test_size = loaded_data['test_size']
    random_state = loaded_data['random_state']

    le = LabelEncoder()
    data['Species'] = le.fit_transform(data['Species'].values)

    train_data, val_data = train_test_split(data, test_size=test_size,
                        stratify=data['Species'], random_state=random_state)

    os.makedirs('data/processed', exist_ok=True)
    train_data.to_csv('data/processed/train.csv', index=False)
    val_data.to_csv('data/processed/val.csv', index=False)


if __name__ == "__main__":
    main()
