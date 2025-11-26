from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import mlflow
import pickle
import yaml
mlflow.set_tracking_uri("http://localhost:5000")

def main():
    with mlflow.start_run():
        train_data = pd.read_csv('data/processed/train.csv')
        val_data = pd.read_csv('data/processed/val.csv')
        with open('params.yaml', 'r') as file:
            loaded_data = yaml.safe_load(file)
        random_state = loaded_data['random_state']
        max_iter = loaded_data['max_iter']

        X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
        X_val, y_val = val_data.iloc[:, :-1], val_data.iloc[:, -1]

        model = LogisticRegression(random_state=random_state, max_iter=max_iter)
        model.fit(X_train, y_train)
        pickle.dump(model, open('model.pkl', 'wb'))

        preds = model.predict(X_val)
        accuracy = accuracy_score(y_val, preds)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_artifact("model.pkl")


if __name__ == "__main__":
    main()
