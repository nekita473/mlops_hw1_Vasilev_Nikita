# Проект по использованию dvc и mlflow

Запуск mflow server:
```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Запуск сборки данных и обучения модели:
```
git clone <this repo>
cd mlops_hw1_Vasilev_Nikita
pip install -r requirements.txt
dvc pull
dvc repro -f
```
Так как данные не меняются, для принудительно запуска используется ключ ```-f```

В mlflow появится эксперимент с точностью 
```0.8666666666666667```

### Публичный read-only ключ для доступа на Yandex Cloud Backet добавлен умышленно для воспроизводимости запуска.