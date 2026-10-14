import hashlib
import sys
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris, load_wine

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main():
    data_dir = Path("data")
    if not data_dir.exists():
        print("Ошибка: директория data отсутствует.")
        sys.exit(1)

    # Ожидаемое количество строк (из scikit-learn)
    expected_rows = {
        "iris.csv": len(load_iris(as_frame=True).frame),
        "wine.csv": len(load_wine(as_frame=True).frame)
    }

    for path in sorted(data_dir.glob("*.csv")):
        # Читаем CSV с явным указанием разделителя и кодировки для избежания ParserError и UnicodeDecodeError
        frame = pd.read_csv(path, sep=",", encoding="utf-8")
        
        # 1. Проверка безымянного столбца индекса
        if "Unnamed: 0" in frame.columns:
            print(f"Ошибка: найден безымянный столбец индекса в {path.name}")
            sys.exit(1)
            
        rows, cols = frame.shape
        
        # 2. Проверка количества строк
        if path.name in expected_rows and rows != expected_rows[path.name]:
            print(f"Ошибка: количество строк в {path.name} ({rows}) отличается от исходного Dataset!")
            sys.exit(1)
            
        # 3. Проверка значений новых столбцов (для iris_features)
        if path.name == "iris.csv" and "sepal_area" in frame.columns:
            expected_sepal = frame["sepal length (cm)"] * frame["sepal width (cm)"]
            if not (frame["sepal_area"] == expected_sepal).all():
                print("Ошибка: неправильные значения в столбце sepal_area")
                sys.exit(1)

        missing = int(frame.isna().sum().sum())
        
        # Вывод характеристик
        print(f"Файл: {path.relative_to('.')}")
        print(f"SHA-256: {sha256_file(path)}")
        print(f"Строк: {rows}, Столбцов: {cols}")
        print(f"Пропусков: {missing}")
        print(f"Столбцы: {list(frame.columns)}\n")

if __name__ == "__main__":
    main()