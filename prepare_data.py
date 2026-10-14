import sys
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris, load_wine

def save_frame(frame: pd.DataFrame, path: Path) -> None:
    # Создаем папку data, если её нет
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Детерминированная сортировка по всем признакам и сброс индекса
    ordered = frame.sort_values(list(frame.columns), kind="mergesort").reset_index(drop=True)
    
    # Сохраняем без индекса, с одинаковыми параметрами для всех версий
    ordered.to_csv(path, index=False, lineterminator="\n", float_format="%.10g", encoding="utf-8")

def main():
    if len(sys.argv) < 2:
        print("Ошибка: укажите режим (base, wine или iris_features)")
        sys.exit(1)
        
    mode = sys.argv[1]
    data_dir = Path("data")

    if mode == "base":
        # Сохраняем только Iris
        iris = load_iris(as_frame=True).frame
        save_frame(iris, data_dir / "iris.csv")
        print("Создан data/iris.csv (режим base)")
        
    elif mode == "wine":
        # Добавляем Wine, не изменяя Iris
        wine = load_wine(as_frame=True).frame
        save_frame(wine, data_dir / "wine.csv")
        print("Создан data/wine.csv (режим wine)")
        
    elif mode == "iris_features":
        # Изменяем Iris, добавляя вычисляемые столбцы
        iris = load_iris(as_frame=True).frame
        iris["sepal_area"] = iris["sepal length (cm)"] * iris["sepal width (cm)"]
        iris["petal_area"] = iris["petal length (cm)"] * iris["petal width (cm)"]
        save_frame(iris, data_dir / "iris.csv")
        print("Изменен data/iris.csv (режим iris_features)")
        
    else:
        print(f"Неизвестный режим: {mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()