from ultralytics import YOLO

def main():
    # Пути к файлам моделей и yaml-файлу с набором данных
    model1_path = r"D:\STUDING_DENZA\4KYRS\TISPIS_KURSUCH\weight_colab\best.pt"
    model2_path = r"D:\STUDING_DENZA\4KYRS\TISPIS_KURSUCH\weight_colab\best (3).pt"
    model3_path = r"D:\STUDING_DENZA\4KYRS\TISPIS_KURSUCH\weight_colab\last_save\yolov8\weights\best.pt"
    dataset_yaml = r"D:\STUDING_DENZA\4KYRS\TISPIS_KURSUCH\working2\dataset\traffic_signs.yaml"

    # Загрузка моделей
    model1 = YOLO(model1_path)
    model2 = YOLO(model2_path)
    model3 = YOLO(model3_path)

    # Оценка моделей
    print("Evaluating Model 1...")
    results1 = model1.val(data=dataset_yaml)
    print("Model 1 Metrics:", results1.box.map50, results1.box.map)

    print("Evaluating Model 2...")
    results2 = model2.val(data=dataset_yaml)
    print("Model 2 Metrics:", results2.box.map50, results2.box.map)

    print("Evaluating Model 3...")
    results3 = model3.val(data=dataset_yaml)
    print("Model 3 Metrics:", results3.box.map50, results3.box.map)

    # Сравнение метрик
    best_map50 = max(results1.box.map50, results2.box.map50, results3.box.map50)
    best_map = max(results1.box.map, results2.box.map, results3.box.map)

    if best_map50 == results1.box.map50:
        print("Model 1 is the best based on mAP@50.")
    elif best_map50 == results2.box.map50:
        print("Model 2 is the best based on mAP@50.")
    else:
        print("Model 3 is the best based on mAP@50.")
    
    if best_map == results1.box.map:
        print("Model 1 is the best based on mAP.")
    elif best_map == results2.box.map:
        print("Model 2 is the best based on mAP.")
    else:
        print("Model 3 is the best based on mAP.")

if __name__ == "__main__":
    main()
