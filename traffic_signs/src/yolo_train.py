from ultralytics import YOLO  # Import YOLO from ultralytics

def main():
    path_save_last_model = r"D:\STUDING_DENZA\4KYRS\TISPIS_KURSUCH\weight_colab\last (3).pt"
    model = YOLO(path_save_last_model)

    # Specifying paths to the data via YAML file
    yaml_path = r"D:\STUDING_DENZA\4KYRS\TISPIS_KURSUCH\working2\dataset\traffic_signs.yaml"

    save_dir = r"D:\STUDING_DENZA\4KYRS\TISPIS_KURSUCH\weight_colab\last_save"

    # Training the YOLOv8n model
    model.train(
        data=yaml_path,            # Path to the YAML file
        imgsz=630,                 # Image size
        batch=8,                   # Batch size
        epochs=6,                  # Number of epochs
        save=True,                 # Enable model saving
        project=save_dir,          # Directory path to save the model and logs
        name="yolov8",             # Name for saving the model
        exist_ok=True              # If the directory already exists, do not overwrite it
    )

if __name__ == "__main__":
    main()
