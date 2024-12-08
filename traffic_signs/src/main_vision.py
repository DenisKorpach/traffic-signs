from ultralytics import YOLO
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def detect_on_video(video_path, model_path):
    # Load the trained model
    model = YOLO(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video:", video_path)
        return

    # Get video parameters
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Prepare for saving the processed video
    output_path = video_path.replace(".mp4", "_output.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply the YOLO model for object detection
        results = model(frame)

        # Draw detected objects on the frame
        annotated_frame = results[0].plot()  # Apply results to the image
        
        # Display the processed frame
        cv2.imshow("Detections", annotated_frame)
        
        # Save the processed frame to the video
        out.write(annotated_frame)
        
        # Exit when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("Processed video saved to:", output_path)

def select_file():
    # Dialog for selecting an image
    Tk().withdraw()  # Hide the main tkinter window
    file_path = askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    print(f"Selected file: {file_path}")  # Added for debugging
    return file_path

if __name__ == '__main__':
    # Paths to video files
    video_day = r"D:\STUDING_DENZA\4KYRS\ОиВИС\laba3\signs\Брест день.mp4"
    video_night = r"D:\STUDING_DENZA\4KYRS\ОиВИС\laba3\signs\Брест ночь.mp4"
    
    # Path to the best model weights
    model_weights = r"D:\STUDING_DENZA\4KYRS\TISPIS_KURSUCH\weight_colab\last_save\yolov8\weights\best.pt"
    
    # User choice for analysis type
    print("Choose the type of analysis:")
    print("1. Video (Brest Day)")
    print("2. Video (Brest Night)")
    choice = input("Enter your choice number: ")
    
    # Processing user choice
    if choice == '1':
        detect_on_video(video_day, model_weights)
    elif choice == '2':
        detect_on_video(video_night, model_weights)
    else:
        print("Invalid choice.")
