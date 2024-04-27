import cv2

# Load pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to perform object detection on an image
def detect_objects(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw bounding boxes around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return image

# Function to process an image or video stream
def process_input(input_source):
    capture = cv2.VideoCapture(input_source)
    
    while True:
        # Read a frame from the input source
        ret, frame = capture.read()
        if not ret:
            break
        
        # Perform object detection on the frame
        detected_frame = detect_objects(frame)
        
        # Display the frame with detected objects
        cv2.imshow('Object Detection', detected_frame)
        
        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture object and close all OpenCV windows
    capture.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Path to the input image or video file
    input_source = 'path_to_image_or_video_file'
    
    # Process the input source
    process_input(input_source)

if __name__ == "__main__":
    main()
