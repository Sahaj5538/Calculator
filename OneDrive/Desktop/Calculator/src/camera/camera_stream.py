import subprocess
import sys

# Auto install OpenCV if not installed
try:
    import cv2
except ImportError:
    print("OpenCV not found. Installing opencv-python...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
    import cv2


def start_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access camera")
        return

    print("Press ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally (fix mirrored camera)
        frame = cv2.flip(frame, 1)

        cv2.imshow("Air Canvas Camera", frame)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera()