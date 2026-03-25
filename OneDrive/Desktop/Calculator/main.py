import cv2
import sys
import os

# Add the project root to the sys path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.hand_tracking.hand_detector import HandDetector

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot access camera")
        return

    # Initialize the hand detector
    detector = HandDetector(maxHands=1, detectionCon=0.7, trackCon=0.7)

    print("Starting Main App: Hand Tracking")
    print("Press ESC to exit")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally to fix the mirroring
        img = cv2.flip(img, 1)

        # Find hands and draw the skeleton!
        img = detector.findHands(img, draw=True)

        # Optionally get landmark list and check fingers
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            fingers = detector.fingersUp(lmList)
            # Display which fingers are up
            cv2.putText(img, f'Fingers Up: {sum(fingers)}', (10, 50), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Show the camera feed with hand-tracking skeleton
        cv2.imshow("Air Canvas Calculator", img)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
