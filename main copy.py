import cv2
import mediapipe as mp
import RPi.GPIO as GPIO

# GPIO setup
red_led_pin = 17
green_led_pin = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(red_led_pin, GPIO.OUT)
GPIO.setup(green_led_pin, GPIO.OUT)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start capturing from the webcam
cap = cv2.VideoCapture(0)

# Define the threshold for open and closed hand
open_hand_threshold = 0.2

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two MediaPipe landmarks."""
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance

try:
    while True:
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert image to RGB
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)

        # Check if a hand is detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get landmarks for index fingertip and thumb tip
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                distance = calculate_distance(index_finger_tip, thumb_tip)

                # Determine hand state and control LEDs
                if distance < open_hand_threshold:
                    print(f"Distance: {distance:.4f} - Hand is CLOSED. Red LED ON.")
                    GPIO.output(red_led_pin, GPIO.HIGH)
                    GPIO.output(green_led_pin, GPIO.LOW)
                else:
                    print(f"Distance: {distance:.4f} - Hand is OPEN. Green LED ON.")
                    GPIO.output(green_led_pin, GPIO.HIGH)
                    GPIO.output(red_led_pin, GPIO.LOW)

                # Draw landmarks and annotate distance on the video
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                h, w, _ = image.shape
                cv2.putText(image, f"Distance: {distance:.4f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            print("No hand detected. Turning off LEDs.")
            GPIO.output(red_led_pin, GPIO.LOW)
            GPIO.output(green_led_pin, GPIO.LOW)

        # Display the video feed
        cv2.imshow("Debug Open Hand Threshold with LEDs", image)

        # Exit the loop when 'ESC' is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break
finally:
    # Release resources and clean up
    cap.release()
    GPIO.cleanup()
    cv2.destroyAllWindows()
