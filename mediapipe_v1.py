import cv2
import mediapipe as mp

print(mp.__file__)
print(mp.__version__)
print(dir(mp))
print(dir(mp.tasks.vision))

image = cv2.imread("person.jpg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

results = pose.process(rgb)

landmarks = results.pose_landmarks.landmark