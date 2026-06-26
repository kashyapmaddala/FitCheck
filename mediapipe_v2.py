import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans

def main():
    image = cv2.imread("person.jpg")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(rgb)

    if results.pose_landmarks is None:
        raise ValueError("No person detected.")

    print(type(results))
    # print(type(results.pose_landmarks))
    # print(type(results.pose_landmarks[0]))
    # print(type(results.pose_landmarks[0][0]))

    # print(type(results.pose_landmarks))
    # print(type(results.pose_landmarks.landmark))
    # print(type(results.pose_landmarks.landmark[0]))

    landmarks = results.pose_landmarks.landmark

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]


    h, w, _ = image.shape

    ## Converts a normalised coordiante (0 to 1) into a pixel number nased on image size
    def to_pixel(landmark):
        return (int(landmark.x * w), int(landmark.y * h))

    # ## SHIRT REGION
    ls = to_pixel(left_shoulder)
    rs = to_pixel(right_shoulder)
    lh = to_pixel(left_hip)
    rh = to_pixel(right_hip)

    ## ISOLATING THE SHIRT
    shirt_points = np.array([ls, rs, rh, lh], dtype=np.int32)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [shirt_points], 255)

    shirt = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("Shirt", shirt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    shirt_pixels = image[mask == 255]

    ## Finding dominant colour via clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(shirt_pixels)
    colors = kmeans.cluster_centers_

    counts = np.bincount(kmeans.labels_)
    dominant_color = colors[np.argmax(counts)]
    dominant_color = np.uint8([[dominant_color]])
    print(dominant_color)

    rgb = cv2.cvtColor(dominant_color, cv2.COLOR_BGR2RGB)
    print(rgb)

    hsv = cv2.cvtColor(dominant_color, cv2.COLOR_BGR2HSV)
    print(hsv)
    
if __name__ == '__main__':
   main() 

