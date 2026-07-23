import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans

DEBUG_SHOW_MASKS = True  # flip to True to visually inspect each region as it's extracted


def to_pixel(landmark, w, h):
    """Normalized (0-1) mediapipe coordinate -> pixel coordinate."""
    return (int(landmark.x * w), int(landmark.y * h))


def linear_interpolation(p1, p2, t):
    """Linear interpolation between two pixel points, t in [0, 1]."""
    return (int(p1[0] + (p2[0] - p1[0]) * t), int(p1[1] + (p2[1] - p1[1]) * t))


def polygon_mask(image_shape, points):
    '''
    Create a polygon mask
    '''
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    return mask


def _bgr_to_all(bgr_triplet):
    '''
    Convert BGR color and return RGB and HSV with it.
    '''
    dom_bgr = np.uint8([[bgr_triplet]])
    dom_rgb = cv2.cvtColor(dom_bgr, cv2.COLOR_BGR2RGB)[0][0]
    dom_hsv = cv2.cvtColor(dom_bgr, cv2.COLOR_BGR2HSV)[0][0]
    return {
        "bgr": tuple(int(v) for v in dom_bgr[0][0]),
        "rgb": tuple(int(v) for v in dom_rgb),
        "hsv": tuple(int(v) for v in dom_hsv),
    }


def get_dominant_colors(image, mask, k=6, top_n=2, min_fraction=0.05, label=None):
    """
    Runs KMeans on the pixels under `mask` and returns up to `top_n` cluster
    colors, sorted by prevalence (largest first). Clusters covering less than
    `min_fraction` of the masked pixels are dropped as noise (anti-aliased
    edges, shadow gradients, stray background bleed, etc).

    Use k > top_n so there's room for KMeans to isolate noise into its own
    small cluster rather than blending it into a real color.

    Returns a list of dicts: [{"bgr", "rgb", "hsv", "fraction"}, ...]
    """
    pixels = image[mask == 255]
    if pixels.size == 0:
        return []

    if DEBUG_SHOW_MASKS:
        cv2.imshow(label or "region", cv2.bitwise_and(image, image, mask=mask))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    n_clusters = min(k, len(pixels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_
    counts = np.bincount(kmeans.labels_, minlength=n_clusters)
    total = counts.sum()

    order = np.argsort(-counts)  # largest cluster first
    results = []
    for idx in order:
        fraction = counts[idx] / total
        if fraction < min_fraction:
            continue
        entry = _bgr_to_all(colors[idx])
        entry["fraction"] = round(float(fraction), 4)
        results.append(entry)
        if len(results) >= top_n:
            break

    return results


def dominant_color(image, mask, k=3, label=None):
    """Convenience wrapper: single dominant color (backward-compatible)."""
    palette = get_dominant_colors(image, mask, k=k, top_n=1, min_fraction=0.0, label=label)
    return palette[0] if palette else None


def build_regions(landmarks, w, h):
    """
    Builds pixel-space polygons for each garment region from pose landmarks.
    Returns a dict of {region_name: [points]}.
    """
    L = mp.solutions.pose.PoseLandmark
    pt = lambda idx: to_pixel(landmarks[idx], w, h)

    ls, rs = pt(L.LEFT_SHOULDER), pt(L.RIGHT_SHOULDER)
    lh, rh = pt(L.LEFT_HIP), pt(L.RIGHT_HIP)
    lk, rk = pt(L.LEFT_KNEE), pt(L.RIGHT_KNEE)
    la, ra = pt(L.LEFT_ANKLE), pt(L.RIGHT_ANKLE)
    l_heel, r_heel = pt(L.LEFT_HEEL), pt(L.RIGHT_HEEL)
    l_toe, r_toe = pt(L.LEFT_FOOT_INDEX), pt(L.RIGHT_FOOT_INDEX)

    regions = {}

    # SHIRT: shoulders -> hips trapezoid
    regions["shirt"] = [ls, rs, rh, lh]

    # PANTS: hips -> ankles trapezoid, trimming 10% of the ankle to separate from shoes
    # for shorts, detect hem
    la_trim = linear_interpolation(lh, la, 0.9)
    ra_trim = linear_interpolation(rh, ra, 0.9)
    regions["pants"] = [lh, rh, ra_trim, la_trim]

    # SHOES: small polygon per foot using ankle, heel, and toe landmarks,
    # padded slightly since these three points alone form a thin sliver.
    def foot_polygon(ankle, heel, toe, pad=8):
        pts = np.array([ankle, heel, toe])
        center = pts.mean(axis=0)
        padded = [tuple((p + (p - center) / np.linalg.norm(p - center + 1e-6) * pad).astype(int))
                  for p in pts]
        return padded

    regions["left_shoe"] = foot_polygon(np.array(la), np.array(l_heel), np.array(l_toe))
    regions["right_shoe"] = foot_polygon(np.array(ra), np.array(r_heel), np.array(r_toe))

    return regions


def main():
    image = cv2.imread("person.jpg")
    if image is None:
        raise FileNotFoundError("Could not load person.jpg")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(rgb)

    if results.pose_landmarks is None:
        raise ValueError("No person detected.")

    landmarks = results.pose_landmarks.landmark
    regions = build_regions(landmarks, w, h)

    outfit_colors = {}
    for name, points in regions.items():
        mask = polygon_mask(image.shape, points)
        # k=6 gives KMeans room to isolate noise/edge pixels into their own
        # small cluster; top_n=2 + min_fraction filters those out and caps
        # the palette at the 3 most prevalent real colors.
        palette = get_dominant_colors(image, mask, k=6, top_n=2, min_fraction=0.05, label=name)
        outfit_colors[name] = palette
        print(f"{name}:")
        for color in palette:
            print(f"    {color}")

    return outfit_colors


if __name__ == "__main__":
    main()