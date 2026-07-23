import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import cv2
import mediapipe as mp
import numpy as np
from sklearn.cluster import KMeans

DEBUG_SHOW_MASKS = True  # flip to True to visually inspect each region as it's extracted

def to_pixel(landmark, w, h):
    """
    Converts a normalized (0-1) mediapipe coordinate into pixel coordinate for a landmark.
    It uses the width and height of the image.
    """
    return (int(landmark.x * w), int(landmark.y * h))


def linear_interpolation(p1, p2, t):
    """Linear interpolation between two pixel points, t in [0, 1]."""
    return (int(p1[0] + (p2[0] - p1[0]) * t), int(p1[1] + (p2[1] - p1[1]) * t))


def polygon_mask(image_shape, polygons):
    '''
    Create a mask from one polygon or a list of polygons (unioned together).
 
    `polygons` is either:
      - a single polygon: [(x, y), (x, y), ...]
      - a list of polygons: [[(x, y), ...], [(x, y), ...], ...]
 
    We tell the two apart by checking what the first element is: if it's a
    tuple/point, we got a single polygon; if it's itself a list, we got
    multiple polygons to union.
    '''
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    is_single_polygon = len(polygons) > 0 and isinstance(polygons[0], tuple)
    poly_list = [polygons] if is_single_polygon else polygons
    for poly in poly_list:
        cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 255)
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


def get_dominant_colors(image, mask, k=6, top_n=3, min_fraction=0.05, label=None):
    """
    Runs KMeans on the pixels under the given `mask` and returns up to `top_n` cluster colors.
    These clusters are sorted by prevalence (largest/most frequent colors first).
    Clusters covering less than `min_fraction` of the masked pixels are dropped as noise.
    Examples of noise include anti-aliased edges, shadow gradients, stray background bleed, etc.

    Use k > top_n so there's room for KMeans to isolate noise into its own
    small cluster rather than blending it into a real color.

    Returns a list of dicts for each of the top_n colors: [{"bgr", "rgb", "hsv", "fraction"}, ...]
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
    le, re = pt(L.LEFT_ELBOW), pt(L.RIGHT_ELBOW)
    lh, rh = pt(L.LEFT_HIP), pt(L.RIGHT_HIP)
    lk, rk = pt(L.LEFT_KNEE), pt(L.RIGHT_KNEE)
    la, ra = pt(L.LEFT_ANKLE), pt(L.RIGHT_ANKLE)
    l_heel, r_heel = pt(L.LEFT_HEEL), pt(L.RIGHT_HEEL)
    l_toe, r_toe = pt(L.LEFT_FOOT_INDEX), pt(L.RIGHT_FOOT_INDEX)

    regions = {}

    # Generic limb polygon builder that walks an arbitrary path of landmarks
    # (2 points for a straight limb segment like shoulder->elbow, 3+ for a bending one like hip->knee->ankle)
    # offsets perpendicular to the path at each joint, tapering width along the way.
    def seg_normal(a, b):
        d = np.array(b, dtype=float) - np.array(a, dtype=float)
        norm = np.linalg.norm(d)
        if norm == 0:
            return np.array([1.0, 0.0])
        return np.array([-d[1], d[0]]) / norm


    def limb_polygon(points, half_widths):
        """
        points: [(x, y), ...] path along the limb, in order (e.g. shoulder, elbow).
        half_widths: half-width of the limb at each corresponding point


        At interior joints, the offset direction is the average of the two adjacent segment normals (a "miter" join).
        This way, the polygon doesn't kink visibly where two segments meet, e.g. at the knee or elbow.
        
        Half widths are provided so that the final segmentation is more reflective of the actual figure.
        """
        pts = [np.array(p, dtype=float) for p in points]
        n = len(pts)
        offsets = []
        for i in range(n):
            if i == 0:
                normal = seg_normal(pts[0], pts[1])
            elif i == n - 1:
                normal = seg_normal(pts[-2], pts[-1])
            else:
                n1 = seg_normal(pts[i - 1], pts[i])
                n2 = seg_normal(pts[i], pts[i + 1])
                summed = n1 + n2
                norm = np.linalg.norm(summed)
                normal = summed / norm if norm != 0 else n1
            offsets.append(normal * half_widths[i])
 
        left_side = [pts[i] + offsets[i] for i in range(n)]
        right_side = [pts[i] - offsets[i] for i in range(n - 1, -1, -1)]
        return [tuple(p.astype(int)) for p in left_side + right_side]
    
    def interp_widths(start_w, end_w, n):
        if n == 1:
            return [start_w]
        return [start_w + (end_w - start_w) * i / (n - 1) for i in range(n)]

    # SHIRT: shoulders -> elbow + torso trapezoid
    shoulder_width = np.linalg.norm(np.array(ls) - np.array(rs))
    sleeve_hw_shoulder = shoulder_width * 0.22
    sleeve_hw_elbow = shoulder_width * 0.16
    left_sleeve = limb_polygon([ls, le], interp_widths(sleeve_hw_shoulder, sleeve_hw_elbow, 2))
    right_sleeve = limb_polygon([rs, re], interp_widths(sleeve_hw_shoulder, sleeve_hw_elbow, 2))
    regions["shirt"] = [[ls, rs, rh, lh], left_sleeve, right_sleeve]  # list of polygons, unioned in the mask


    # PANTS: two separate bent polygons (one per leg), routed hip -> knee -> ankle
    hip_width = np.linalg.norm(np.array(lh) - np.array(rh))
    leg_hw_hip = hip_width * 0.35
    leg_hw_ankle = leg_hw_hip * 0.5
    regions["pants_left"] = limb_polygon([lh, lk, la], interp_widths(leg_hw_hip, leg_hw_ankle, 3))
    regions["pants_right"] = limb_polygon([rh, rk, ra], interp_widths(leg_hw_hip, leg_hw_ankle, 3))


    # SHOES: small polygon per foot using ankle, heel, and toe landmarks,
    # padded slightly since these three points alone form a thin sliver.
    def foot_polygon(ankle, heel, toe, pad=8):
        pts = np.array([ankle, heel, toe])
        center = pts.mean(axis=0)
        padded = [tuple((p + (p - center) / np.linalg.norm(p - center + 1e-6) * pad).astype(int)) for p in pts]
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