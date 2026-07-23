"""
outfit_scoring.py

Two of the three components of the outfit score:
  1. Saturation / "eye-popping" score        (weight ~30%)
  2. Color-harmony score (comp/analog/triad) (weight ~15%)

The third component, a Markov-chain trend score, plugs into the same
interface (`compute_trend_score`) once we've settled on a data source -
see the TODO near the bottom.

Expects the `outfit_colors` dict produced by mediapipe_v4.py's main():
    {
        "shirt": [{"bgr":..., "rgb":..., "hsv": (h, s, v), "fraction": 0.62}, ...],
        "pants_left": [...],
        "pants_right": [...],
        "left_shoe": [...],
        "right_shoe": [...],
    }
where hsv is OpenCV-style: H in [0,179], S in [0,255], V in [0,255].
Each region's list is already sorted by prevalence (most dominant color first).
"""

import math

# ----------------------------------------------------------------------
# 1. SATURATION / "EYE-POPPING" SCORE
# ----------------------------------------------------------------------

# How much each region counts toward the saturation score. Pants and shoes matter more than the shirt.
# These are relative to each other, so pants are 44% more sensitive to saturation than shoes
# Shoes are 50% more sensitive than shirts.
REGION_SAT_WEIGHTS = {
    "shirt": 0.6,
    "pants_left": 1.3,
    "pants_right": 1.3,
    "left_shoe": 0.9,
    "right_shoe": 0.9,
}

# How harshly we punish "pop" (saturation * brightness together - that's
# what actually reads as neon/eye-popping, not saturation alone). Gamma > 1
# is forgiving through the middle of the range and punishing near the top.
POP_PENALTY_GAMMA = 2.0


def _normalize_hsv(hsv):
    h, s, v = hsv
    return h / 179.0, s / 255.0, v / 255.0


def _pop_penalty(s_norm, v_norm):
    """
    0 = calm (navy, black, grey, white, dark green...)
    1 = maximally eye-popping (bright saturated neon)

    White passes fine because S is near 0 regardless of V.
    Black/navy pass fine because V is low regardless of S.
    Only colors that are BOTH saturated AND bright get punished hard.
    
    s_norm is the normalized saturation
    v_norm is the normalized value
    """
    pop = s_norm * v_norm
    return pop ** (1.0 / POP_PENALTY_GAMMA)


def _region_saturation_score(color_entries):
    """Fraction-weighted saturation score for a single region's palette."""
    if not color_entries:
        return None
    total_fraction = sum(c["fraction"] for c in color_entries) or 1.0
    penalty = 0.0
    for c in color_entries:
        _, s_norm, v_norm = _normalize_hsv(c["hsv"])
        penalty += (c["fraction"] / total_fraction) * _pop_penalty(s_norm, v_norm)
    return 1.0 - penalty


def compute_saturation_score(outfit_colors, region_weights=None):
    region_weights = region_weights or REGION_SAT_WEIGHTS
    scored, weight_total = 0.0, 0.0
    for region, entries in outfit_colors.items():
        region_score = _region_saturation_score(entries)
        if region_score is None:
            continue
        w = region_weights.get(region, 1.0)
        scored += w * region_score
        weight_total += w
    return scored / weight_total if weight_total else 0.5


# Ideal hue-distance targets in degrees (0-180) and how forgiving each is
# (sigma of the Gaussian bump around that angle).
HARMONY_TARGETS = {
    "analogous":     (30,  10),
    "triadic":       (120, 10),
    "complementary": (180, 10),
}

# A color counts as "neutral" (black/white/grey/very dark) if it's
# desaturated or very dark. Neutrals harmonize with everything, so they
# short-circuit the hue comparison instead of getting scored on hue distance.
NEUTRAL_SAT_THRESHOLD = 0.18
NEUTRAL_VALUE_THRESHOLD = 0.20


def _is_neutral(hsv):
    _, s_norm, v_norm = _normalize_hsv(hsv)
    return s_norm < NEUTRAL_SAT_THRESHOLD or v_norm < NEUTRAL_VALUE_THRESHOLD


def _region_representative_hsv(color_entries):
    """Most prevalent color in the region."""
    return color_entries[0]["hsv"] if color_entries else None


def _hue_distance_deg(h1, h2):
    h1_deg, h2_deg = h1 / 179.0 * 360.0, h2 / 179.0 * 360.0
    d = abs(h1_deg - h2_deg) % 360.0
    return min(d, 360.0 - d)


def _pair_harmony_score(hsv_a, hsv_b):
    if _is_neutral(hsv_a) or _is_neutral(hsv_b):
        return 1.0  # neutrals never clash

    dist = _hue_distance_deg(hsv_a[0], hsv_b[0])
    best = 0.0
    for ideal, sigma in HARMONY_TARGETS.values():
        best = max(best, math.exp(-((dist - ideal) ** 2) / (2 * sigma ** 2)))
    return best


def compute_harmony_score(outfit_colors, pairs=None):
    """
    Compares representative hues across garment pairs. Defaults to
    shirt-vs-pants and shirt-vs-shoes (the relationships people actually
    notice first), but you can pass e.g.
    pairs=[("shirt","pants_left"), ("pants_left","left_shoe")] to change
    which relationships count.
    """
    reps = {
        region: _region_representative_hsv(entries)
        for region, entries in outfit_colors.items()
        if entries
    }

    if pairs is None:
        pants = "pants_left" if "pants_left" in reps else None
        shoe = "left_shoe" if "left_shoe" in reps else None
        pairs = [
            p for p in [("shirt", pants), ("shirt", shoe), (pants, shoe)]
            if p[0] and p[1] and p[0] in reps and p[1] in reps
        ]

    if not pairs:
        return 0.5

    scores = [_pair_harmony_score(reps[a], reps[b]) for a, b in pairs]
    return sum(scores) / len(scores)


# ----------------------------------------------------------------------
# 3. TREND SCORE - placeholder until we pick a data source
# ----------------------------------------------------------------------

def compute_trend_score(outfit_colors):
    """
    TODO: replace with the Markov-chain lookup once we've settled:
      - state definition (hue bins / named families / your KMeans centers)
      - training sequence (Pantone COTY history, WGSN, scraped runway
        data, or a mocked sequence to prototype with)
    """
    return 0.5


# ----------------------------------------------------------------------
# COMBINE
# ----------------------------------------------------------------------

def score_outfit(outfit_colors, weights=(0.30, 0.15, 0.55)):
    sat_w, harmony_w, trend_w = weights
    saturation = compute_saturation_score(outfit_colors)
    harmony = compute_harmony_score(outfit_colors)
    trend = compute_trend_score(outfit_colors)

    total = sat_w * saturation + harmony_w * harmony + trend_w * trend
    return total, {"saturation": saturation, "harmony": harmony, "trend": trend}


if __name__ == "__main__":
    # Swap this for the real outfit_colors dict returned by mediapipe_v4.main()
    examples = {
        "black shirt + dark green pants + black shoes": {
            "shirt": [{"hsv": (0, 0, 30), "fraction": 1.0}],
            "pants_left": [{"hsv": (60, 180, 90), "fraction": 1.0}],
            "pants_right": [{"hsv": (60, 180, 90), "fraction": 1.0}],
            "left_shoe": [{"hsv": (0, 0, 20), "fraction": 1.0}],
            "right_shoe": [{"hsv": (0, 0, 20), "fraction": 1.0}],
        },
        "white shirt + neon pink pants + white shoes": {
            "shirt": [{"hsv": (0, 0, 240), "fraction": 1.0}],
            "pants_left": [{"hsv": (170, 255, 255), "fraction": 1.0}],
            "pants_right": [{"hsv": (170, 255, 255), "fraction": 1.0}],
            "left_shoe": [{"hsv": (0, 0, 235), "fraction": 1.0}],
            "right_shoe": [{"hsv": (0, 0, 235), "fraction": 1.0}],
        },
    }

    for label, outfit_colors in examples.items():
        total, breakdown = score_outfit(outfit_colors)
        print(f"{label}")
        print(f"  total: {total:.3f}  {breakdown}")