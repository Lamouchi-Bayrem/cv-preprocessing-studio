def crop_roi(image, coords):
    if coords is None:
        return image

    x1, y1, x2, y2 = coords
    h, w = image.shape[:2]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    return image[y1:y2, x1:x2]
