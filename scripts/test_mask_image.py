import os
import cv2
import numpy as np


def mask_image(image):
        """Keep the tabletop polygon and fill the remaining pixels with the background color."""

        polygon_points = [
            [1540, 600],
            [1410, 720],
            [1790, 960],
            [1500, 2000],
            [560, 340],
            [920, 280],
        ]
        background_color = (204, 255, 229)

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        polygon = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)

        masked_image = np.full_like(image, background_color, dtype=image.dtype)
        masked_image[mask > 0] = image[mask > 0]

        return masked_image


if __name__ == '__main__':
    input_path = "/home/yuan/self_vla/residual-offpolicy-rl/outputs/camera_inspection/current_four/port_30002__serial_54121932.jpg"
    output_path = "/home/yuan/self_vla/scripts/masked_annotated_image_30002_.jpg"

    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {input_path}")

    masked_img = mask_image(img_bgr)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    success = cv2.imwrite(output_path, masked_img)
    if not success:
        raise RuntimeError(f"Failed to save image: {output_path}")

    print(f"Masked image saved to: {output_path}")
