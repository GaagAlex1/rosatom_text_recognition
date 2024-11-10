from skimage import util, transform
from skimage.feature import canny
from skimage.util import img_as_ubyte
from skimage import color
from skimage.transform import hough_circle, hough_circle_peaks
import numpy as np
from PIL import Image

def get_textbox_rectangle(image, text_box_detector):
    predictions = text_box_detector.predict(image, verbose=False)
    boxes = predictions[0].boxes.xyxy.cpu()
    convex_hull = [
        int(boxes[:, 0].min()),
        int(boxes[:, 1].min()),
        int(boxes[:, 2].max()),
        int(boxes[:, 3].max()),
    ]
    return convex_hull


def is_circle_text(image, textbox_rectangle):
    alpha = 0.2
    max_image_size = 380
    side_ratio_for_circle = 1.61
    radius_min = 0.4
    radius_max = 0.55

    delta_x = textbox_rectangle[2] - textbox_rectangle[0]
    delta_y = textbox_rectangle[3] - textbox_rectangle[1]

    convex_hull_extended = [
        textbox_rectangle[0] - int(alpha * delta_x),
        textbox_rectangle[1] - int(alpha * delta_y),
        textbox_rectangle[2] + int(alpha * delta_x),
        textbox_rectangle[3] + int(alpha * delta_y),
    ]
    cropped_image = image.crop(textbox_rectangle)
    cropped_image_for_hough = image.crop(convex_hull_extended)

    if max(cropped_image.size) > max_image_size:
        if cropped_image.size[0] > cropped_image.size[1]:
            cropped_image = cropped_image.resize(
                (
                    max_image_size,
                    int(
                        max_image_size
                        * cropped_image.size[1]
                        / cropped_image.size[0]
                    ),
                )
            )
            cropped_image_for_hough = cropped_image_for_hough.resize(
                (
                    max_image_size,
                    int(
                        max_image_size
                        * cropped_image_for_hough.size[1]
                        / cropped_image_for_hough.size[0]
                    ),
                )
            )
        else:
            cropped_image = cropped_image.resize(
                (
                    int(
                        max_image_size
                        * cropped_image.size[0]
                        / cropped_image.size[1]
                    ),
                    max_image_size,
                )
            )
            cropped_image_for_hough = cropped_image_for_hough.resize(
                (
                    int(
                        max_image_size
                        * cropped_image_for_hough.size[0]
                        / cropped_image_for_hough.size[1]
                    ),
                    max_image_size,
                )
            )

    image = img_as_ubyte(color.rgb2gray(cropped_image_for_hough))
    edges = canny(image, sigma=3, low_threshold=10, high_threshold=70)

    hough_radii = np.linspace(
        min(image.shape[:1]) * radius_min, max(image.shape[:1]) * radius_max, 30
    ).round()
    hough_res = hough_circle(edges, hough_radii)

    accums, _, _, _ = hough_circle_peaks(
        hough_res, hough_radii, total_num_peaks=1, threshold=0.15
    )

    return (
            len(accums) > 0
            and max(delta_x, delta_y) / min(delta_x, delta_y)
            < side_ratio_for_circle
    )


def unbend_circle_img(image, textbox_rectangle):
    cropped_image = image.crop(textbox_rectangle)
    cropped_image = cropped_image.resize((380, 380))
    ans = []
    for i in range(4):
        float_image = util.img_as_float(cropped_image)
        image_polar = transform.warp_polar(
            float_image,
            scaling="linear",
            radius=float_image.shape[0],
            channel_axis=-1,
        )
        rotated_polat_image = transform.rotate(image_polar, angle=90)
        ans.append(Image.fromarray((rotated_polat_image * 255).astype("uint8")))
        cropped_image = cropped_image.rotate(90)
    return ans


def get_text_and_box_from_image(image, text_recognizer, tokenizer, text_box_detector):
    texts = []
    imgs_views = []
    textbox_rectangle = get_textbox_rectangle(image, text_box_detector)
    if is_circle_text(image, textbox_rectangle):
        imgs_views = unbend_circle_img(image, textbox_rectangle)
    else:
        imgs_views = []
        for i in range(4):
            imgs_views.append(image.rotate(i * 90))

    for img in imgs_views:
        texts.append(text_recognizer.chat(tokenizer, img, ocr_type='ocr', gradio_input=True))

    return max(texts, key=len), textbox_rectangle  # most long text