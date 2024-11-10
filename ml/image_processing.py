from skimage import util, transform
from skimage.feature import canny
from skimage.util import img_as_ubyte
from skimage import color
from skimage.transform import hough_circle, hough_circle_peaks
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz import process
from PIL import Image
import pandas as pd
import os

# символы, которые могут встречаться в артикулах деталей
DETAILS_UNIQUE_SIMBOLS = [
    "9",
    "Н",
    "4",
    "-",
    "Е",
    "P",
    " ",
    '"',
    "1",
    "8",
    "2",
    "7",
    "L",
    "5",
    "0",
    "М",
    "А",
    "/",
    "6",
    ".",
    "3",
]

DETAILS_PATH = os.path.join(os.path.dirname(__file__), 'details.xlsx')

def get_textbox_rectangle(image, text_box_detector):
    """
    Функция для получения прямоугольника, описывающего текстовый блок на изображении,
    она запускает YOLO

    Args:
    image (PIL.Image): изображение
    text_box_detector (YOLO): объект класса YOLO для детекции текстовых блоков

    Returns:
    list: прямоугольник, описывающий текстовый блок в формате [x1, y1, x2, y2]
    """
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
    """
    Функция проверяющая, является ли текстовый блок на изображении круглым
    Запускается поиск кругов определенного размера на изображении, обрезанном по прямоугольнику текстового блока
    Если изображение содержит круг, определенной четкости и соотношения сторон примерно соответствует квадрату,
    то функция возвращает True

    Args:
    image (PIL.Image): изображение
    textbox_rectangle (list): прямоугольник, описывающий текстовый блок в формате [x1, y1, x2, y2]

    Returns:
    bool: является ли текстовый блок на изображении круглым
    """
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


def process_image(img):
    '''
    Функция для обрезки слишком больших и детализированных изображений
    Это нужно, для того чтобы ускорить работу модели распознавания текста
    и отсеять лишнюю информацию, которая мешает распознаванию

    Args:
    img (PIL.Image): изображение

    Returns:
    PIL.Image: обрезанное изображение
    '''
    img_max_size = 720
    if max(img.size) > img_max_size:
        if img.size[0] > img.size[1]:
            img = img.resize(
                (img_max_size, int(img_max_size * img.size[1] / img.size[0]))
            )
        else:
            img = img.resize(
                (int(img_max_size * img.size[0] / img.size[1]), img_max_size)
            )
    return img


def unbend_circle_img(image, textbox_rectangle):
    '''
    Функция для преобразования круглого текстового блока прямой
    По сколько мы не знаем где находится шов, то мы делаем 4 поворота на 90 градусов

    Args:
    image (PIL.Image): изображение
    textbox_rectangle (list): прямоугольник, описывающий текстовый блок в формате [x1, y1, x2, y2]

    Returns:
    list: список из 4 изображений, полученных из круглого текстового блока
    '''
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


def get_text_and_box_from_image(
        image, text_recognizer, tokenizer, text_box_detector
):
    '''
    Функция для получения текста и прямоугольника, описывающего текстовый блок на изображении
    Здесь запускается трансформер, под разными углами, чтобы улучшить качество распознавания.

    Args:
    image (PIL.Image): изображение
    text_recognizer (AutoModel): GOT-OCR2_0
    tokenizer (AutoTokenizer): токенайзер для GOT-OCR2_0
    text_box_detector (YOLO): объект класса YOLO для детекции текстовых блоков

    Returns:
    list: список текстов, полученных из изображения
    list: прямоугольник, описывающий текстовый блок в формате [x1, y1, x2, y2]
    '''
    texts = []
    imgs_views = []
    textbox_rectangle = get_textbox_rectangle(image, text_box_detector)
    if is_circle_text(image, textbox_rectangle):
        imgs_views = unbend_circle_img(image, textbox_rectangle)
    else:
        imgs_views = []
        for i in range(4):
            imgs_views.append(process_image(image.rotate(i * 90)))

    for img in imgs_views:
        texts.append(
            text_recognizer.chat(
                tokenizer, img, ocr_type="ocr", gradio_input=True
            )
        )

    return texts, textbox_rectangle


def change_eng_symbols(text):
    '''
    Функция для замены английских символов на русские
    Это нужно, так как вывод модели распознавания текста содержит только английские символы

    Args:
    text (str): текст

    Returns:
    str: текст с замененными английскими символами
    '''
    text = text.replace("M", "М")
    text = text.replace("E", "Е")
    text = text.replace("A", "А")
    text = text.replace("H", "Н")
    return text


def find_nearest_number(text, article, article_end, parts_df):
    '''
    Функция для поиска наиболее правдоподобного номера детали в базе с деталями.
    Для этого используется библиотека rapidfuzz, которая очень быстро находит наиболее похожее слово в списке.
    Args:
    text (str): текст
    article (str): артикул детали
    article_end (int): индекс конца артикула в тексте
    parts_df (pd.DataFrame): датафрейм с информацией о деталях

    Returns:
    str: номер детали
    float: близость номера детали к базе
    '''
    num_variants = list(
        parts_df[parts_df.fixed_article == article]["ПорядковыйНомер"]
        .astype(int)
        .astype(str)
        .values
    )
    best_res = 0
    best_idx = 0
    for begin in range(article_end, len(text)):
        for end in range(len(text), begin, -1):
            res = process.extractOne(
                text[begin:end],
                num_variants,
                scorer=fuzz.ratio,
                score_cutoff=40,
            )
            if res is not None and res[1] > best_res:
                best_res = res[1]
                best_idx = num_variants.index(res[0])
    if best_res == 0:
        return "", 0
    return num_variants[best_idx], best_res


def find_nearest_article(
        text, possible_articles_masks, possible_articles, parts_df, mask_symbols
):
    '''
    Функция для поиска наиболее правдоподобного артикула детали в базе с деталями.
    Аналогично предыдущей функции, используется библиотека rapidfuzz.

    Args:
    text (str): текст
    possible_articles_masks (list): список артикулов деталей, очищенных от лишних символов
    possible_articles (list): список артикулов деталей
    parts_df (pd.DataFrame): датафрейм с информацией о деталях
    mask_symbols (list): список символов, которые нужно удалить из текста, так как они не влияют на артикул

    Returns:
    str: артикул детали
    str: номер детали
    float: близость артикула к базе
    float: близость номера к базе
    '''
    text_masked = change_eng_symbols(text)
    text_masked = "".join(filter(lambda x: x in mask_symbols, text_masked))
    best_res = 0
    best_idx = 0
    best_end = 0
    for begin in range(len(text_masked)):
        for end in range(len(text_masked), begin, -1):
            res = process.extractOne(
                text_masked[begin:end],
                possible_articles_masks,
                scorer=fuzz.ratio,
                score_cutoff=80,
            )
            if res is not None and res[1] > best_res:
                best_res = res[1]
                best_idx = possible_articles_masks.index(res[0])
                best_end = end
    if best_res == 0:
        return text, "", 0, 0
    num, num_score = find_nearest_number(
        text_masked, possible_articles[best_idx], best_end, parts_df
    )
    return possible_articles[best_idx], num, best_res, num_score


def fing_best_answer(
        texts, possible_articles_masks, possible_articles, parts_df, mask_symbols
):
    '''
    Функция для поиска наиболее правдоподобного артикула и номера детали в базе с деталями.

    Args:
    texts (list): список текстов
    possible_articles_masks (list): список артикулов деталей, очищенных от лишних символов
    possible_articles (list): список артикулов деталей
    parts_df (pd.DataFrame): датафрейм с информацией о деталях
    mask_symbols (list): список символов, которые нужно удалить из текста, так как они не влияют на артикул

    Returns:
    str: артикул детали и номер детали
    float: близость артикула к базе
    float: близость номера к базе
    '''

    best_art_score = 0
    best_num_score = 0
    best_article = ""
    best_num = ""
    for text in texts:
        if len(text) > 100:
            text = text[:100]
        article, num, art_score, num_score = find_nearest_article(
            text,
            possible_articles_masks,
            possible_articles,
            parts_df,
            mask_symbols,
        )

        if art_score * len(article) + num_score * len(
                num
        ) > best_art_score * len(best_article) + best_num_score * len(best_num):
            best_art_score = art_score
            best_num_score = num_score
            best_article = article
            best_num = num
    return (
        '"' + best_article + " " + best_num + '"',
        best_art_score,
        best_num_score,
    )


def get_detail_dataset_info():
    '''
    Функция для получения информации о деталях из базы

    Returns:
    list: possible_articles_masks - список артикулов деталей, очищенных от лишних символов (используется для быстрого поиска)
    list: possible_articles - список артикулов деталей
    pd.DataFrame: parts_df - датафрейм с информацией о деталях
    list: mask_symbols - список символов, которые нужно удалить из текста, так как они не влияют на артикул
    '''
    parts_df = pd.read_excel(DETAILS_PATH)
    possible_articles = parts_df.apply(
        lambda x: f"{x['ДетальАртикул'][1:-1]}", axis=1
    ).tolist()
    possible_articles = list(
        map(
            lambda x: "".join(
                filter(lambda y: y in DETAILS_UNIQUE_SIMBOLS, x)
            ).split()[0],
            possible_articles,
        )
    )
    parts_df.loc[:, "fixed_article"] = possible_articles
    parts_df = parts_df.dropna()
    possible_articles = list(set(parts_df.fixed_article))
    mask_symbols = list(
        filter(
            lambda x: x not in [".", '"', "'", "-", "/", " "],
            DETAILS_UNIQUE_SIMBOLS,
        )
    )
    possible_articles_masks = list(
        map(
            lambda x: "".join(filter(lambda y: y in mask_symbols, x)),
            possible_articles,
        )
    )
    return possible_articles_masks, possible_articles, parts_df, mask_symbols


def predict_on_image(
        image,
        possible_articles_masks,
        possible_articles,
        parts_df,
        mask_symbols,
        text_recognizer,
        tokenizer,
        text_box_detector,
):
    '''
    Функция для предсказания артикула и номера детали на изображении
    Основная функция проекта, которая совершает предсказание по изображению

    Args:
    image (PIL.Image): изображение
    possible_articles_masks (list): список артикулов деталей, очищенных от лишних символов
    possible_articles (list): список артикулов деталей
    parts_df (pd.DataFrame): датафрейм с информацией о деталях
    mask_symbols (list): список символов, которые нужно удалить из текста, так как они не влияют на артикул
    text_recognizer (AutoModel): GOT-OCR2_0
    tokenizer (AutoTokenizer): токенайзер для GOT-OCR2_0
    text_box_detector (YOLO): объект класса YOLO для детекции текстовых блоков

    Returns:
    str: артикул и номер детали
    list: прямоугольник, описывающий текстовый блок в формате [x1, y1, x2, y2]
    '''
    texts, textbox_rectangle = get_text_and_box_from_image(
        image, text_recognizer, tokenizer, text_box_detector
    )
    return (
        fing_best_answer(
            texts,
            possible_articles_masks,
            possible_articles,
            parts_df,
            mask_symbols,
        )[0],
        textbox_rectangle,
    )