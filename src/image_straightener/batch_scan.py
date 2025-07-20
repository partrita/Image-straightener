import click
import cv2
import numpy as np
import glob
from os.path import basename, join, exists, splitext
from os import makedirs
from multiprocessing import Pool, cpu_count
from functools import partial
import logging


class PyImageSearch:
    @staticmethod
    def four_point_transform(image, pts):
        # 입력된 점들을 (왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래) 순서로 정렬합니다.
        # 이 부분이 좌우 반전 문제를 해결하는 핵심입니다.
        rect = np.array(pts, dtype="float32")

        # 1. 각 점의 합(x+y)을 계산하여 왼쪽 위(가장 작음)와 오른쪽 아래(가장 큼)를 찾습니다.
        s = rect.sum(axis=1)
        tl = rect[np.argmin(s)]  # top-left
        br = rect[np.argmax(s)]  # bottom-right

        # 2. 각 점의 차이(x-y)를 계산하여 오른쪽 위(음수 중 가장 작음)와 왼쪽 아래(양수 중 가장 큼)를 찾습니다.
        diff = np.diff(rect, axis=1)
        tr = rect[np.argmin(diff)]  # top-right
        bl = rect[np.argmax(diff)]  # bottom-left

        # 이제 정렬된 점들을 사용하여 사각형을 구성합니다.
        rect = np.array([tl, tr, br, bl], dtype="float32")

        # 이제 원본 코드의 변환 로직을 그대로 사용합니다.
        (tl, tr, br, bl) = rect  # 정렬된 점들을 다시 할당

        # 너비와 높이를 계산
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # 변환될 목적지 점들을 정의 (항상 정방향으로)
        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        # 원근 변환 행렬을 얻고 이미지를 변환
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    @staticmethod
    def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        return cv2.resize(image, dim, interpolation=inter)


imutils = PyImageSearch()
four_point_transform = PyImageSearch.four_point_transform


def auto_canny(image, sigma=0.33):
    """Applies automatic Canny edge detection."""
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def increase_contrast(image):
    """Increases the contrast of an image using CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge((l_channel_enhanced, a_channel, b_channel))
    img_contrasted = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return img_contrasted


def process_image(image_path, output_dir):
    """
    Processes a single image for document scanning, including edge detection,
    perspective transform, and saving the result.
    """
    file_name = basename(image_path)
    file_name_without_ext, file_extension = splitext(file_name)
    output_path = join(output_dir, f"{file_name_without_ext}{file_extension}")

    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"Skipping {file_name}: Not a valid image file or could not be read by OpenCV.")
            return

        original_image = image.copy()
        ratio = original_image.shape[0] / 500.0
        display_image = imutils.resize(image.copy(), height=500)

        gray = cv2.cvtColor(display_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = auto_canny(blurred)

        results = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = results[0] if len(results) == 2 else results[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        screen_contour = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screen_contour = approx
                break

        if screen_contour is not None:
            warped = four_point_transform(original_image, screen_contour.reshape(4, 2) * ratio)
            warped = increase_contrast(warped)
            cv2.imwrite(output_path, warped)
            logging.info(f"Successfully scanned and saved {file_name} to {output_path}")
        else:
            logging.info(f"No clear document outline found for {file_name}. Saving original image.")
            cv2.imwrite(output_path, original_image)
    except Exception as e:
        logging.error(f"ERROR processing {file_name}: {e}")
        try:
            original_image = cv2.imread(image_path)
            if original_image is not None:
                cv2.imwrite(output_path, original_image)
                logging.warning(f"Saved original {file_name} to {output_path} due to error.")
        except Exception as save_e:
            logging.error(f"Further error saving original {file_name}: {save_e}")


@click.command()
@click.option(
    "-i",
    "--input_path",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to the directory containing images to be scanned.",
)
@click.option(
    "-o",
    "--output_path",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Path to the directory where scanned images will be saved.",
)
@click.option(
    "-w",
    "--workers",
    default=cpu_count(),
    type=int,
    help="Number of worker processes to use for parallel processing.",
)
@click.option(
    "-l",
    "--log_file",
    default="batch_scan.log",
    type=str,
    help="Path to the log file.",
)
def batch_scan(input_path, output_path, workers, log_file):
    """
    A batch document scanning tool.

    This script processes all files in the INPUT_PATH directory that OpenCV can
    read as images. For each valid image, it attempts to detect document edges.
    The script automatically applies perspective transform and saves the scanned image.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    if not exists(output_path):
        makedirs(output_path)
        logging.info(f"Created output directory: {output_path}")

    all_files_in_input_dir = glob.glob(join(input_path, "*"))

    image_files = []
    logging.info("Detecting readable image files...")
    for file_path in all_files_in_input_dir:
        temp_img = cv2.imread(file_path)
        if temp_img is not None:
            image_files.append(file_path)

    image_files.sort()

    if not image_files:
        logging.warning(
            f"No readable image files found in {input_path}. Please ensure your directory contains valid image formats that OpenCV can decode. Exiting."
        )
        return

    total_images = len(image_files)
    logging.info(f"Found {total_images} readable images to process in {input_path}")

    process_func = partial(process_image, output_dir=output_path)

    with Pool(workers) as pool:
        results = list(pool.imap_unordered(process_func, image_files))

    logging.info("\nAll images processed. DONE!")


if __name__ == "__main__":
    batch_scan()
