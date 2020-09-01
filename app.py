import os

from flask import Flask, request, send_file

import numpy as np
import cv2
import io
import tensorflow as tf

import tensorflow_hub as hub
import PIL

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize the Flask application
app = Flask(__name__)

style_path = tf.keras.utils.get_file('kandinsky5.jpg',
                                     'https://i.pinimg.com/736x/9b/da/83/9bda8346cfb5ed5474b67729be969213.jpg')


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    # return PIL.Image.fromarray(tensor)
    return tensor


def load_img(img):
    img = img[tf.newaxis, :]
    return img


def load_style(path_to_img):
    # max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    img = img[tf.newaxis, :]
    return img


def blend_images(mask, img):
    gpmask = [mask]
    gpimg = [img]
    for i in range(6):
        gpmask.append(cv2.pyrDown(mask))
        gpimg.append(cv2.pyrDown(img))
    lpmask = [gpmask[5]]
    lpimg = [gpimg[5]]
    for i in range(5, 0, -1):
        Gmask = cv2.pyrUp(gpmask[i])
        print(Gmask.shape, gpmask[i - 1].shape)
        Lmask = cv2.subtract(gpmask[i - 1], Gmask)
        Gimg = cv2.pyrUp(gpimg[i])
        Limg = cv2.subtract(gpimg[i - 1], Gimg)
        lpmask.append(Lmask)
        lpimg.append(Limg)
    LS = []
    for la, lb in zip(lpmask, lpimg):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
        LS.append(ls)
    ls_ = LS[0]
    for i in range(1, 6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    return PIL.Image.fromarray(ls_)


def calculate_heat_map(img):
    orignal = img
    dims = (orignal.shape[1], orignal.shape[0])

    style_image = load_style(style_path)
    # == Parameters =======================================================================
    BLUR = 5

    MASK_DILATE_ITER = 20
    MASK_ERODE_ITER = 20
    MASK_COLOR = (220 / 255, 220 / 255, 202 / 255)  # In BGR format

    # == Processing =======================================================================

    # -- Read image -----------------------------------------------------------------------
    # img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.blur(img, ksize=(5, 5))
    median_pix = np.median(img)
    lower = int(max(0, 0.7 * median_pix))
    upper = int(min(255, 1.3 * median_pix))
    img_grey = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    high_thresh, thresh_im = cv2.threshold(img_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.2 * high_thresh

    # -- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, high_thresh, lowThresh)
    # edges = cv2.Canny(gray, upper, lower)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # -- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    for c in contour_info:
        cv2.fillConvexPoly(mask, c[0], (255))
    # cv2.fillConvexPoly(mask, max_contour[0], (255))

    # -- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask] * 3)
    # Create 3-channel alpha mask

    # -- Blend masked img into MASK_COLOR background --------------------------------------
    mask_stack = mask_stack.astype('float32') / 255.0  # Use float matrices,
    img = img.astype('float32') / 255.0  # for easy blending
    hub_module = hub.load('style_transfer')
    stylized_image = hub_module(tf.constant(load_img(img)), tf.constant(style_image))[0]
    img = tensor_to_image(stylized_image)
    img = cv2.resize(img, dims)

    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)  # Blend

    masked = (255 - masked * 255).astype('uint8')
    img = ((1 - img) * 255).astype('uint8')
    masked = PIL.Image.fromarray(masked)
    img = PIL.Image.fromarray(img)

    blended = PIL.Image.blend(masked, img, 0.4)

    return blended


@app.route('/')
def index():
    return 'hi'


@app.route('/api/heatmap', methods=['POST'])
def heatmap():
    # convert string of image data to uint8

    r = request.data
    # convert string of image data to uint8
    nparr = np.fromstring(r, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # do some fancy processing here....
    processed_img = calculate_heat_map(img)

    rawBytes = io.BytesIO()
    processed_img.save(rawBytes, 'jpeg')

    rawBytes.seek(0)

    content_type = 'image/jpeg'
    headers = {'content-type': content_type}
    return send_file(rawBytes, mimetype='image/jpeg')


# start flask app
if __name__ == '__main__':
    app.run()
