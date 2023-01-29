import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import numpy as np
import PIL.Image

tf.config.set_visible_devices([], 'GPU')

hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

async def load_image(contentPath, stylePath):
    content_image = plt.imread(contentPath)
    style_image = plt.imread(stylePath)
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = tf.image.resize(style_image, (256, 256))
    return content_image, style_image

def transform(content_image, style_image):
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    tf.keras.backend.clear_session()
    return outputs[0]

async def show(tensor, images_ids):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    image = PIL.Image.fromarray(tensor)
    image.save('photos/' + images_ids[0] + images_ids[1] + '.jpg')

async def download_images(album):
    images_ids = []
    for obj in album:
        if obj.photo:
            file_id = obj.photo[-1].file_id
        else:
            file_id = obj[obj.content_type].file_id
        file = await bot.get_file(file_id)
        images_ids.append(file_id)
        await bot.download_file(file.file_path, 'photos/' + file_id + '.jpg')
    return images_ids
