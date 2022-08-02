import tensorflow as tf

def encode_single_sample(img_path, img_height=32, img_width=32):
    try:
        img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [img_height, img_width])
        img = tf.transpose(img, perm=[1, 0, 2])
        return img
    
    except Exception as e:
        print('file_path', img_path)
        print(e)
        return e