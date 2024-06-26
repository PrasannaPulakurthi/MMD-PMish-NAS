# python cal_fid_stat.py --data_path data/celeba --output_file fid_stat/fid_stats_celeba64_train.npz
# python cal_fid_stat.py --data_path cifar100 --output_file fid_stat/fid_stats_cifar100_train.npz

import os
import glob
import argparse
import numpy as np
import imageio
import tensorflow.compat.v1 as tf

import utils.fid_score as fid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='set path to training set jpg images dir')
    parser.add_argument(
        '--output_file',
        type=str,
        default='fid_stat/fid_stats_celeba64_train.npz',
        help='path for where to store the statistics')

    opt = parser.parse_args()
    print(opt)
    return opt

def load_images(image_list, batch_size=100):
    images = []
    total_images = len(image_list)
    for i in range(0, total_images, batch_size):
        batch_files = image_list[i:i + batch_size]
        batch_images = [imageio.imread(fn).astype(np.float32) for fn in batch_files]
        images.extend(batch_images)
        print(f"Loaded {len(images)} of {total_images} images")
    return np.array(images)

def main():
    args = parse_args()

    ########
    # PATHS
    ########
    data_path = args.data_path
    print("Checking in directory:", data_path)

    output_path = args.output_file
    # if you have downloaded and extracted
    #   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    # set this path to the directory where the extracted files are, otherwise
    # just set it to None and the script will later download the files for you
    inception_path = None
    print("check for inception model..", end=" ", flush=True)
    inception_path = fid.check_or_download_inception(inception_path)  # download inception if necessary
    print("ok")

    # loads all images into memory (this might require a lot of RAM!)
    print("load images..", end=" ", flush=True)
    # Search for images with different extensions and include subdirectories
    image_list = glob.glob(os.path.join(data_path, '**', '*.jpg'), recursive=True) + \
                 glob.glob(os.path.join(data_path, '**', '*.png'), recursive=True) 
    
    print("Image list length:", len(image_list))
    images = load_images(image_list, batch_size=1000)
    print("%d images found and loaded" % len(images))

    print("create inception graph..", end=" ", flush=True)
    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    print("ok")

    print("calculte FID stats..", end=" ", flush=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
        np.savez_compressed(output_path, mu=mu, sigma=sigma)
    print("finished")


if __name__ == '__main__':
    main()
