# CIFAR10
# python cal_fid_stat.py --dataset cifar10 --img_size 32 --image_data_path data/cifar10_images --output_file fid_stat/fid_stats_cifar10_train.npz

# CIFAR100
# python cal_fid_stat.py --dataset cifar100 --img_size 32 --image_data_path data/cifar100_images --output_file fid_stat/fid_stats_cifar100_train.npz

# CELEBA
# python cal_fid_stat.py --dataset celeba --img_size 64 --image_data_path data/celeba64_images --output_file fid_stat/fid_stats_celeba64_train.npz
# python cal_fid_stat.py --dataset celeba --img_size 128 --image_data_path data/celeba128_images --output_file fid_stat/fid_stats_celeba128_train.npz

import os
import glob
import argparse
import numpy as np
import imageio
import tensorflow.compat.v1 as tf
import datasets
import utils.fid_score as fid
import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='dataset'
    )
    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')
    parser.add_argument('--img_size', type=int, default=32, help='image size, 32 for cifar10, 48 for stl10')
    parser.add_argument('--dis_bs', type=int, default=100, help='batch size of D')
    parser.add_argument('--num_workers', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--image_data_path', type=str, required=True,
        help='set path to training set jpg images dir')
    parser.add_argument('--output_file', type=str, default='fid_stat/fid_stats_celeba64_train.npz',
        help='path for where to store the statistics')

    opt = parser.parse_args()
    print(opt)
    return opt

def load_images(image_list, batch_size=100):
    images = []
    total_images = len(image_list)
    for i in range(0, total_images, batch_size):
        batch_files = image_list[i:i + batch_size]
        batch_images = [imageio.imread(fn) for fn in batch_files]
        images.extend(batch_images)
        print(f"Loaded {len(images)} of {total_images} images")
    return np.array(images)

# Function to convert a tensor to a uint8 numpy array
def tensor_to_uint8(tensor):
    array = tensor.numpy()  # Convert the tensor to a numpy array
    array = np.transpose(array, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    array = (((0.5*array)+0.5) * 255).astype(np.uint8)  # Scale and convert to uint8
    return array

def loadstats(output_path):
    loaded_data = np.load(output_path)
    loaded_mu = loaded_data['mu']
    loaded_sigma = loaded_data['sigma']
    print('Loaded mu:', loaded_mu)
    print('Loaded sigma:', loaded_sigma)

def main():
    args = parse_args()

    ########
    # PATHS
    ########
    output_path = args.output_file
    image_data_path = args.image_data_path
    if not os.path.exists(image_data_path):
        os.makedirs(image_data_path)

    # if you have downloaded and extracted
    #   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    # set this path to the directory where the extracted files are, otherwise
    # just set it to None and the script will later download the files for you
    inception_path = None
    print("check for inception model..", end=" ", flush=True)
    inception_path = fid.check_or_download_inception(inception_path)  # download inception if necessary
    print("ok")

    # Dataloader
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train
    print(len(train_loader))

    # Iterate over the train_loader and save the images
    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx)
        '''
        for i in range(data.size(0)):
            image = tensor_to_uint8(data[i])
            image_path = os.path.join(image_data_path, f'image_{batch_idx * data.size(0) + i}.png')
            imageio.imsave(image_path, image)
        '''
            
    print("Checking in directory:", image_data_path)

    # loads all images into memory (this might require a lot of RAM!)
    print("load images..", end=" ", flush=True)


    # Search for images with different extensions and include subdirectories
    image_list = glob.glob(os.path.join(image_data_path, '**', '*.jpg'), recursive=True) + \
                 glob.glob(os.path.join(image_data_path, '**', '*.png'), recursive=True) 
    print("Image list length:", len(image_list))

    print("create inception graph..", end=" ", flush=True)
    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    print("ok")

    print("calculte FID stats..", end=" ", flush=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        mu,sigma= fid._handle_path(image_data_path, sess, low_profile=True, batch_size=100)
        np.savez_compressed(output_path, mu=mu, sigma=sigma)
    print("finished")

    loadstats(output_path)

if __name__ == '__main__':
    main()
