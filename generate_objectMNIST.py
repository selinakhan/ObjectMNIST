import random
from PIL import Image
import numpy as np
import random
import json
from tqdm import tqdm, trange
import argparse

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as T

# TODO generate from small to largest

def generate_dataset(image_size, n_digits, n_datapoints, data, resize_range, im_dir, split):
    dataset = {}
    width, height = image_size[0], image_size[1]

    to_PIL = T.ToPILImage()

    print("Generating images...")
    for i, datapoint in enumerate(trange(n_datapoints)):
        img_bg = to_PIL(torch.zeros(image_size))
        key = f'{i}'
        dataset[key] = []
        
        img_digits = random.randint(n_digits[0], n_digits[1])

        for j in range(img_digits):
            image, target = next(iter(data))
            metadata = {}
            img = to_PIL(torch.squeeze(image))
            
            resiz = random.randint(resize_range[0], resize_range[1])
            img = img.resize((resiz, resiz))
            
            metadata['target'] = target.item()

            im_size = img.size[0]
            half_im = int(im_size / 2)
            x = random.randint(0, width-im_size)
            y = random.randint(0, height-im_size)

            overlap = np.array(img_bg)[y:y+im_size, x:x+im_size]

            attempts = 0
            while np.any(overlap) and attempts < 25:
                x = random.randint(0, width-im_size)
                y = random.randint(0, height-im_size)

                overlap = np.array(img_bg)[y:y+im_size, x:x+im_size]
                attempts +=1

            if attempts == 25:
                n_datapoints +=1
                break
            
            metadata['bbox'] = [x, y, im_size, im_size]

            dataset[key].append(metadata)
            img_bg.paste(img, (x, y, x+im_size, y+im_size))
            
        img_bg.save(f'{im_dir}/{split}/{i}.png')
        
        
    return dataset





if __name__ == "__main__":
    parser = argparse.ArgumentParser() 

    # Model
    parser.add_argument('--image_size', default=(640, 640), type=tuple, required=False, help="Size of generated images.")
    parser.add_argument('--n_digits', default=(2, 8), type=tuple, required=False, help='Amount of digits per image. Can be a range for randomization.')
    parser.add_argument('--n_datapoints', default=100, type=int, required=False, help='Amount of images to generate.')
    parser.add_argument('--split', default='train', required=False, help='Sample either from the MNIST train or test set.')
    parser.add_argument('--resize_range', default=(82, 252), type=tuple, required=False, help='Resizing scales of digits. Can be a range for randomization.')
    parser.add_argument('--out_dir' ,default='out', type=str, required=False, help='Path to store output metadata.')
    parser.add_argument('--im_dir' ,default='out/images', type=str, required=False, help='Path to store output images.')
    parser.add_argument('--filename' ,default='test', type=str, required=False, help='Filename for generated dataset.')

    args = parser.parse_args()

    if args.image_size[0] != args.image_size[1]:
        raise ValueError(f'Images must be square. Ensure height and width are the same.')

    if args.n_digits[0] > args.n_digits[1]:
        raise ValueError(f'Range improperly formatted. Should be: tuple(min_images, max_images).')

    if args.resize_range[0] > args.resize_range[1]:
        raise ValueError(f'Range improperly formatted. Should be: tuple(min_size, max_size).')

    if args.split not in ['train', 'test']:
        raise ValueError(f'Split must be either "train" or "test".')
    

    transform = T.Compose([T.ToTensor()])

    mnist_train = datasets.MNIST(root='MNIST', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='MNIST', train=False, download=True, transform=transform)

    mnis_train = torch.utils.data.DataLoader(mnist_train, batch_size=1, shuffle=True)
    mnis_test = torch.utils.data.DataLoader(mnist_test, batch_size=1, shuffle=True)


    if args.split == 'train':
        data = mnis_train
    else: 
        data = mnis_test

    dataset = generate_dataset(image_size=args.image_size, n_digits=args.n_digits, n_datapoints=args.n_datapoints, data=data, resize_range=args.resize_range, im_dir=args.im_dir, split=args.split)
    
    print("Saving file...")
    out_file = open(f'{args.out_dir}/{args.filename}.json', 'w')
  
    json.dump(dataset, out_file, indent=3)

    print("Done!")