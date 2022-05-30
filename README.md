# Generating an ObjectMNIST dataset
This repository includes the code needed to generate a dataset with MNIST digits randomly placed on an image. It exports the labels with corresponding bounding boxes; useful for object detection.

## Code files

The repositoru includes two files:

`generate.py` - This file is used to generate a dataset with a specified amount of datapoints and degrees of randomness in both size and amount of digits per image. 

`mnist.py` - This file includes a data processor class which can be used for training models using the generated dataset.

`requirements.txt` - This file includes the required packages to properly run the code.

## Code Usage

**To generate an ObjectMNIST dataset, the following command is used.**

`python3 generate.py --image_size [size of images (tuple)] --n_digits [amount of digits per image (tuple)] --n_datapoints [amount of datapoints to generate] --split [sample from MNIST train or test set] --resize_range [range of scaling individual digits (tuple)] --out_dir [path to store metadata] --im_dir [path to store images] --filename [filename for metadata .json file]`

The `--image_size` argument specifies the size of the images to be generated. Should be passed as a tuple with *(N, N)*, generating square images.

The `--n_digits` argument specifies how many digits should be generated on each image. If each image should have exactly the same amount of digits, this should be a tuple with *(N, N)* digits. If a random amount of digits between *M* and *N* should be placed, this should be a tuple with *(M, N)*, where *M < N.*  

The `--resize_range` argument specifies the image sizes to which the digits are scaled for each image. If each image should have exactly the same size, this should be a tuple with *(N, N)* digits. If a random amount of resizes between *M* and *N* should be done, this should be a tuple with *(M, N)*, where *M < N.*  


### Dataset Examples

```
"7": [{"target":  2, "bbox": [268, 334, 110, 110]},
	  {"target":  3, "bbox": [398, 391, 232, 232]},
		{ "target": 6, "bbox": [366, 240, 142, 142]},
		{"target":  0, "bbox": [10, 437, 145, 145]},
		{"target":  0, "bbox": [149, 1, 239, 239]},
		{"target":  9, "bbox": [229, 534, 101, 101]},
		{"target":  4, "bbox": [435, 69, 91, 91]}
	 ], ...
```

<img src="https://github.com/selinakhan/ObjectMNIST/blob/main/7.png?raw=true", width="128"/>

![example](https://github.com/selinakhan/ObjectMNIST/blob/main/7.png?raw=true)
 
