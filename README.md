# Image to Haiku
## Installation

1. Download the  [BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar](https://drive.google.com/drive/folders/189VY65I_n4RTpQnmLGj7IzVnOF6dmePC) into the root project directory

2. Open terminal at the root of the project directory and run `python3 setup.py install`

3. After install, run `python3 poetry_writer.py ./images/hiking.jpeg`

## Usage
`python3 poetry_writer.py path/to/image/image.jpeg` 

## NOTE
When NLTK first installs, it doesn't install all necessary data with it. If you get an error saying NLTK couldn't find data, look for the line of code in the error that looks like `nltk.download('FILE_NAME')`. If you run that line of python with the proper file name (which will be stated in the error) it will download the missing data. This only needs to be done once, and can be done from anywhere.
