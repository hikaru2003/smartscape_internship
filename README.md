# smartscape_internship
this is smartscape internship repo.

project detail is described in [presentation.pptx](https://github.com/hikaru2003/smartscape_internship/blob/main/presentation.pptx).
so please check it.

# Quick Start

## Set up

```sh
mkdir venv output
python3 -m venv ./venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## execution

```sh
python3 main.py
file_type: [file\ type]
```

filetype is defined as sample/[filetype]_1.png.
for example, if you want to take the difference between sample/chess_1.png and sample/chess_2.png, file type is chess.
```sh
python3 main.py
file_type: chess
```

result .png file will be stored in the output directory.
there are several .png files.

* adjust_light: images adjusted light
* blur: images applied Gaussian Blur
* diff_filled: diff image filled with red color
* diff: diff image emphasized with red line
* homo: images applied Homography
* imge_diff: image of absolute difference
* img_th: image applied threshold
* mask: image applied a mask
* sample: images of input files

## directory description

In sample directory, set of sample images are stored. you can use these images as input.
In success directory, there are successful images which the diff of two images is accurate.
In result directory, there are some experiments of algorithms.
