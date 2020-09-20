# tf-siammask
[![PyPI version](https://badge.fury.io/py/tf-siammask.svg)](https://badge.fury.io/py/tf-siammask)
![Upload Python Package](https://github.com/Licht-T/tf-siammask/workflows/Upload%20Python%20Package/badge.svg)

[SiamMask](https://github.com/foolwood/SiamMask) implementation with Tensorflow 2.

## Install
```bash
pip install tf-siammask
```

## Example
```python
import numpy as np
import PIL.Image
import siammask

sm = siammask.SiamMask()

# Weight files are automatically retrieved from GitHub Releases
sm.load_weights()

# Adjust this parameter for the better mask prediction
sm.box_offset_ratio = 1.5

img_prev = np.array(PIL.Image.open('data/cat1.jpg'))[..., ::-1]
box_prev = np.array([[227, 184], [381, 274]])
img_next = np.array(PIL.Image.open('data/cat2.jpg'))[..., ::-1]

# Predicted box and mask images is created if `debug=True`
box, mask = sm.predict(img_prev, box_prev, img_next, debug=True)
```

### Test data

| |  Previous frame  |  Next frame |
| ---- | ---- | ---- |
| File name | `./data/cat1_with_box.jpg` | `./data/cat2.jpg` |
| Image |  ![cat](https://raw.githubusercontent.com/Licht-T/tf-siammask/master/data/cat1_with_box.jpg)  |  ![cat](https://raw.githubusercontent.com/Licht-T/tf-siammask/master/data/cat2.jpg)  |

### Predicted mask for `./data/cat2.jpg`
![mask](https://raw.githubusercontent.com/Licht-T/tf-siammask/master/data/predicted_mask.png)

## TODO
* [x] Bounding-box regression
* [x] Mask refinement network
* [x] Pre-trained model for Tensorflow 2.0
* [ ] Training code
* [ ] Object tracking code

## Reference
```
@inproceedings{wang2019fast,
    title={Fast online object tracking and segmentation: A unifying approach},
    author={Wang, Qiang and Zhang, Li and Bertinetto, Luca and Hu, Weiming and Torr, Philip HS},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    year={2019}
}
```
