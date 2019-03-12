from itertools import cycle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2

colors_ = cycle(['cyan', 'yellow', 'magenta'])

def show_n_save_img(img_path, boxes, segments, out_path, classes, 
                    height, width, show_img_sz=600):
    """
    boxes       n x 4 numpy array OR a list of tupes, each of which has 4 elements
    segments    n x 4 numpy array
    classes     n x 1
    """
    assert len(boxes) == len(segments)
    assert len(boxes) == len(classes)
    my_dpi = 100
    fig = plt.figure()
    fig.set_size_inches(show_img_sz / my_dpi, show_img_sz / my_dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.set_xlim([0, show_img_sz])
    ax.set_ylim([show_img_sz, 0])
    #ax.set_aspect('equal')
    im = cv2.imread(img_path)
    im = cv2.resize(im, (show_img_sz, show_img_sz))
    im = im[:, :, (2, 1, 0)]
    ratio = show_img_sz / height
    print('ratio = {}'.format(ratio))

    ax.imshow(im, aspect='equal')

    for idx, bbox in enumerate(boxes):
        bbox = [p * ratio for p in bbox]
        nc = next(colors_)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=nc, linewidth=1.0)
            )
        
        bseg = segments[idx]
        bseg = [p * ratio for p in bseg]
        ax.add_patch(
            plt.Rectangle((bseg[0], bseg[1]),
                          bseg[2] - bseg[0],
                          bseg[3] - bseg[1], fill=False,
                          edgecolor=nc, linewidth=1.0)
        )
        
        class_name = str(classes[idx])

        ax.text(bbox[0], bbox[1] - 2,
                '{:s}'.format(class_name),
                bbox=dict(facecolor='None', alpha=0.4, edgecolor='None'),
                fontsize=14, color='white')
        
    plt.axis('off')
    plt.draw()
    plt.savefig(out_path, dpi=my_dpi)
    plt.close()