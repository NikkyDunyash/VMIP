import argparse
from tabnanny import verbose
import skimage.io as io
import numpy as np
from functional import *
import optim
import time
from tqdm import tqdm


def create_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('input_image')
    parser.add_argument('kernel')
    parser.add_argument('output_image')
    parser.add_argument('noise_level', type=float)
    return parser

def normalize(image):
    return (image-image.min())/(image.max()-image.min())

def main():
    namespace=create_parser().parse_args()
    input_image=255*normalize(
        np.array(io.imread(namespace.input_image), dtype=float)[:, :, 0])
    kernel=np.array(io.imread(namespace.kernel), dtype=float)[:, :, 0]
    kernel/=np.sum(kernel)

    N=500
    residual_norm=ResL1(A=Conv(kernel), b=input_image)
    directions=[[0 , 1], [1, 0], [1, 1], [1, -1]]
    alpha1, alpha2=(0.01, 0.02)
    func=residual_norm+alpha1*TV(directions)+alpha2*TV2(directions)
    # optimizer=optim.GD(func, x0=, lr=1)
    optimizer=optim.GD(func, x0=np.zeros(input_image.shape), lr=1)
    scheduler=optim.ReduceLROnPlateau(optimizer, patience=1, verbose=True)
    start_time=time.perf_counter()
    for i in tqdm(range(N)):
        scheduler.step()
    end_time=time.perf_counter()
    print(f'{N} calls:', end_time-start_time)
    output_image=255*normalize(optimizer.x[ :, :, np.newaxis])
    output_image=np.array(np.repeat(output_image,repeats=3, axis=2), dtype=np.uint8)
    io.imsave(namespace.output_image, output_image)


if __name__=='__main__':
    main()