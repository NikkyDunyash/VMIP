import argparse
import string
import numpy as np
import scipy.ndimage
import skimage.io as io




def create_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('input_image')
    parser.add_argument('kernel')
    parser.add_argument('output_image')
    parser.add_argument('noise_level', type=float)
    return parser



def main():
    namespace=create_parser().parse_args()
    image=np.array(io.imread(namespace.input_image), dtype=float)

    sum_func=Exp()+Exp()
    print(call_func(sum_func, 0))

if __name__=='__main__':
    main()