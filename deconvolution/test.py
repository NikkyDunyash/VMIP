import argparse
import numpy as np
import skimage.io as io
import scipy.ndimage 
# import progressbar


def create_parser():
    parser=argparse.ArgumentParser()
    subparsers=parser.add_subparsers(dest='command', required=True)

    mse_parser=subparsers.add_parser('mse')
    mse_parser.add_argument('input_images', nargs=2)

    psnr_parser=subparsers.add_parser('psnr')
    psnr_parser.add_argument('input_images', nargs=2)

    conv_parser=subparsers.add_parser('convolve')
    conv_parser.add_argument('input_image')
    conv_parser.add_argument('kernel')
    conv_parser.add_argument('noise_level', type=float)
    conv_parser.add_argument('output_image')

    return parser


def mse(namespace):
    image_0=np.array(io.imread(namespace.input_images[0])[:, :, 0], dtype=float)
    image_1=np.array(io.imread(namespace.input_images[1])[:, :, 0], dtype=float)
    return np.mean((image_0-image_1)**2)
    
def psnr(namespace):
    MAX_VAL=2**8-1
    return 10*np.log10(MAX_VAL**2/mse(namespace))

def normalize(image):
    return (image-image.min())/(image.max()-image.min())

def main():
    namespace=create_parser().parse_args()
    if namespace.command=='mse':
        print(mse(namespace))
    elif namespace.command=='psnr':
        print(psnr(namespace))
    elif namespace.command=='convolve':
        input_image=np.array(io.imread(namespace.input_image, as_gray=True), dtype=float)
        input_image=255*normalize(input_image)
        kernel=np.array(io.imread(namespace.kernel), dtype=float)[:, :, 0]
        kernel/=np.sum(kernel)
        std=namespace.noise_level
        output_image=(scipy.ndimage.convolve(input_image, kernel)+
            +np.random.normal(0, std, input_image.shape))
        
        output_image=255*normalize(output_image)
        output_image=np.array(np.repeat(output_image[ :, :, np.newaxis],
            repeats=3, axis=2), dtype=np.uint8)
        io.imsave('blurred_image.png', output_image)




if __name__=='__main__':
    main()

    
    