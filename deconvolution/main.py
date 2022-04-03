import argparse
import skimage.io as io
import numpy as np
from functional import *
import optim
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# import sklearn.linear_model


def create_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('input_image')
    parser.add_argument('kernel')
    parser.add_argument('output_image')
    parser.add_argument('noise_level', type=float)
    return parser

def normalize(image):
    return (image-image.min())/(image.max()-image.min())

def gen_alpha1(noise_level):
    return 0.0094*noise_level+0.04

def gen_alpha2(noise_level):
    return 0.0004*noise_level+0.018


def main():
    namespace=create_parser().parse_args()
    input_image=255*normalize(
        np.array(io.imread(namespace.input_image), dtype=float)[:, :, 0])
    kernel=np.array(io.imread(namespace.kernel), dtype=float)[:, :, 0]
    kernel/=np.sum(kernel)

    N=200
    residual_norm=ResL1(A=Conv(kernel), b=input_image)
    directions=[[0 , 1], [1, 0], [1, 1], [1, -1]]

    # best alphas were found for 0, 5, 10, 15, 20 noise levels
    # and linearly approximated

    # noise_lvls=np.arange(25, step=5)
    # alpha1=[0.02, 0.09, 0.17, 0.18, 0.21]  # best alphas found
    # alpha2=[0.01, 0.03, 0.02, 0.03, 0.02]  # best alphas found
    # plt.scatter(noise_lvls, alpha1, label=r'$\alpha_1$', marker='+')
    # plt.scatter(noise_lvls, alpha2, label=r'$\alpha_2$', marker='+')
    # lin_reg=sklearn.linear_model.LinearRegression().fit(noise_lvls.reshape(-1, 1), alpha1)
    # print(lin_reg.coef_, lin_reg.intercept_)
    # lin_reg.fit(noise_lvls.reshape(-1, 1), alpha2)
    # print(lin_reg.coef_, lin_reg.intercept_)
    # plt.plot(np.arange(21), gen_alpha1(np.arange(21)))
    # plt.plot(np.arange(21), gen_alpha2(np.arange(21)))
    # plt.legend()
    # plt.show()
    
    # print(gen_alpha1(namespace.noise_level), gen_alpha2(namespace.noise_level))

    func=residual_norm + 0.3*TV(directions)\
        # +gen_alpha1(namespace.noise_level)*TV(directions) \
        # +gen_alpha2(namespace.noise_level)*TV2(directions) 
    optimizer=optim.GD(func, x0=np.zeros(input_image.shape), lr=1, 
        momentum_factor=0.9, nesterov=True)
    scheduler=optim.ReduceLROnPlateau(optimizer, patience=1, verbose=True)
    # for i in tqdm(range(N)):
    for k in range(N):
        scheduler.step()
    
    output_image=255*normalize(optimizer.x[ :, :, np.newaxis])
    output_image=np.array(np.repeat(output_image, repeats=3, axis=2), dtype=np.uint8)
    io.imsave(namespace.output_image, output_image)


if __name__=='__main__':
    main()