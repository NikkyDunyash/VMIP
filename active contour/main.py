import argparse
import sys
import skimage.io 
import skimage.draw
import numpy as np
import scipy.ndimage
import scipy.interpolate
import scipy.sparse.linalg
import functional
import utils
import matplotlib.pyplot as plt
import os

def create_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('input_image')
    parser.add_argument('initial_snake')
    parser.add_argument('output_image')
    parser.add_argument('alpha', type=float)
    parser.add_argument('beta', type=float)
    parser.add_argument('tau', type=float)
    parser.add_argument('w_line', type=float)
    parser.add_argument('w_edge', type=float)
    parser.add_argument('kappa', nargs='?', type=float)
    return parser

def linear_interpolation(x, x0, x1, v0, v1):
    return (x-x1)/(x0-x1)*v0+(x-x0)/(x1-x0)*v1

def bilinear_interpolation(x, y, x0, x1, y0, y1,
                           v00, v01, v10, v11):
    return ((x-x1)/(x0-x1)*((y-y1)/(y0-y1)*v00+(y-y0)/(y1-y0)*v01)
           +(x-x0)/(x1-x0)*((y-y1)/(y0-y1)*v10+(y-y0)/(y1-y0)*v11))

def bilinear_image_inter(points, image):
    x0, x1=np.floor(points[:, 0]).astype(int), np.ceil(points[:, 0]).astype(int)
    y0, y1=np.floor(points[:, 1]).astype(int), np.ceil(points[:, 1]).astype(int)
    x, y=points[:, 0], points[:, 1]
    values=np.ones(points.shape)
    all_same=(x0==x1) & (y0==y1)
    x_same=(x0==x1) & (y0!=y1)
    y_same=(x0!=x1) & (y0==y1)
    all_diff=(x0!=x1) & (y0!=y1)

    # all same
    values[all_same]=image[:, x0[all_same], y0[all_same]].transpose()

    # x same
    values[x_same]=linear_interpolation(y[x_same], y0[x_same], y1[x_same],
                                            image[:, y0[x_same], x0[x_same]],
                                            image[:, y0[x_same], x1[x_same]]).transpose()

    # y same
    values[y_same]=linear_interpolation(x[y_same], x0[y_same], x1[y_same],
                                            image[:, y0[y_same], x0[y_same]],
                                            image[:, y1[y_same], x0[y_same]]).transpose()

    # all diff
    values[all_diff]=bilinear_interpolation(x[all_diff], y[all_diff], 
                                                x0[all_diff], x1[all_diff],
                                                y0[all_diff], y1[all_diff],
                                                image[:, y0[all_diff], x0[all_diff]],
                                                image[:, y0[all_diff], x1[all_diff]],
                                                image[:, y1[all_diff], x0[all_diff]],
                                                image[:, y1[all_diff], x1[all_diff]]).transpose()
    return values

def inner_forces(n, a, b):
    if n<5:
        raise ValueError('n<5')
    ifm=np.zeros((n, n))
    row=np.append([b, -a-4*b, 2*a+6*b, -a-4*b, b], 
                  np.zeros(n-5))
    for i in range(n):
        ifm[i]=np.roll(row, shift=i-2)
    return ifm

class GVFEulerEqs(scipy.sparse.linalg.LinearOperator):
    def __init__(self, p_grad, mu):
        self.p_grad=p_grad
        self.mu=mu
        self.shape=(2*p_grad[0].size, 2*p_grad[0].size)
        self.dtype=np.dtype(float)
        

    def _matvec(self, g):
        g=np.copy(g).reshape(self.p_grad.shape)
        return (self.mu*scipy.ndimage.laplace(g, mode='nearest')
               -(self.p_grad[0]**2+self.p_grad[1]**2)*g).flatten()
    
    def _rhs(self):
        return ((self.p_grad[0]**2+self.p_grad[1]**2)
                *(-1)*self.p_grad).flatten()

    def _gmres_callback(self, curr_iter):
        pr_norm=np.linalg.norm(self._matvec(curr_iter)-self._rhs())/np.linalg.norm(self._rhs())
        # print(pr_norm)


def external_forces(image, w_line, w_edge, 
                    std, gvf=False, mu=None):
    rad=int(np.ceil(3*std))
    if rad<1:
        raise ValueError(f'std={std}, expected>0')
    args_range=np.linspace(-rad, rad, num=2*rad+1)
    x=np.repeat(args_range[None, :], 2*rad+1, axis=0)
    y=np.repeat(args_range[:, None], 2*rad+1, axis=1)
    gaussian=functional.Gaussian(std=std)
    gauss_matrix=gaussian(x, y)
    gauss_matrix=gauss_matrix/gauss_matrix.sum()
    p_line=-1*scipy.ndimage.convolve(image, weights=gauss_matrix, mode='nearest')
    
    gauss_grad_matrix=gaussian.grad(x, y)
    image_grad=np.ndarray((2,)+image.shape)
    for i in range(2):
        image_grad[i]=scipy.ndimage.convolve(image, weights=gauss_grad_matrix[i], mode='nearest')
    p_edge=-1*(image_grad[0]**2+image_grad[1]**2)
   
    p=w_line*p_line+w_edge*p_edge
    p_grad=np.array([np.gradient(p, axis=1), np.gradient(p, axis=0)])
    for i in range(p_grad.shape[1]):
        for j in range(p_grad.shape[2]):
            norm=np.linalg.norm(p_grad[:, i, j])
            if norm<1e-5:
                p_grad[:, i, j]=0
            else:
                p_grad[:, i, j]/=norm


    efm=np.ndarray(p_grad.shape)
    if gvf:
        gvf_euler=GVFEulerEqs(p_grad=p_grad, mu=mu)
        # print('GMRes relative residual norm:')
        g_grad, info=scipy.sparse.linalg.gmres(gvf_euler, gvf_euler._rhs(), maxiter=10, 
                                               callback=gvf_euler._gmres_callback, callback_type='x')
        efm=-g_grad.reshape(efm.shape)
    else:
        efm=-p_grad
    return efm

def balloon_force(snake):
    snake_shifted=np.roll(snake, 1, axis=0)
    res=snake_shifted-snake
    normals=np.append(-res[:, 1].reshape(-1, 1), res[:, 0].reshape(-1, 1), axis=1)
    return normals/np.repeat(np.linalg.norm(normals, axis=1)[:, None], 1, axis=1)

# def discretization_step(snake):
#     return np.mean(np.linalg.norm(np.roll(snake, 1, axis=0)-snake, axis=1))

# X^{t+1}=(I+tau*A)^{-1}X^t+tau*F(X^t)
def move_snake(image, initial_snake, alpha, beta, tau, 
               w_line, w_edge, gvf=False, 
               b_factor=None, t_max=2000):

    ifm=inner_forces(initial_snake.shape[0], alpha, beta)
    if b_factor is None:
        b_factor=0.0
    efm=external_forces(image, w_line, w_edge, std=1.0,  gvf=gvf, mu=0.5)

    inverse_matrix=np.linalg.inv(np.eye(ifm.shape[0])+tau*ifm)
    curr_snake=initial_snake
    new_snake=np.zeros(initial_snake.shape)
    t=0
    while True:
        new_snake=(np.matmul(inverse_matrix, curr_snake)
                   +tau*(bilinear_image_inter(curr_snake, efm)
                        +b_factor*balloon_force(curr_snake)))
        
        if t>t_max:
            break
        if t%100==0:
            fig, ax=plt.subplots()
            plt.imshow(image, cmap='gray')
            ax.set_frame_on(False)
            ax.scatter(curr_snake[:, 0], curr_snake[:, 1], s=2, c='b')
            # ax.plot(result_snake[:, 0], result_snake[:, 1], '-b', lw=2)
            ax.set_xticks([]), ax.set_yticks([])
            ax.axis([0, image.shape[1], image.shape[0], 0])
            plt.savefig(f'plots/{t}.jpg')
            plt.close()
        if t%5==0:
            tck, u=scipy.interpolate.splprep([new_snake[:,0], new_snake[:,1]], s=0)
            new_points=scipy.interpolate.splev(np.linspace(0, 1, num=u.size), tck)
            new_snake[:,0], new_snake[:,1]=new_points[0], new_points[1]
        curr_snake=new_snake
        t+=1
    return new_snake

def snake_to_mask(snake, image_shape):
    snake=np.copy(snake)
    snake[:, 0], snake[:, 1]=np.copy(snake[:, 1]), np.copy(snake[:, 0])
    return 255*np.array(skimage.draw.polygon2mask(image_shape, snake), 
                        np.uint8)

def intersection(mask1, mask2):
    mask=np.zeros(mask1.shape)
    mask[(mask1==255) & (mask2==255)]=255
    return mask

def union(mask1, mask2):
    mask=np.zeros(mask1.shape)
    mask[(mask1==255) | (mask2==255)]=255
    return mask

def IoU(mask1, mask2):
    return intersection(mask1, mask2).sum()/union(mask1, mask2).sum()
    

def main():
    namespace=create_parser().parse_args()
    input_image=np.array(skimage.io.imread(namespace.input_image, as_gray=True), dtype=float)
    initial_snake=np.loadtxt(namespace.initial_snake, dtype=float)[:-1]
    alpha=namespace.alpha
    beta=namespace.beta
    tau=namespace.tau
    w_line=namespace.w_line
    w_edge=namespace.w_edge
    b_factor=namespace.kappa
    t_max=2000

    print(b_factor)
    result_snake=move_snake(input_image, initial_snake, alpha=alpha, beta=beta, tau=tau,
                            w_line=w_line, w_edge=w_edge, gvf=True,
                            b_factor=b_factor,
                            t_max=t_max)
    utils.display_snake(input_image, initial_snake, result_snake)
    skimage.io.imsave(namespace.output_image, snake_to_mask(result_snake, input_image.shape))

    # mask=skimage.io.imread(namespace.output_image, as_gray=True)
    # true_mask=skimage.io.imread(os.path.splitext(namespace.input_image)[0]+'_mask'+
    #     os.path.splitext(namespace.input_image)[1])
    # print(f'IoU(mask, true_mask)={IoU(mask, true_mask)}')


if __name__=='__main__':
    main()
