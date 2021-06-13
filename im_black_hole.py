import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numba import njit

rs = 1

@njit(cache=True)
def euler_evolution(phi, u, dudphi, sg=1):
    dphi=1e-3
    dudphi_new = dudphi + (3/2*rs*u**2 - u) * dphi * sg
    u_new = u + dudphi_new * dphi * sg
    phi_new = phi + dphi * sg
    return phi_new, u_new, dudphi_new

@njit(cache=True)
def init_sim(phi_0, r_0, alpha):
    alpha_rad = -np.radians(alpha)
    phi_0_rad = np.radians(phi_0)
    if alpha_rad == 0:
        alpha_rad =  0.001
        if alpha < 0 :
            alpha_rad *= -1
    u_0 = 1/r_0
    dudphi_0 = u_0/np.tan(alpha_rad)
    return  phi_0_rad + np.pi, u_0, dudphi_0, alpha_rad

@njit(cache=True)
def rot_x(r_angle, x_list, y_list):
    R = np.array([[1.,  0,  0],
                 [0 ,  np.cos(r_angle), -np.sin(r_angle)],
                 [0 ,  np.sin(r_angle),  np.cos(r_angle)]])
    new_x = []
    new_y = []
    new_z = []
    for x, y in zip(x_list, y_list):
        vec = np.array([x, y, 0])
        x_tmp, y_tmp, z_tmp = R @ vec
        new_x.append(x_tmp)
        new_y.append(y_tmp)
        new_z.append(z_tmp)

    return new_x, new_y, new_z

def compute_image(image_path,d_im, d_obs, obs_length_npix = 10, length = 10):
    image = np.asarray(Image.open(image_path))
    obs_width_npix = int(obs_length_npix * image.shape[0]/image.shape[1])
    after_hole = np.zeros(shape=(obs_width_npix, obs_length_npix, image.shape[2]))
    width = length * image.shape[0]/image.shape[1]
    dwidth_obs = width / obs_width_npix
    dlength_obs = length / obs_length_npix
    dwidth_im = width / image.shape[0]
    dlength_im = length / image.shape[1]

    for i in range(after_hole.shape[0]):
        for j in range(after_hole.shape[1]):
            for theta in np.linspace(-7, 7, 10):
                for alpha in np.linspace(-7, 7, 10):
                    theta_rad = np.radians(theta)
                    phi_0 = np.atan2()
                    phi, u, dudphi, alpha_rad = init_sim(phi_0, d_obs, alpha)

@njit(cache=True)
def run_sim(phi, u, dudphi, alpha_rad, theta_rad, dlength_obs, dwidth_obs, dlength_im,  dwidth_im):

    for n in range(1e7):
        phi_tmp, u_tmp, dudphi_tmp = euler_evolution(phi, u, dudphi, sq=np.sign(alpha_rad))
        x = 1/u_tmp * np.cos(phi_tmp)
        y = 1/u_tmp * np.sin(phi_tmp)

        x, y, z = rot_x(theta_rad, [x], [y])

        if
