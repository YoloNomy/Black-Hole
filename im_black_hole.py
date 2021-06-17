import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import interp1d
from numba import njit
import progressbar

rs = 1

def euler_evolution(phi, u, dudphi):
    dphi=1e-3
    dudphi_new = dudphi + (3/2 * rs * u**2 - u) * dphi
    u_new = u + dudphi_new * dphi
    phi_new = phi + dphi
    return phi_new, u_new, dudphi_new

def rot_x(r_angle):
    R = np.asarray([[1.,  0,  0],
                    [0 ,  np.cos(r_angle), -np.sin(r_angle)],
                    [0 ,  np.sin(r_angle),  np.cos(r_angle)]])

    return R

def create_matrix(X_size, Y_size, img_res_x, img_res_y, interp):
    X, Y = np.meshgrid(np.arange(0, X_size), np.arange(0, Y_size))

    phi = X / img_res_x
    theta = Y / img_res_y

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    with np.errstate(all='ignore'):
        beta = -np.arctan(z/y)

    R1 = rot_x(beta)

    new_x = R1[0, 0] * x
    new_y = R1[1, 1] * y + R1[1, 2] * z
    new_z = R1[2, 1] * y + R1[2, 2] * z

    seen_angle = np.mod(np.arctan2(new_y, new_x), 2 * np.pi)

    deviated_angle = np.zeros(seen_angle.shape)

    deviated_angle[seen_angle < np.pi] = interp(seen_angle[seen_angle < np.pi])
    deviated_angle[seen_angle >= np.pi] = 2 * np.pi - interp(2 * np.pi - seen_angle[seen_angle >= np.pi])

    u = np.sin(np.pi / 2) * np.cos(deviated_angle)
    v = np.sin(np.pi / 2) * np.sin(deviated_angle)
    w = np.cos(np.pi / 2)

    R2 = rot_x(-beta)

    new_u = R2[0, 0] * u
    new_v = R2[1, 1] * v + R2[1, 2] * w
    new_w = R2[2, 1] * v + R2[2, 2] * w

    new_phi =  np.mod(np.arctan2(new_v, new_u), 2 * np.pi)
    new_theta = np.mod(np.arccos(new_w), np.pi)

    img_aft_hole_x = new_phi * img_res_x
    img_aft_hole_y = new_theta * img_res_y


    img_aft_hole_x[np.isnan(img_aft_hole_x)] = -1
    img_aft_hole_y[np.isnan(img_aft_hole_y)] = -1

    return np.array(img_aft_hole_x, dtype=int), np.array(img_aft_hole_y, dtype=int)

def get_img(image, img_aft_hole_x, img_aft_hole_y, X_size, Y_size):
        pixels = np.array(image)
        pixels_aft_hole = np.array(image)

        img_aft_hole_y[img_aft_hole_y >= Y_size] = -2  # locate pixels outside of the image
        img_aft_hole_x[img_aft_hole_x >= X_size] = -2

        pixels2 = pixels[img_aft_hole_y, img_aft_hole_x]  # apply the black hole deformation

        pixels_aft_hole[img_aft_hole_x == -1] = [0, 0, 0]  # color the black hole in black
        pixels_aft_hole[img_aft_hole_y == -2] = [255, 192, 203]  # color pixels outside
        pixels_aft_hole[img_aft_hole_x == -2] = [255, 192, 203]

        return Image.fromarray(pixels2.astype('uint8'), 'RGB')

def compute_image(image_path, d_obs=10, n_pix_x=5000):
    image = Image.open(image_path)
    image = image.convert('RGB')
    X_size = int(n_pix_x)
    Y_size = int(image.size[1] * n_pix_x/image.size[0])

    X_size -= 1 * (X_size % 2 != 0)
    Y_size -= 1 * (Y_size % 2 != 0)

    image = image.resize((X_size, Y_size), Image.ANTIALIAS)

    img_res_x = X_size / (2 * np.pi)
    img_res_y = Y_size / np.pi

    interp = compute_traj_grid(d_obs, 0.001)

    img_aft_hole_x, img_aft_hole_y = create_matrix(X_size, Y_size, img_res_x, img_res_y, interp)

    img_after_hole =  get_img(image, img_aft_hole_x, img_aft_hole_y, X_size, Y_size)

    plt.figure()
    plt.imshow(img_after_hole)
    plt.show()
    return 

def min_angle_dicothomie(alpha1, alpha2, resolution, d_obs):
    alpha = 0.5 * (alpha1 + alpha2)
    res = run_sim(0, 1 / d_obs, 1 / (d_obs * np.tan(alpha)))
    if abs(alpha2 - alpha1) < resolution:
        return alpha1, alpha2
    elif res == (0, 0):
        return min_angle_dicothomie(alpha, alpha2, resolution, d_obs)
    else:
        return min_angle_dicothomie(alpha1, alpha, resolution, d_obs)

def compute_min_angle(d_obs, res):
    print(f"Min angle computation with {res} resolution")
    alpha1 = np.arctan(rs / d_obs)

    res1 = run_sim(0, 1/d_obs, 1/(d_obs * np.tan(alpha1)))
    alpha2 = 2 * alpha1
    res2 = run_sim(0,  1/d_obs, 1/(d_obs * np.tan(alpha2)))
    step = 0.1

    while res1 != (0, 0):
        alpha1 -= step
        res1 = run_sim(0,  1/d_obs, 1/(d_obs * np.tan(alpha1)))
    while res2 == (0,0):
        if res2 == (0, 0):
            alpha2 += step
        else:
            alpha2 -= step
        res2 = run_sim(0,  1 / d_obs, 1 / (d_obs * np.tan(alpha2)))

    a1, a2 = min_angle_dicothomie(alpha1, alpha2, res, d_obs)
    return a2

def compute_traj_grid(d_obs, im_res):
    alpha_min = compute_min_angle(d_obs, im_res)
    alpha_grid = np.linspace(alpha_min, np.pi, 50)
    seen_angle = []
    aft_hole_angle = []
    for alpha in alpha_grid:
        res = run_sim(0,  1 / d_obs, 1 / (d_obs * np.tan(alpha)))
        if res != (0, 0) and res != (-1, -1):
            seen_angle.append(np.pi - alpha)
            aft_hole_angle.append(res[0] + np.arcsin(d_obs / res[1] * np.sin(res[0])))
    return interp1d(seen_angle, aft_hole_angle, bounds_error=False)

def run_sim(phi, u, dudphi):
    dont_stop = True
    compt = 0
    while dont_stop:
        phi_tmp, u_tmp, dudphi_tmp = euler_evolution(phi, u, dudphi)
        if 1/u_tmp <= 1:
            res = (0, 0)
            dont_stop = False
        elif abs(1/u_tmp) > 15 * rs:
            res = (phi_tmp, 1/u_tmp)
            dont_stop = False
        elif compt > 1e6:
            res = (-1, -1)
            dont_stop = False
        compt += 1
        phi, u, dudphi = phi_tmp, u_tmp, dudphi_tmp
    return res
