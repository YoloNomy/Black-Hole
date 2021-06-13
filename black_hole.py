#Module to simualate black hole
#Matplotlib function from :
#https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
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
    return [u_0], [phi_0_rad + np.pi], [dudphi_0], alpha_rad

def traj_sim(u, phi, dudphi, alpha_rad, dist_max = 10*rs):
    not_escape = True
    compt = 0
    while not_escape:
        phi_new, u_new, dudphi_new = euler_evolution(phi[-1], u[-1], dudphi[-1], sg=np.sign(alpha_rad))
        phi.append(phi_new)
        u.append(u_new)
        dudphi.append(dudphi_new)

        if 1/u_new <= 1:
            not_escape = False
            u[-1] = 1
        elif 1/u_new > dist_max:
            not_escape = False

        if compt >= 1e5:
            break
        compt += 1
    u = np.asarray(u)
    phi = np.asarray(phi)
    dudphi = np.asarray(dudphi)
    return u, phi, dudphi

def launch_sim_2D(phi_0, r_0,  alpha, show=True, ax=None):

    u_ini, phi_ini, dudphi_ini, alpha_rad = init_sim(phi_0, r_0, alpha)
    u, phi, dudphi = traj_sim(u_ini, phi_ini, dudphi_ini, alpha_rad)

    x = 1/u * np.cos(phi)
    y = 1/u * np.sin(phi)

    if show:
        fig, ax = plt.subplots()
    ax.plot(x, y, color='r')
    if show:
        circle=plt.Circle((0,0),1,color='k')
        ax.add_patch(circle)
        ax.axis('equal')
        plt.show()

def plot_multi_traj_2D(r_0, alpha_min = 0, alpha_max = 50, step=1):
    fig, ax = plt.subplots()
    for alpha in np.arange(alpha_min, alpha_max, step):
        launch_sim_2D(phi_0=0, r_0=r_0, alpha=alpha, show=False, ax=ax)
    circle=plt.Circle((0,0), 1, color='k')
    ax.add_patch(circle)
    ax.axis('equal')
    plt.show()

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

def plot_sphere(ax):
    # Generate and plot a unit sphere
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v)) # np.outer() -> outer vector product
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='k')

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def launch_sim_3D(phi_0, r_0, alpha, theta, show=True, ax=None):
    theta_rad = np.radians(theta)
    u_ini, phi_ini, dudphi_ini, alpha_rad = init_sim(phi_0, r_0, alpha)
    u, phi, dudphi = traj_sim(u_ini, phi_ini, dudphi_ini, alpha_rad)

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    x = 1/u * np.cos(phi)
    y = 1/u * np.sin(phi)

    x, y, z = rot_x(theta_rad, x, y)
    ax.plot(x, y, z, color='r', zorder=3)

    if show:
        plot_sphere(ax)
        ax.set_box_aspect([1,1,1])
        set_axes_equal(ax)
        plt.show()


def plot_multi_traj_3D(r_0, alpha = 10, theta_min = 0, theta_max = 360, step=20):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for theta in np.arange(theta_min, theta_max, step):
        launch_sim_3D(phi_0=0, r_0 = r_0, alpha=alpha, theta=theta, show=False, ax=ax)
    plot_sphere(ax)
    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)
    plt.show()

@njit(cache=True)
def launch_compute_image(image, d_im, d_obs, length=10):
    after_hole = np.zeros(shape=image.shape)
    alpha_grid = np.linspace(-90,90,2)
    theta_grid = np.linspace(0,379,2)
    h_im_metric = np.linspace(-length/2, length/2, image.shape[1]+1)
    width = length * image.shape[0]/image.shape[1]
    v_im_metric = np.linspace(-width/2, -width/2, image.shape[0]+1)
    intensity = image/(len(alpha_grid)*len(theta_grid))

    for c in range(1):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for alpha in alpha_grid:
                    for theta in theta_grid:
                        theta_rad = np.radians(theta)
                        phi_0 = np.arctan2(0.5 * (h_im_metric[i] + h_im_metric[i+1]),
                                           0.5 * (v_im_metric[i] + v_im_metric[i+1]))
                        u_ini, phi_ini, dudphi_ini, alpha_rad = init_sim(phi_0, d_im, alpha)
                        coord_idx = traj_sim_image(phi_ini[0], u_ini[0], dudphi_ini[0], d_obs, alpha_rad, theta_rad, v_im_metric, h_im_metric, intensity[i,j])
                        if coord_idx != [-1]:
                            after_hole[coord_idx[0], coord_idx[1], c] += intensity[i, j, c]
    return after_hole

def compute_image(im_path, d_im, d_obs, length=10):
    image = np.asarray(Image.open(im_path))
    after_hole = launch_compute_image(image, d_im, d_obs, length=length)
    plt.imshow(after_hole)
    plt.show()

@njit(cache=True)
def traj_sim_image(phi_ini, u_ini, dudphi_ini, d_obs, alpha_rad, theta_rad, v_im_metric, h_im_metric, intensity):
    not_escape = True
    compt = 0
    phi, u, dudphi = phi_ini, u_ini, dudphi_ini
    while not_escape:
        phi, u, dudphi = euler_evolution(phi, u, dudphi, sg=np.sign(alpha_rad))
        if 1/u <= 1:
            not_escape = False
            res = [-1]
            break
        elif 1/u * np.cos(phi) >= d_obs:
            not_escape = False
            x = 1/u * np.cos(phi)
            y = 1/u * np.sin(phi)
            x, y, z = rot_x(theta_rad, [x], [y])
            x, y, z = x[0], y[0], z[0]
            if (y >= h_im_metric[0] and y <= h_im_metric[-1]) and (z >= v_im_metric[0] and z <= v_im_metric[-1]):
                for i in range(len(v_im_metric[:-1])):
                    if z >= v_im_metric[i] and z <= v_im_metric[i+1]:
                        v_idx = i
                        break
                for i in range(len(h_im_metric[:-1])):
                    if y >= h_im_metric[i] and y <= h_im_metric[i+1]:
                        h_idx = i
                        break
                res = [v_idx, h_idx]
            else:
                res = [-1]
        elif compt >= 1e6:
            not_escape = False
            res = [-1]
        else:
            compt += 1
    return res
