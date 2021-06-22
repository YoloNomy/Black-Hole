import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import interp1d
from numba import njit
import time

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
    new_phi[new_phi == 2 * np.pi] = 0
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

        pixels_aft_hole = pixels[img_aft_hole_y, img_aft_hole_x]  # apply the black hole deformation
        pixels_aft_hole[img_aft_hole_x == -1] = [0, 0, 0]  # color the black hole in black
        pixels_aft_hole[img_aft_hole_y == -2] = [255, 192, 203]  # color pixels outside
        pixels_aft_hole[img_aft_hole_x == -2] = [255, 192, 203]
        pixels_aft_hole.astype(np.uint8)

        return Image.fromarray(pixels_aft_hole, 'RGB')


@njit(cache=True)
def pol2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

class BlackHole:
    def __init__(self, G = 1, M = 1/2, c = 1, dobs = 10, phi_obs = 0):
        self._G = G
        self._M = M
        self._c = c
        self._dphi = 1e-3
        self._dobs = dobs
        self.phi_obs = phi_obs
        self.compute_min_angle(1e-4)

    @property
    def rs(self):
        return 2 * self.G * self.M / self.c**2

    @property
    def alpha_min(self):
        return self._alpha_min

    @property
    def dobs(self):
        return self._dobs

    @dobs.setter
    def dobs(self, val):
        if val > self.rs:
            self._dobs = val
            self.compute_min_angle(1e-4)
        else:
            print(f"dobs must be larger than the Schwarchild Radius {self.rs}")

    @property
    def c(self):
        return self.c

    @c.setter
    def c(self, val):
        self._c = val

    @property
    def G(self):
        return self._G

    @G.setter
    def G(self, val):
        self._G = val

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, val):
        self._M = val

    @rs.setter
    def rs(self, val):
        if val > 0:
            self._rs = val
        else:
            print("rs must be positive")

    def _euler_evolution(self, phi, u, dudphi):
        dudphi_new = dudphi + (3/2 * self.rs * u**2 - u) * self._dphi
        u_new = u + dudphi_new * self._dphi
        phi_new = phi + self._dphi
        return phi_new, u_new, dudphi_new

    def _traj_sim(self, alpha, all_traj=False):
        phi = np.radians(self.phi_obs)
        alpha = np.radians(alpha)
        if alpha == 0:
            alpha = 0.0001
        u = 1 / self.dobs
        dudphi = 1 / (self.dobs * abs(np.tan(alpha)))
        dont_stop = True
        compt = 0
        if all_traj:
            phi_list = [phi]
            r_list = [r0]

        while dont_stop:
            phi_tmp, u_tmp, dudphi_tmp = self._euler_evolution(phi, u, dudphi)

            if all_traj:
                phi_list.append(phi_tmp)
                r_list.append(1 / u_tmp)

            if 1 / u_tmp <= self.rs:
                res = (0, 0)
                dont_stop = False
                if all_traj:
                    r_list[-1] = self.rs
            elif abs(1/u_tmp) > 15 * self.rs:
                res = (phi_tmp, 1/u_tmp)
                dont_stop = False
            elif compt > 1e6:
                res = (-1, -1)
                dont_stop = False
            compt += 1
            phi, u, dudphi = phi_tmp, u_tmp, dudphi_tmp
        if all_traj:
            res = (phi_list, r_list)
        return res

    def draw_hole(self, ax, dim):
        if dim == '2D':
            ax.add_patch(plt.Circle((0,0), self.rs, color='k'))
        elif dim == '3D':
            plot_sphere(ax)

    def traj_2D(self, alpha, show=True, ax=None):
        phi, r = self._traj_sim(abs(alpha), all_traj = True)
        x, y = pol2cart(r, phi)
        if alpha < 0:
            y *= -1
        if show:
            fig, ax = plt.subplots()

        ax.plot(x, y, color='r')

        if show:
            self.draw_hole(ax, dim ='2D')
            ax.axis('equal')
            plt.show()

    def multi_traj_2D(self, alpha_min = 0, alpha_max = 50, step=1):
        fig, ax = plt.subplots()
        for alpha in np.arange(alpha_min, alpha_max, step):
            self.traj_2D(alpha = alpha, show = False, ax = ax)
        self.draw_hole(ax, dim='2D')
        ax.axis('equal')
        plt.show()

    def traj_3D(self, alpha, theta, show=True, ax=None):
        phi, r = self._traj_sim(alpha, all_traj = True)

        if show:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

        x, y = pol2cart(r, phi)
        z = np.zeros(x.size)
        R = rot_x(np.radians(theta))
        vec = np.array([x, y, z])

        x, y, z = R @ vec

        ax.plot(x, y, z, color='r')

        if show:
            self.draw_hole(ax, dim='3D')
            ax.set_box_aspect([1,1,1])
            set_axes_equal(ax)
            plt.show()

    def multi_traj_3D(self, alpha = 10, theta_min = 0, theta_max = 360, step=20):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for theta in np.arange(theta_min, theta_max, step):
            self.traj_3D(alpha = alpha, theta = theta, show=False, ax=ax)
        self.draw_hole(ax, dim='3D')
        ax.set_box_aspect([1,1,1])
        set_axes_equal(ax)
        plt.show()

    def min_angle_dicothomie(alpha1, alpha2, resolution):
        alpha = 0.5 * (alpha1 + alpha2)
        res = run_sim(0, 1 / self.dobs, 1 / (self.dobs * np.tan(alpha)))
        if abs(alpha2 - alpha1) < resolution:
            return alpha1, alpha2
        elif res == (0, 0):
            return min_angle_dicothomie(alpha, alpha2, resolution, self.dobs)
        else:
            return min_angle_dicothomie(alpha1, alpha, resolution, self.dobs)

    def compute_min_angle(precision, verbose = False):
        if verbose:
            print(f"Min angle computation with {precision} precision")
            stime = time.time()

        alpha1 = np.arctan(self.rs / self.dobs)

        res1 = run_sim(0, 1/self.dobs, 1/(self.dobs * np.tan(alpha1)))
        alpha2 = 2 * alpha1
        res2 = run_sim(0, 1/self.dobs, 1/(self.dobs * np.tan(alpha2)))
        step = 0.1

        while res1 != (0, 0):
            alpha1 -= step
            res1 = run_sim(0,  1/self.dobs, 1/(self.dobs * np.tan(alpha1)))
        while res2 == (0,0):
            if res2 == (0, 0):
                alpha2 += step
            else:
                alpha2 -= step
            res2 = run_sim(0,  1 / self.dobs, 1 / (self.dobs * np.tan(alpha2)))

        a1, a2 = min_angle_dicothomie(alpha1, alpha2, precision)
        if verbose:
            print(f"Min angle computed in {time.time() - stime} seconds")
        self._alpha_min = a2

    def _compute_traj_grid(alpha_min_res, grid_n_angle):
        self.compute_min_angle(alpha_min_res, verbose = True)
        alpha_grid = np.linspace(self.alpha_min, np.pi, grid_n_angle)
        seen_angle = []
        aft_hole_angle = []
        for alpha in alpha_grid:
            res = run_sim(0,  1 / self.dobs, 1 / (self.dobs * np.tan(alpha)))
            if res != (0, 0) and res != (-1, -1):
                seen_angle.append(np.pi - alpha)
                aft_hole_angle.append(res[0] + np.arcsin(self.dobs / res[1] * np.sin(res[0])))
            elif alpha == np.pi:
                seen_angle.append(np.pi - alpha)
                aft_hole_angle.append(np.pi - alpha)
        return interp1d(seen_angle, aft_hole_angle, bounds_error=False)

    @staticmethod
    def _open_image(path, n_pix_x):
        img = Image.open(image_path)
        img = img.convert('RGB')

        X_size = int(n_pix_x)
        Y_size = int(image.size[1] * n_pix_x/image.size[0])

        X_size -= 1 * (X_size % 2 != 0)
        Y_size -= 1 * (Y_size % 2 != 0)

        img = img.resize((X_size, Y_size), Image.ANTIALIAS)
        return img

    def image(image_path, n_pix_x = 5000, grid_n_angle = 150):
        stime = time.time()
        img  = self._open_image(path, n_pix_x)

        # Give the delta angle by pixel
        img_res_x = img.size[0] / (2 * np.pi)
        img_res_y = img.size[1] / np.pi

        interp = self._compute_traj_grid(1 / img_res_x, grid_n_angle)

        img_aft_hole_x, img_aft_hole_y = create_matrix(X_size, Y_size, img_res_x, img_res_y, interp)
        img_after_hole = get_img(image, img_aft_hole_x, img_aft_hole_y, X_size, Y_size)

        print(f"Image computed in {time.time()-stime} s")
        plt.figure()
        plt.imshow(img_after_hole)
        plt.show()
