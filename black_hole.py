"""This code allow to simulate light trajectories around a Schwarchild Black hole
This was inspired by https://www.youtube.com/watch?v=PjWjZFwz3rQ&t=460s
and the image code is fork from https://github.com/Python-simulation/Black-hole-simulation-using-python
(with some personal modifications)
"""

__version__ = '1.0.0'

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import interp1d
import time
import warnings
warnings.filterwarnings("ignore")

def plot_sphere(ax):
    """Plot a 3D sphere on ax.

    Parameters
    ----------
    ax : matplotlib.Axes.ax
        The matplotlib ax on which plot the sphere.

    Notes
    -------
        Taken from stackoverflow (I've to retrieve the exact link)
    """

    # Generate and plot a unit sphere
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v)) # np.outer() -> outer vector product
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='k')

def _set_axes_radius(ax, origin, radius):
    """Set 3D axes equal aspect.

    Parameters
    ----------
    ax : matplotlib.Axes.ax
        matplotlib axe on which apply the code.
    origin : numpy.array(float, float, float)
        Origin point.
    radius : float
        Half-size of the axis.

    Notes
    -------
        Taken from stackoverflow (I've to retrieve the exact link)

    """
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.

    Notes
    -----
        Taken from stackoverflow (I've to retrieve the exact link)
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
    """Return the rotation matrix around x-axis.

    Parameters
    ----------
    r_angle : float
        The rotation angle in radians.

    Returns
    -------
    numpy.ndarray(float, shape=(3,3))
        The corresponding rotation matrix.

    """
    R = np.array([[1.,  0,  0],
                  [0 ,  np.cos(r_angle), -np.sin(r_angle)],
                  [0 ,  np.sin(r_angle),  np.cos(r_angle)]])
    return R

def rot_vec_x(r_angle, x, y, z):
    """Transform coordinates with a x-axis rotation.

    Parameters
    ----------
    r_angle : float
        Rotation angle.
    x : numpy.ndarray(float)
        X coordinates.
    y : numpy.ndarray(float)
        Y coordinates.
    z : numpy.ndarray(float)
        Z coordinates.

    Returns
    -------
    numpy.ndarray(float), numpy.ndarray(float), numpy.ndarray(float)
        New coordinates after rotation.

    """
    R = rot_x(r_angle)
    new_x = R[0, 0] * x
    new_y = R[1, 1] * y + R[1, 2] * z
    new_z = R[2, 1] * y + R[2, 2] * z
    return new_x, new_y, new_z

def sph2cart(phi, theta):
    """Convert unit sphere coordinates into carthesians coordinates.

    Parameters
    ----------
    phi : float
        Angle in xy-plane, x-axis corresponding to 0.
    theta : float
        Angle with the z-axis.

    Returns
    -------
    numpy.ndarray(float), numpy.ndarray(float), numpy.ndarray(float)
        Carthesians coordinates.

    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def pol2cart(r, theta):
    """Polar to carthesians coordinates.

    Parameters
    ----------
    r : numpy.ndarray(float)
        The radial coordinate.
    theta : float
        Angle with respect to x-axis.

    Returns
    -------
    numpy.ndarray(float), numpy.ndarray(float)
        Carthesians coordinates.

    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

class BlackHole:
    """The Black Hole simulation class.

    Parameters
    ----------
    G : float, opt (default = 1)
        Graviational constant.
    M : float, opt (default = 1/2)
        Black Hole mass.
    c : float, opt (default = 1)
        Light velocity.
    dobs : float, opt (default = 10)
        Distance of the observer from the black hole.
    phi_obs : float, opt (default = 0)
        Angle of the observer.

    Methods
    -------
    traj_2D :
        Description of attribute `_G`.
    _M : type
        Description of attribute `_M`.
    _c : type
        Description of attribute `_c`.
    _dphi : type
        Description of attribute `_dphi`.
    _dobs : type
        Description of attribute `_dobs`.
    compute_min_angle : type
        Description of attribute `compute_min_angle`.
    phi_obs

    """
    def __init__(self, G = 1, M = 1/2, c = 1, dobs = 10, phi_obs = 0, interp_res = 2e-2):
        self._G = G
        self._M = M
        self._c = c
        self._dphi = 1e-3
        self._dobs = dobs
        self.phi_obs = phi_obs
        self.compute_min_angle(1e-4)

    @property
    def rs(self):
        """Get Schwarchild Radius"""
        return 2 * self.G * self.M / self.c**2

    @property
    def alpha_min(self):
        """Get minimal incidence angle"""
        return self._alpha_min

    @property
    def dobs(self):
        """Get observer distance"""
        return self._dobs

    @dobs.setter
    def dobs(self, val):
        """Set observer distance"""
        if val > self.rs:
            self._dobs = val
            self.compute_min_angle(1e-3)
        else:
            print(f"dobs must be larger than the Schwarchild Radius {self.rs}")

    @property
    def c(self):
        """Get light velocity"""
        return self._c

    @c.setter
    def c(self, val):
        """Set light velocity"""
        self._c = val

    @property
    def G(self):
        """Get Graviational constant"""
        return self._G

    @G.setter
    def G(self, val):
        """Set Graviational constant"""
        self._G = val

    @property
    def M(self):
        """Get Black Hole mass"""
        return self._M

    @M.setter
    def M(self, val):
        """Set Black Hole mass"""
        self._M = val

    def _euler_evolution(self, phi, u, dudphi):
        """Compute light evolution with euler approimation.

        Parameters
        ----------
        phi : float
            Actual phi coordinate of the photon.
        u : float
            1/r with r the radial coordinate of the photon.
        dudphi : float
            Derivate of u with respect to phi.

        Returns
        -------
        float, float, float
            The new coordinates of the photon.

        """
        dudphi_new = dudphi + (3/2 * self.rs * u**2 - u) * self._dphi
        u_new = u + dudphi_new * self._dphi
        phi_new = phi + self._dphi
        return phi_new, u_new, dudphi_new

    def _traj_sim(self, alpha, all_traj = False, alpha_rad = True):
        """Compute the trajectory of the photon.

        Parameters
        ----------
        alpha : float
            Initial incidence angle.
        all_traj : bool, opt (default = False)
            True for return all the trajectory points.
        alpha_rad : bool, opt (default = True)
            Is the input alpha angle in radians?

        Returns
        -------
        (float, float) or (numpy.ndarray(float), numpy.ndarray(float))
            Either the final phi and r coordinates or all the trajectory coordinates.

        """
        phi = np.radians(self.phi_obs)
        if not alpha_rad:
            alpha = np.radians(alpha)
        if alpha == 0:
            alpha = 0.0001
        u = 1 / self.dobs
        dudphi = 1 / (self.dobs * np.tan(alpha))
        dont_stop = True
        compt = 0
        if all_traj:
            phi_list = [self.phi_obs]
            r_list = [self.dobs]

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
            elif abs(1 / u_tmp) > 15 * self.rs:
                res = (phi_tmp, 1/u_tmp)
                dont_stop = False
            elif compt > 1e6:
                res = (-1, -1)
                dont_stop = False
            compt += 1
            phi, u, dudphi = phi_tmp, u_tmp, dudphi_tmp
        if all_traj:
            res = (np.array(phi_list), np.array(r_list))
        return res

    def _draw_hole(self, ax, dim):
        """Used to draw the hole in matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes
            The matplotlib ax.
        dim : str
            '2D' or '3D'.

        """
        if dim == '2D':
            ax.add_patch(plt.Circle((0,0), self.rs, color='k', zorder=3, transform=ax.transData._b))
        elif dim == '3D':
            plot_sphere(ax)

    def traj_2D(self, alpha, show=True, ax=None, polar = False):
        """Compute a 2D trajectory.

        Parameters
        ----------
        alpha : float
            Initial incident angle in degrees.
        show : bool, opt (default = True)
            Show or not the plot.
        ax : matplotlib.Axes, opt (default = None)
            Used in multi_traj_2D.
        polar : bool, opt (default = False)
            Plot in polar coordinates.

        """
        x, y = self._traj_sim(abs(alpha), all_traj = True, alpha_rad = False)
        if not polar:
            x, y = pol2cart(y, x)
        if alpha < 0 and not polar:
            y *= -1
        elif alpha < 0 and polar:
            x *= -1
        if show:
            projection = None
            if polar:
                projection = 'polar'
            fig = plt.figure()
            ax = fig.add_subplot(projection = projection)

        ax.plot(x, y, color='r')

        if show:
            self._draw_hole(ax, dim ='2D')
            if not polar:
                ax.axis('equal')
            plt.show()

    def multi_traj_2D(self, alpha_min = 0, alpha_max = 50, step=1, polar = False):
        """Plot 2D trajectories for a range of incident angle.

        Parameters
        ----------
        alpha_min : float, opt (default = 0)
            Minimal incident angle in degrees.
        alpha_max : float, opt (default = 50)
            Maximal incident angle in degrees.
        step : float, opt (default = 1)
            step between incident angle.
        polar : bool, opt (default = False)
            Plot in polar coordinates.
        """
        projection = None
        if polar:
            projection = 'polar'
        fig = plt.figure()
        ax = fig.add_subplot(projection = projection)
        for alpha in np.arange(alpha_min, alpha_max, step):
            self.traj_2D(alpha = alpha, show = False, ax = ax, polar = polar)
        self._draw_hole(ax, dim='2D')
        if not polar:
            ax.axis('equal')
        plt.show()

    def traj_3D(self, alpha, theta, show=True, ax=None):
        """Compute a 3D trajectory.

        Parameters
        ----------
        alpha : float
            Initial incident angle in degrees.
        theta : float
            Rotation around x-axis in degrees.
        show : bool, opt (default = True)
            Show or not the plot.
        ax : matplotlib.Axes, opt (default = None)
            Used in multi_traj_2D.
        polar : bool, opt (default = False)
            Plot in polar coordinates.

        """
        phi, r = self._traj_sim(alpha, all_traj = True, alpha_rad = False)

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
            self._draw_hole(ax, dim='3D')
            ax.set_box_aspect([1,1,1])
            set_axes_equal(ax)
            plt.show()

    def multi_traj_3D(self, alpha = 10, theta_min = 0, theta_max = 360, step=20):
        """Plot 3D trajectories for a range of angle around x-axis.

        Parameters
        ----------
        alpha : float, opt (default = 10)
            Initial incident angle in degrees.
        theta_min : float, opt (default = 0)
            Minimal x angle in degrees.
        theta_max : float, opt (default = 360)
            Maximal x angle in degrees.
        step : float, opt (default = 20)
            step between incident angle.
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for theta in np.arange(theta_min, theta_max, step):
            self.traj_3D(alpha = alpha, theta = theta, show=False, ax=ax)
        self._draw_hole(ax, dim='3D')
        ax.set_box_aspect([1,1,1])
        set_axes_equal(ax)
        plt.show()

    def _min_angle_dicothomie(self, alpha1, alpha2, resolution):
        """Dichotomie algorithm to found minimal incident angle.

        Parameters
        ----------
        alpha1 : float
            The upper bound.
        alpha2 : float
            The lower bound.
        resolution : float
            Precision to achieve.

        Returns
        -------
        float
            The minimal angle for which the photon isn't captured by the Black Hole.

        """
        alpha = 0.5 * (alpha1 + alpha2)
        res = self._traj_sim(alpha)
        if abs(alpha2 - alpha1) < resolution:
            return alpha1, alpha2
        elif res == (0, 0):
            return self._min_angle_dicothomie(alpha, alpha2, resolution)
        else:
            return self._min_angle_dicothomie(alpha1, alpha, resolution)

    def compute_min_angle(self, precision, verbose = False):
        """Initialize the dichotomie.

        Parameters
        ----------
        precision : float
            Precision to achieve.
        verbose : bool, opt (default = False)
            To print some informations.
        """
        if verbose:
            print(f"Min angle computation with {precision} precision")
            stime = time.time()

        alpha1 = np.arctan(self.rs / self.dobs)

        res1 = self._traj_sim(alpha1)
        alpha2 = 2 * alpha1
        res2 = self._traj_sim(alpha2)
        step = 0.1

        while res1 != (0, 0):
            alpha1 -= step
            res1 = self._traj_sim(alpha1)
        while res2 == (0,0):
            if res2 == (0, 0):
                alpha2 += step
            else:
                alpha2 -= step
            res2 = self._traj_sim(alpha2)

        a1, a2 = self._min_angle_dicothomie(alpha1, alpha2, precision)
        if verbose:
            print(f"Min angle computed in {time.time() - stime} seconds")
        self._alpha_min = a2

    def _compute_traj_grid(self, alpha_min_res, grid_n_angle):
        """Compute an interpolation deviated_angle = f(seen_angle).

        Parameters
        ----------
        alpha_min_res : float
            Precision to achieve on alpha_min.
        grid_n_angle : int
            Number of seen angle to compute.

        Returns
        -------
        scipy.interpolate.interp1d
            The deviated_angle = f(seen_angle) interpolation.

        """
        self.compute_min_angle(alpha_min_res, verbose = True)
        alpha_grid = np.linspace(self.alpha_min, np.pi, grid_n_angle)
        seen_angle = []
        aft_hole_angle = []
        for alpha in alpha_grid:
            if alpha == np.pi:
                seen_angle.append(np.pi - alpha)
                aft_hole_angle.append(np.pi - alpha)
            else:
                res = self._traj_sim(alpha)
                if res != (0, 0) and res != (-1, -1):
                    seen_angle.append(np.pi - alpha)
                    aft_hole_angle.append(res[0] + np.arcsin(self.dobs / res[1] * np.sin(res[0])))

        return interp1d(seen_angle, aft_hole_angle, bounds_error=False)

    @staticmethod
    def _open_image(img_path, n_pix_x):
        """Open the image file.

        Parameters
        ----------
        img_path : str
            Path to the image.
        n_pix_x : int
            Number of horizontal pixel in the final image..

        Returns
        -------
        PIL.Image
            A PIL image object.

        """
        img = Image.open(img_path)
        img = img.convert('RGB')

        X_size = int(n_pix_x)
        Y_size = int(img.size[1] * n_pix_x / img.size[0])

        X_size -= 1 * (X_size % 2 != 0)
        Y_size -= 1 * (Y_size % 2 != 0)

        img = img.resize((X_size, Y_size), Image.ANTIALIAS)
        return img

    @staticmethod
    def _create_matrix(X_size, Y_size, interp):
        """Compute the matrix of deviated light.

        Parameters
        ----------
        X_size : int
            Image X size.
        Y_size : int
            Image Y size.
        interp : scipy.interpolate.interp1d
            The deviated_angle = f(seen_angle) interpolation.

        Returns
        -------
        numpy.ndarray(shape = img.size)
            An array that contains the corresponding pixel index of the initial image.

        """
        X, Y = np.meshgrid(np.arange(0, X_size), np.arange(0, Y_size))
        rad_by_pix_x = 2 * np.pi /  X_size
        rad_by_pix_y = np.pi / Y_size
        phi = X * rad_by_pix_x
        theta = Y * rad_by_pix_y
        x, y, z = sph2cart(phi, theta)

        with np.errstate(all='ignore'):
            beta = -np.arctan2(z, y)

        new_x, new_y, new_z = rot_vec_x(beta, x, y, z)

        seen_angle = np.mod(np.arctan2(new_y, new_x), 2 * np.pi)
        deviated_angle = np.zeros(seen_angle.shape)

        deviated_angle[seen_angle < np.pi] = interp(seen_angle[seen_angle < np.pi])
        deviated_angle[seen_angle >= np.pi] = 2 * np.pi - interp(2 * np.pi - seen_angle[seen_angle >= np.pi])

        u, v, w = sph2cart(deviated_angle, np.pi / 2)

        new_u, new_v, new_w = rot_vec_x(-beta, u, v, w)

        new_phi =  np.mod(np.arctan2(new_v, new_u), 2 * np.pi)
        new_theta = np.mod(np.arccos(new_w), np.pi)
        #new_phi[new_phi == 2 * np.pi] = 0

        img_aft_hole_x = new_phi / rad_by_pix_x
        img_aft_hole_y = new_theta / rad_by_pix_y

        #Black hole is not in interpolation range, result in nan values
        img_aft_hole_x[np.isnan(img_aft_hole_x)] = -1
        img_aft_hole_y[np.isnan(img_aft_hole_y)] = -1

        return np.array(img_aft_hole_x, dtype=int), np.array(img_aft_hole_y, dtype=int)

    @staticmethod
    def _get_img_aft_hole(img, img_aft_hole_x, img_aft_hole_y):
        """Generate the after black hole image.

        Parameters
        ----------
        image : PIL.Image
            The initial image.
        img_aft_hole_x : numpy.ndarray(float)
            The x index of the initial image pixel in the aft hole image.
        img_aft_hole_y : numpy.ndarray(float)
            The y index of the initial image pixel in the aft hole image.

        Returns
        -------
        PIL.Image
            After Black Hole image.

        """
        pixels = np.array(img)
        pixels_aft_hole = np.array(img)

        # locate pixels outside of the image
        img_aft_hole_x[img_aft_hole_x >= img.size[0]] = -2
        img_aft_hole_y[img_aft_hole_y >= img.size[1]] = -2


        pixels_aft_hole = pixels[img_aft_hole_y, img_aft_hole_x]  # apply the black hole deformation
        pixels_aft_hole[img_aft_hole_x == -1] = [0, 0, 0]  # color the black hole in black

        pixels_aft_hole[img_aft_hole_y == -2] = [255, 192, 203]  # color pixels outside
        pixels_aft_hole[img_aft_hole_x == -2] = [255, 192, 203]
        pixels_aft_hole.astype(np.uint8)
        return Image.fromarray(pixels_aft_hole, 'RGB')

    def image(self, img_path, n_pix_x = 5000, grid_n_angle = 150):
        """Compute and display the image after black hole.

        Parameters
        ----------
        img_path : str
            Path of the initial image.
        n_pix_x : int
            Number of horizontal pixels in the final image.
        grid_n_angle : int
            Number of points in the interpolation.
        """
        
        stime = time.time()
        img  = self._open_image(img_path, n_pix_x)

        interp = self._compute_traj_grid(2 * np.pi / img.size[0], grid_n_angle)

        img_aft_hole_x, img_aft_hole_y = self._create_matrix(img.size[0], img.size[1], interp)
        img_after_hole = self._get_img_aft_hole(img, img_aft_hole_x, img_aft_hole_y)

        print(f"Image computed in {time.time()-stime} s")
        plt.figure()
        plt.imshow(img_after_hole)
        plt.show()
