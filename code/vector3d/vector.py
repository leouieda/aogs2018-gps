"""
Green's functions and gridding for 3D vector elastic deformation
"""
import numpy as np
from sklearn.utils.validation import check_is_fitted

from verde.base import check_fit_input
from verde import Spline, get_region


class Vector3D(Spline):
    r"""

    Parameters
    ----------
    poisson : float
        The Poisson's ratio for the elastic deformation Green's functions.
        Default is 0.5. A value of -1 will lead to uncoupled interpolation of
        the east and north data components.
    fudge : float
        The positive fudge factor applied to the Green's function to avoid
        singularities.
    damping : None or float
        The positive damping regularization parameter. Controls how much
        smoothness is imposed on the estimated forces. If None, no
        regularization is used.
    shape : None or tuple
        If not None, then should be the shape of the regular grid of forces.
        See :func:`verde.grid_coordinates` for details.
    spacing : None or float or tuple
        If not None, then should be the spacing of the regular grid of forces.
        See :func:`verde.grid_coordinates` for details.
    region : None or tuple
        If not None, then the boundaries (``[W, E, S, N]``) used to generate a
        regular grid of forces. If None is given, then will use the boundaries
        of data given to the first call to :meth:`~verde.Vector2D.fit`.

    Attributes
    ----------
    forces_ : array
        The estimated forces that fit the observed data.
    force_coords_ : tuple of arrays
        The easting and northing coordinates of the forces.
    region_ : tuple
        The boundaries (``[W, E, S, N]``) of the data used to fit the
        interpolator. Used as the default region for the
        :meth:`~verde.Vector2D.grid` and :meth:`~verde.Vector2D.scatter`
        methods.

    """

    def __init__(self, poisson=0.5, depth=1000, fudge=1e-5, damping=None,
                 shape=None, spacing=None, region=None, flip_vertical=False):
        self.poisson = poisson
        self.depth = depth
        self.flip_vertical = flip_vertical
        super().__init__(fudge=fudge, damping=damping, shape=shape,
                         spacing=spacing, region=region)

    def fit(self, coordinates, data, weights=None):
        """
        Fit the gridder to the given 3-component vector data.

        The data region is captured and used as default for the
        :meth:`~verde.Vector3D.grid` and :meth:`~verde.Vector3D.scatter`
        methods.

        All input arrays must have the same shape.

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.
        data : tuple of array
            A tuple ``(east_component, north_component, up_component)`` of
            arrays with the vector data values at each point.
        weights : None or tuple array
            If not None, then the weights assigned to each data point. Must be
            one array per data component. Typically, this should be 1 over the
            data uncertainty squared.

        Returns
        -------
        self
            Returns this estimator instance for chaining operations.

        """
        coordinates, data, weights = check_fit_input(coordinates, data,
                                                     weights, unpack=False)
        if len(data) != 3:
            raise ValueError("Need three data components. Only {} given."
                             .format(len(data)))
        # Capture the data region to use as a default when gridding.
        self.region_ = get_region(coordinates[:2])
        self.force_coords_ = self._get_force_coordinates(coordinates)
        if any(w is not None for w in weights):
            weights = np.concatenate([i.ravel() for i in weights])
        else:
            weights = None
        self._check_weighted_exact_solution(weights)
        data = list(data)
        if self.flip_vertical:
            data[-1] *= -data[-1]
        data = np.concatenate([i.ravel() for i in data])
        jacobian = vector3d_jacobian(coordinates[:2], self.force_coords_,
                                     self.poisson, self.depth,
                                     fudge=self.fudge)
        self.force_ = self._estimate_forces(jacobian, data, weights)
        return self

    def predict(self, coordinates):
        """
        Evaluate the fitted gridder on the given set of points.

        Requires a fitted estimator (see :meth:`~verde.Vector3D.fit`).

        Parameters
        ----------
        coordinates : tuple of arrays
            Arrays with the coordinates of each data point. Should be in the
            following order: (easting, northing, vertical, ...). Only easting
            and northing will be used, all subsequent coordinates will be
            ignored.

        Returns
        -------
        data : tuple of arrays
            A tuple ``(east_component, north_component, up_component)`` of
            arrays with the predicted vector data values at each point.

        """
        check_is_fitted(self, ['force_', 'force_coords_'])
        jac = vector3d_jacobian(coordinates[:2], self.force_coords_,
                                self.poisson, self.depth, fudge=self.fudge)
        cast = np.broadcast(*coordinates[:2])
        npoints = cast.size
        components = jac.dot(self.force_).reshape((3, npoints))
        if self.flip_vertical:
            components[-1] *= -1
        return tuple(comp.reshape(cast.shape) for comp in components)


def vector3d_jacobian(coordinates, force_coordinates, poisson, depth,
                      fudge=1e-5, dtype="float32"):
    """

        |J_ee J_en J_ev| |f_e| |d_e|
        |J_ne J_nn J_nv|*|f_n|=|d_n|
        |J_ve J_vn J_vv| |f_v| |d_v|
    """
    force_coordinates = [np.atleast_1d(i).ravel()
                         for i in force_coordinates[:2]]
    coordinates = [np.atleast_1d(i).ravel() for i in coordinates[:2]]
    npoints = coordinates[0].size
    nforces = force_coordinates[0].size
    # Reshaping the data coordinates to a column vector will automatically
    # build a distance matrix between each data point and force.
    east, north = (datac.reshape((npoints, 1)) - forcec
                   for datac, forcec in zip(coordinates, force_coordinates))
    r = np.sqrt(east**2 + north**2 + depth**2)
    # Pre-compute common terms for the Green's functions of each component
    over_r = 1/r
    over_rz = 1/(r + depth)
    aux = (1 - 2*poisson)
    jac = np.empty((npoints*3, nforces*3), dtype=dtype)
    # J_ee
    jac[:npoints, :nforces] = over_r*(1 + (east*over_r)**2 + aux*r*over_rz -
                                      aux*(east*over_rz)**2)
    # J_en
    jac[:npoints, nforces:nforces*2] = east*north*over_r*(over_r**2 -
                                                          aux*over_rz**2)
    # J_ev
    jac[:npoints, nforces*2:] = east*over_r*(depth*over_r**2 - aux*over_rz)
    # J_ne
    jac[npoints:npoints*2, :nforces] = jac[:npoints, nforces:nforces*2]
    # J_nn
    jac[npoints:npoints*2, nforces:nforces*2] = over_r*(1 + (north*over_r)**2 +
                                                        aux*r*over_rz -
                                                        aux*(north*over_rz)**2)
    # J_nv
    jac[npoints:npoints*2, nforces*2:] = north*over_r*(depth*over_r**2 -
                                                       aux*over_rz)
    # J_ve
    jac[npoints*2:, :nforces] = east*over_r*(depth*over_r**2 + aux*over_rz)
    # J_vn
    jac[npoints*2:, nforces:nforces*2:] = north*over_r*(depth*over_r**2 +
                                                       aux*over_rz)
    # J_vv
    jac[npoints*2:, nforces*2:] = over_r*(1 + aux + (depth*over_r)**2)
    return jac
