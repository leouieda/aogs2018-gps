# Joint Interpolation of 3-component GPS Velocities Constrained by Elasticity

[Leonardo Uieda](http://www.leouieda.com),
[David Sandwell](http://topex.ucsd.edu/sandwell/),
and
[Paul Wessel](http://www.soest.hawaii.edu/wessel/)

Abstract accepted for presentation at the Asia Oceania Geosciences Society
(AOGS) 2018 15th Annual Meeting at Honolulu, HI, USA.


## Abstract

Vertical ground motion at fault systems can be difficult to detect due to their
small amplitude and contamination from non-tectonic sources, such as ground
water loading. However, it may play an important role in our understanding of
the earthquake cycle and the associated seismic hazards. Ground motion
measurements from GPS are often sparse and must be interpolated onto a regular
grid (e.g., for computing strain rate), ideally taking into account the varying
degrees of uncertainty of the data. Traditionally, each vector component is
interpolated separately using minimum curvature or biharmonic spline methods.
Recently, a joint interpolation of the two horizontal components has been
developed using the Green's functions for a point force deforming a thin
elastic sheet. The elasticity constraints provide a coupling between the two
vector components and lead to improved results because the underlying physics
of the method approximately matches that of the GPS observations. We propose an
expansion of this method into 3D in order to incorporate vertical GPS velocity
measurements. To smooth the model and avoid singularities, we formulate the
interpolation as a weighted least-squares inverse problem with damping
regularization. Optimal values of the regularization parameter and the
Poisson's ratio of the elastic medium are determined through K-fold
cross-validation, a technique often used in machine learning for model
selection. Additionally, the cross-validation provides a measure of the
accuracy of model predictions and eliminates the need for manual configuration.
The computational load of the inversion is lessened by imposing a cutoff
distance to the Green's functions computations, which makes the sensitivity
matrix sparse. We will present preliminary results from an application to
EarthScope GPS data from the San Andreas Fault system. In the future, we aim to
develop a joint inversion of 3D GPS and InSAR line-of-sight velocities to
improve data coverage.

## License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img
alt="Creative Commons License" style="border-width:0"
src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br>
This content is licensed under a <a rel="license"
href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution
4.0 International License</a>.

All source code is distributed under the [BSD 3-clause
License](https://opensource.org/licenses/BSD-3-Clause).

