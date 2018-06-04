"""
Utility functions for this particular presentation
"""
import itertools
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# We need these two classes to set proper ticklabels for Cartopy maps
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import verde as vd


def plot_data(data, every=1, maxabs=3, pad=None):
    """
    Plot the 3 data components in 2 maps.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5),
                             subplot_kw=dict(projection=ccrs.Mercator()))
    crs = ccrs.PlateCarree()
    # Plot the horizontal components
    ax = axes[0]
    if data.east_velocity.ndim == 1:
        east = data.east_velocity.values[::every]
        north = data.north_velocity.values[::every]
    else:
        east = data.east_velocity.values[::every, ::every]
        north = data.north_velocity.values[::every, ::every]
    tmp = ax.quiver(data.longitude.values[::every],
                    data.latitude.values[::every],
                    east, north, scale=300, width=0.0015,
                    transform=crs)
    ax.set_title('Horizontal velocity')
    # Plot the vertical component
    ax = axes[1]
    if data.up_velocity.ndim == 1:
        pc = ax.scatter(data.longitude, data.latitude,
                        c=data.up_velocity, s=10, cmap='seismic',
                        vmin=-maxabs, vmax=maxabs, transform=crs)
    else:
        pc = ax.pcolormesh(data.longitude, data.latitude, data.up_velocity,
                           cmap='seismic', vmin=-maxabs, vmax=maxabs,
                           transform=crs)
        ax.coastlines()
    plt.colorbar(pc, ax=ax, pad=0, aspect=50).set_label('mm/yr')
    ax.set_title('Vertical velocity')
    ax.quiverkey(tmp, 0.60, 0.10, 30, label='30 mm/yr', coordinates='figure')
    # Setup the axis labels and ticks
    region = vd.get_region((data.longitude, data.latitude))
    if pad is not None:
        region = vd.pad_region(region, pad)
    for ax in axes:
        # Setup the map ticks
        ax.set_xticks(np.arange(-123, -113, 2), crs=crs)
        ax.set_yticks(np.arange(32, 42, 2), crs=crs)
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.add_feature(cfeature.LAND, facecolor='gray')
        ax.add_feature(cfeature.OCEAN)
        ax.set_extent(region, crs=crs)
    plt.tight_layout(w_pad=0)
    return fig, axes


def combinations(**kwargs):
    """
    Iterate over dictionaries of all combinations of the given arguments.
    """
    names = list(kwargs.keys())
    options = [kwargs[name] for name in names]
    combs = [dict(zip(names, comb)) for comb in itertools.product(*options)]
    return combs


def sample_from_grid(grid, coordinates=None, size=None, random_state=None):
    """
    Extact uniformly random samples from an grid.
    """
    if coordinates is None:
        if isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
        elif random_state is None:
            random_state = np.random.RandomState()
        coords = {name: xr.DataArray(random_state.randint(0, grid[name].size,
                                                          size=size),
                                     dims='p')
                  for name in grid.coords}
        sampled_grid = grid.isel(**coords)
    else:
        coords = {name: xr.DataArray(coordinates[name], dims='p')
                  for name in coordinates}
        sampled_grid = grid.sel(**coords, method='nearest')
    columns = {name: sampled_grid[name].values for name in grid.data_vars}
    for name in coords:
        columns[name] = sampled_grid[name].values
    sample = pd.DataFrame(columns)
    return sample


def longitude_shift(lon, allpositive=True, copy=True):
    """
    Shift longitude between [-180, 180] and [0, 360].
    """
    if copy:
        shifted = lon.copy()
    else:
        shifted = lon
    if allpositive:
        negative = shifted < 0
        shifted[negative] = 360 + shifted[negative]
    else:
        over = shifted > 180
        shifted[over] = shifted[over] - 360
    return shifted
