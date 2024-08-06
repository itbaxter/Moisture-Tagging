import xarray as xr
from scipy.stats import linregress as _linregress

# from utils import *
__all__ = ["jja","linregress","detrend","area_weighted_ave"]

def linregress(da_y, da_x, dim=None):
    '''xarray-wrapped function of scipy.stats.linregress.
    Note the order of the input arguments x, y is reversed to the original scipy function.'''
    if dim is None:
        dim = [d for d in da_y.dims if d in da_x.dims][0]

    slope, intercept, r, p, stderr = xr.apply_ufunc(_linregress, da_x, da_y,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[], [], [], [], []],
        vectorize=True,
        dask='allowed')
    predicted = da_x * slope + intercept

    slope.attrs['long_name'] = 'slope of the linear regression'
    intercept.attrs['long_name'] = 'intercept of the linear regression'
    r.attrs['long_name'] = 'correlation coefficient'
    p.attrs['long_name'] = 'p-value'
    stderr.attrs['long_name'] = 'standard error of the estimated gradient'
    predicted.attrs['long_name'] = 'predicted values by the linear regression model'

    return xr.Dataset(dict(slope=slope, intercept=intercept,
        r=r, p=p, stderr=stderr, predicted=predicted))

def detrend(ds,axis=0):
    #slope,recon = xlinregress(ds)
    dim = list(ds.dims)[axis]
    print(dim)
    if ds[dim][0].dtype == '<M8[ns]':
        time = xr.DataArray(np.arange(1,len(ds[dim])+1,step=1),dims=ds[dim].dims,coords=ds[dim].coords)
        recon = linregress(ds,time).predicted
    else:
        recon = linregress(ds,ds[dim]).predicted
    return ds-recon

def jja(ds):
    ds_jja = ds.resample(time='QS-DEC').mean('time')
    return ds_jja.where(ds_jja['time.season'] == 'JJA',drop=True).groupby('time.year').mean('time')
