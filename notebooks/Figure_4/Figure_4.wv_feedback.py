# %%
import xarray as xr
import numpy as np
import glob as glob
import cartopy
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
import matplotlib as mpl
import colormaps
import glob as glob
import matplotlib.path as mpath
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.ticker import ScalarFormatter,AutoLocator,MultipleLocator,AutoMinorLocator,FixedLocator
from matplotlib.patches import Rectangle

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

from metpy.calc import mixing_ratio_from_specific_humidity,saturation_mixing_ratio,mixing_ratio_from_specific_humidity
from metpy.units import units
from metpy.constants import Rv,Lv

# Create a custom RdYlBu colormap with white in the middle
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 7
mpl.rcParams['hatch.color'] = 'silver'

from imods.stats import *

# %%
## Lapse rate

# %%
t = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/ERA5_kernels/54region/remapped/isotope-nudging2.cam.h0.T.197901-202212.latlon.ERA5.nc')
t

# %%
dTA2 = preprocess(t['T'])
dTA2

# %%
dta = xr.open_dataset('./t_trend.month.1981-2022.nc')
dta

# %%
def month_loop(ds,month=1):
    print(month)
    ds_sel = mon(ds,month=month).sel(year=slice(1981,2022))
    r = linregress(ds_sel,ds_sel.year) #.slope
    return r

def preprocess(ds):
    ds.coords['time'] = np.arange('1979-01-01','2023-01-01',dtype='datetime64[M]')
    dqs = [month_loop(ds,month=month) for month in range(1,13)]
    dqs = xr.concat(dqs,dim='month')
    dqs.coords['month'] = np.arange(1,13,step=1)
    return dqs

TS = preprocess(ts['TS'])
TS

# %%
_,TS_lev = xr.broadcast(dta.slope,TS.slope)
TS_lev

# %%
TS_col = area_weighted_ave(TS_lev.sel(latitude=slice(90,70)).rename({'latitude':'lat','longitude':'lon'})).sel(month=slice(6,8)).mean('month')
TS_col

# %%
dT_col = area_weighted_ave(dta.slope.sel(latitude=slice(90,70)).rename({'latitude':'lat','longitude':'lon'})).sel(month=slice(6,8)).mean('month')
dT_col

# %%
dT_col.plot(y='plev',yincrease=False)
TS_col.plot(y='plev',yincrease=False)

# %%
_,TAS = xr.broadcast(T0['T'],tas)
TAS

# %%
dTl = dqo.slope/q0['Q']*Rv.magnitude/Lv.magnitude*TAS**2
#dT.coords['plev'] = dT['plev']/100
dTl

# %%
dlRq_LW=(Kq_LW['SFC_all']*(dT-TS_lev).rename({'plev':'level'})*dp/100) #.sum('level')
dlRq_SW=(Kq_SW['SFC_all']*(dT-TS_lev).rename({'plev':'level'})*dp/100) #.sum('level')
dlRq_SW

# %%
fig = plt.figure(figsize=(7.5,7.5))
cmap = colormaps.BlueWhiteOrangeRed

ax = fig.add_subplot(311)
ax.set_title('WV feedback using ERA5 kernels')
for axis in ['bottom','left', 'right', 'top']:
    ax.spines[axis].set_linewidth(1.25)
ax.invert_yaxis()
ax.text(-0.05, 1.2, 'a',
        weight='bold',
        fontsize=13,
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)

wvf = (10*(dlRq_LW+dlRq_SW)).sel(month=slice(6,8)).mean(('month','longitude'))
p = wvf.plot(ax=ax,cmap=cmap,
             #vmin=-0.2,vmax=0.2,
             yincrease=False)
p.axes.set_title('WV feedback using TS')

cmap = colormaps.BlueWhiteOrangeRed
ax = fig.add_subplot(312)
ax.set_title('WV feedback using ERA5 kernels')
for axis in ['bottom','left', 'right', 'top']:
    ax.spines[axis].set_linewidth(1.25)
ax.invert_yaxis()
ax.text(-0.05, 1.2, 'a',
        weight='bold',
        fontsize=13,
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)

p = (10*(dRq_LW+dRq_SW)).sel(month=slice(6,8)).mean(('month','longitude')).plot(ax=ax,cmap=cmap,
                                                                             #vmin=-0.2,vmax=0.2,
                                                                             yincrease=False)
p.axes.set_title('WV feedback using ERA5 kernels')

ax = fig.add_subplot(313)
ax.set_title('WV feedback using TA')
for axis in ['bottom','left', 'right', 'top']:
    ax.spines[axis].set_linewidth(1.25)
ax.invert_yaxis()
ax.text(-0.05, 1.2, 'a',
        weight='bold',
        fontsize=13,
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)

wvf = (10*((dlRq_LW+dlRq_SW)-(dRq_LW+dRq_SW))).sel(month=slice(6,8)).mean(('month','longitude'))
p = wvf.plot(ax=ax,cmap=cmap,
             #vmin=-0.2,vmax=0.2,
             yincrease=False)
p.axes.set_title('Difference')

plt.tight_layout()

# %%
## Create new fixed relative humidity feedback
k_ts = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/ERA5_kernels/ERA5_kernel_ts_TOA.nc')

dir = '/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/ERA5_kernels/thickness_normalized_ta_wv_kernel/'

k_t = xr.open_dataset(dir+'ERA5_kernel_ta_dp_TOA.nc')
k_t

k_q_sw = xr.open_dataset(dir+'ERA5_kernel_wv_sw_dp_TOA.nc')
k_q_sw

k_q_lw = xr.open_dataset(dir+'ERA5_kernel_wv_lw_dp_TOA.nc')
k_q_lw

k_q = k_q_sw+k_q_lw
k_q

# %%
k_t_col = area_weighted_ave(k_t.rename({'latitude':'lat','longitude':'lon'}))
k_q_col = area_weighted_ave(k_q.rename({'latitude':'lat','longitude':'lon'}))

k_t_col

# %%
dp = xr.open_dataset(dir+'dp_era5.nc')
dp

# %%
k_tf = k_t['TOA_all']+k_q_lw['TOA_all']+k_q_sw['TOA_all']
k_tf

# %%
#  create a new dictionary to hold all the feedbacks
flux = {}
#   Compute the feedbacks!
flux['lw_q'] = (k_q_lw['TOA_all'] * dq.rename({'plev':'level'})).integrate('level')
#  shortwavewater vapor
flux['sw_q'] = (k_q_sw['TOA_all'] * dq.rename({'plev':'level'})).integrate('level')

"""
#  longwave temperature  (Planck and lapse rate)
flux['Planck'] =  (kernels['lw_ts'] +
                   integral(kernels['lw_t'])) * DeltaTS_interp
flux['lapse'] = integral(kernels['lw_t'] * (DeltaT_interp-DeltaTS_interp))
flux['lw_t'] = flux['Planck'] + flux['lapse']
#  Add up the longwave feedbacks
flux['lw_net'] = flux['lw_t'] + flux['lw_q']
"""

# %%
flux['lw_q'].mean(dim=['month','longitude']).plot()

# %%
# Alternate description using fixed relative humidity as state
# variable following Held and Shell J. Clim. 2012


# temperature kernel for fixed relative humidity is the sum of
#  traditional temperature and traditional water vapor kernels
k_talt = k_t['SFC_all'] + k_q_lw['SFC_all']


Planck_alt =  (k_ts['SFC_all']+k_t['SFC_all']) * TS_lev.rename({'plev':'level'}) * 10
lapse_alt = k_t['SFC_all'] * (dta.slope - TS_lev).rename({'plev':'level'}) * 10
    
H = k_q['SFC_all']*dT.rename({'plev':'level'}) - (k_q_lw['SFC_all'] * TS_lev.rename({'plev':'level'}))

# Get RH feedback by subtracting original water vapor kernel times
# atmospheric temperature response from traditional water vapor feedback.
#flux['RH'] = flux['lw_q'] - integral(kernels['lw_q'] * DeltaT_interp)


# %%
Planck_alt.mean(dim=['month','longitude']).plot.contourf(x='latitude',y='level',
                                                                               #levels=np.arange(-0.45,0.451,step=0.05),
                                                                               cmap=cmap,
                                                                               yincrease=False)

# %%
lapse_alt.sel(month=slice(6,8)).mean(dim=['month','longitude']).plot.contourf(x='latitude',y='level',
                                                        #levels=np.arange(-0.3,0.31,step=0.02),
                                                           cmap=cmap,
                                                           yincrease=False)

# %%
k_q['SFC_all'].mean(dim=['month','longitude']).plot.contourf(x='latitude',y='level',
                                                        #levels=np.arange(-0.045,0.0451,step=0.005),
                                                           cmap=cmap,
                                                           yincrease=False)

# %%
H.mean(dim=['month','longitude']).plot.contourf(x='latitude',y='level',
                                                        levels=np.arange(-0.045,0.0451,step=0.005),
                                                           cmap=cmap,
                                                           yincrease=False)

# %%
iPa = area_weighted_ave(Planck_alt.rename({'latitude':'lat','longitude':'lon'})).sum('level').mean(dim=['month'])
iPa.values

# %%
iLa = area_weighted_ave(lapse_alt.rename({'latitude':'lat','longitude':'lon'})).sum('level').mean(dim=['month'])
iLa.values

# %%
iH = area_weighted_ave(H.rename({'latitude':'lat','longitude':'lon'})).sum('level').mean(dim=['month'])
iH.values

# %%
"""
    #  finite difference approximation to the slope d/dT (q_saturation)
    small = 0.01
    dqsatdT = (qsat(ctrl.data_vars['TA']+small, ctrl.lev) -
               qsat(ctrl.data_vars['TA']-small, ctrl.lev)) / (2*small)

    #  actual specific humidity anomalies
    DeltaQ = pert.Q - ctrl.Q
    #  relative humidity in control run (convert from percent to fraction)
    RH_ctrl = ctrl.RELHUM / 100.

    #  Equivalent temperature change
    #  (actual humidity change expressed as temperature change at fixed RH)
    DeltaTequiv = DeltaQ / (RH_ctrl * dqsatdT )
    #  Scaled by local surface temp. anomaly
    #DeltaTequiv_scaled = DeltaTequiv / DeltaTS
     #  But actually we are supposed to be using log(q)
    DeltaLogQ = np.log(pert.Q) - np.log(ctrl.Q)
    dlogqsatdT = (np.log(qsat(ctrl.data_vars['TA']+small, ctrl.lev)) -
               np.log(qsat(ctrl.data_vars['TA']-small, ctrl.lev))) / (2*small)
    DeltaTequiv_log = DeltaLogQ / (dlogqsatdT)
    #  Interpolated to kernel grid:
    field = regrid(ctrl.lat.data, ctrl.lev.data, DeltaTequiv_log.data,
                   kernels.lat.data, lev_kernel.data)
    DeltaTequiv_interp = np.ma.masked_array(field, np.isnan(field))

    #  create a new dictionary to hold all the feedbacks
    flux = {}
    #   Compute the feedbacks!
    flux['lw_q'] = integral(kernels['lw_q'] * DeltaTequiv_interp)
    #  shortwave water vapor
    flux['sw_q'] = integral(kernels['sw_q'] * DeltaTequiv_interp)
    #  longwave temperature  (Planck and lapse rate)
    flux['Planck'] =  (kernels['lw_ts'] +
                       integral(kernels['lw_t'])) * DeltaTS_interp
    flux['lapse'] = integral(kernels['lw_t'] * (DeltaT_interp-DeltaTS_interp))
    flux['lw_t'] = flux['Planck'] + flux['lapse']
    #  Add up the longwave feedbacks
    flux['lw_net'] = flux['lw_t'] + flux['lw_q']
"""

# %% [markdown]
# ## CAM5 Relative Humidity Framework

# %%
def month_loop(ds,month=1):
    print(month)
    ds_sel = mon(ds,month=month).sel(year=slice(1981,2022))
    r = linregress(ds_sel,ds_sel.year) #.slope
    return r

def preprocess(ds):
    ds.coords['time'] = np.arange('1979-01-01','2023-01-01',dtype='datetime64[M]')
    dqs = [month_loop(ds,month=month) for month in range(1,13)]
    dqs = xr.concat(dqs,dim='month')
    dqs.coords['month'] = np.arange(1,13,step=1)
    return dqs

# %%
## Create new fixed relative humidity feedback
k_ts = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/ERA5_kernels/ERA5_kernel_ts_TOA.nc')

dir = '/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/other_kernels/kernel_cam5_res2.5_level37_toa_sfc.nc'

kernels = xr.open_dataset(dir)
kernels = kernels.rename({'time':'month'})
kernels.coords['month'] = np.arange(1,13,step=1)
kernels

# %%
dq = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/ERA5_kernels/54region/wv_feedback/q_trend.month.1981-2022.new.nc').slope
dq = dq.rename({'plev':'level'})
dq

# %%
files = sorted(glob.glob('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/ERA5_kernels/54region/wv_feedback/q_trend.TA*V.month.1981-2022.nc'))
print(files[0],files[-1])

dqs = xr.open_mfdataset(files,combine='nested',concat_dim='region')
dqs.coords['region'] = np.arange(1,55,step=1)
#dqs.reindex(plev=list(reversed(dqs.plev)))
dqs

# %%
rh = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/ERA5_kernels/54region/remapped/isotope-nudging2.cam.h0.RELHUM.197901-202212.latlon.ERA5.nc')['RELHUM']
rh = rh.rename({'plev':'level'})
rh.coords['time'] = np.arange('1979-01-01','2023-01-01',dtype='datetime64[M]')
rh

# %%
q = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/ERA5_kernels/54region/remapped/isotope-nudging2.cam.h0.Q.197901-202212.latlon.ERA5.nc')['Q']
q = q.rename({'plev':'level'})
q.coords['time'] = np.arange('1979-01-01','2023-01-01',dtype='datetime64[M]')
q

# %% [markdown]
# q = xr.open_mfdataset('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/ERA5_kernels/54region/remapped/isotope-nudging2.cam.h0.TA*.197901-202212.latlon.ERA5.nc',combine='nested',concat_dim='region')
# q = q.rename({'plev':'level'})
# q.coords['time'] = np.arange('1979-01-01','2023-01-01',dtype='datetime64[M]')
# q

# %%
ta = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/ERA5_kernels/54region/remapped/isotope-nudging2.cam.h0.T.197901-202212.latlon.ERA5.nc')['T']
ta = ta.rename({'plev':'level'})
ta.coords['time'] = np.arange('1979-01-01','2023-01-01',dtype='datetime64[M]')
ta

# %%
dta = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/ERA5_kernels/54region/wv_feedback/t_trend.month.1981-2022.nc')
dta

# %%
trefht = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/ERA5_kernels/54region/temp/isotope-nudging2.cam.h0.TREFHT.197901-202212.latlon.ERA5.nc')['TREFHT']
trefht.coords['time'] = np.arange('1979-01-01','2023-01-01',dtype='datetime64[M]')
trefht

# %%
ts = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/ERA5_kernels/54region/temp/isotope-nudging2.cam.h0.TS.197901-202212.latlon.ERA5.nc')['TS']
ts.coords['time'] = np.arange('1979-01-01','2023-01-01',dtype='datetime64[M]')
ts

# %%
qrefht = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/ERA5_kernels/54region/temp/isotope-nudging2.cam.h0.QREFHT.197901-202212.latlon.ERA5.nc')['QREFHT']
qrefht.coords['time'] = np.arange('1979-01-01','2023-01-01',dtype='datetime64[M]')
qrefht

# %%
QREFHT = preprocess(qrefht)
QREFHT

# %%
TREFHT = preprocess(trefht)
TREFHT

# %%
DeltaTREFHT = trefht.sel(time=slice('2000-01-01','2022-12-01')).groupby('time.month').mean('time')-trefht.sel(time=slice('1979-01-01','1999-12-01')).groupby('time.month').mean('time')

# %%
TS = preprocess(ts)
TS

# %%
dTA = dta.slope.rename({'plev':'level'})

# %%
_,dTREFHT = xr.broadcast(dTA,TREFHT.slope)
dTREFHT

# %%
_,dQREFHT = xr.broadcast(dq,QREFHT.slope)
dQREFHT

# %%
dTA_new = dTA.fillna(dTREFHT)
dTA_new

# %%
dq_new = dq.fillna(dQREFHT)
dq_new

# %%
_,dTS = xr.broadcast(dTA,TS.slope)
dTS

# %%
RH_ctrl = rh.sel(time=slice('1979-01-01', '1999-01-01')).groupby('time.month').mean('time') / 100.
Q_ctrl = q.sel(time=slice('1979-01-01', '1999-01-01')).groupby('time.month').mean('time')
T_ctrl = ta.sel(time=slice('1979-01-01', '1999-01-01')).groupby('time.month').mean('time')


# %%
dp = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/ERA5_kernels/thickness_normalized_ta_wv_kernel/dp_era5.nc')['dp']
dp.coords['month'] = np.arange(1,13,step=1)

#   The kernels are not on the same grid as the CAM4 output
lev_kernel = kernels.level
dp_kernel = dp.mean(dim=['month','longitude'])

# Flux is calculated by convolving the kernel with
#  temperature / humidity response in the TROPOSPHERE

#   define a mask from surface to 100 mb at equator,
#    decreasing linearly to 300 mb at the poles
#      (following Feldl and Roe 2013)
maxlev = xr.DataArray(np.tile(100. + (200. * np.abs(kernels.latitude)/90.), (lev_kernel.size, 1)),
                      dims=('level','latitude'),
                      coords={'level':dp['level'],
                              'latitude':dp['latitude']})

#p_kernel = np.tile(np.expand_dims(lev_kernel, axis=1), (1, kernels.latitude.size))

#dp_masked = np.ma.masked_array(dp_kernel, np.where(p_kernel > maxlev, False, True))

#  this custom function will compute vertical integrals
#  taking advantage of xarray named coordinates
def integral(field):
    return (field.where(field['level'] > maxlev)).sum(dim='level')


# %%
def clausius_clapeyron(T):
    """Compute saturation vapor pressure as function of temperature T.

    Input: T is temperature in Kelvin
    Output: saturation vapor pressure in mb or hPa

    Formula from Rogers and Yau "A Short Course in Cloud Physics" (Pergammon Press), p. 16
    claimed to be accurate to within 0.1% between -30degC and 35 degC
    Based on the paper by Bolton (1980, Monthly Weather Review).

    """
    Tcel = T - 273.15
    es = 6.112 * np.exp(17.67*Tcel/(Tcel+243.5))
    return es

def qsat(T,p):
    """Compute saturation specific humidity as function of temperature and pressure.

    Input:  T is temperature in Kelvin
            p is pressure in hPa or mb
    Output: saturation specific humidity (dimensionless).

    """
    es = clausius_clapeyron(T)
    eps = Rv.magnitude/Lv.magnitude
    q = eps * es / (p - (1 - eps) * es )
    return q

# %%
#  actual specific humidity anomalies
DeltaQ = dq_new
#  relative humidity in control run (convert from percent to fraction)

small = 0.01
dqsatdT = (qsat(T_ctrl+small, T_ctrl.level) -
           qsat(T_ctrl-small, T_ctrl.level)) / (2*small)

#dqsatdT = (qsat(T_ctrl + 1.0, T_ctrl.level) - 
#           qsat(T_ctrl, T_ctrl.level))

#  Equivalent temperature change
#  (actual humidity change expressed as temperature change at fixed RH)
DeltaTequiv_new = DeltaQ / (RH_ctrl * 100 * dqsatdT)
DeltaTequiv =  (DeltaQ/Q_ctrl*Rv.magnitude/Lv.magnitude*T_ctrl**2)
#  Scaled by local surface temp. anomaly
#DeltaTequiv_scaled = DeltaTequiv / DeltaTS
 #  But actually we are supposed to be using log(q)

dlogqsatdT = (np.log(qsat(T_ctrl+small, T_ctrl.level)) -
           np.log(qsat(T_ctrl-small, T_ctrl.level))) / (2*small)
    
DeltaLogQ = (np.log(dq*42)-np.log(Q_ctrl)) #np.log(dq)
#dlogqsatdT = np.log(dqsatdT)

DeltaTequiv_log = DeltaLogQ / (dlogqsatdT)
#  Interpolated to kernel grid:

# %%
#  create a new dictionary to hold all the feedbacks
flux = {}
#   Compute the feedbacks!
flux['lw_q'] = (kernels['wv_lw_toa_all'] * DeltaTequiv_log)
#  shortwave water vapor
flux['sw_q'] = (kernels['wv_sw_toa_all'] * DeltaTequiv_log)

#  longwave temperature  (Planck and lapse rate)
flux['Planck'] =  (kernels['ts_toa_all'] +
                   (kernels['ta_toa_all'])) * dTREFHT

flux['lapse'] = kernels['ta_toa_all'] * (dTA_new - dTREFHT)
flux['lw_t'] = flux['Planck'] + flux['lapse']

#  Add up the longwave feedbacks
flux['lw_net'] = flux['lw_t'] + flux['lw_q']

flux['q_net'] = flux['lw_q'] + flux['sw_q']

# %%
# temperature kernel for fixed relative humidity is the sum of
#  traditional temperature and traditional water vapor kernels
kernels['lw_talt'] = kernels['ta_toa_all'] + kernels['wv_lw_toa_all']
flux['Planck_alt'] =  (kernels['ts_toa_all'] +
                       (kernels['lw_talt'])) * dTREFHT
flux['lapse_alt'] = (kernels['lw_talt'] *
                             (dTA_new-dTREFHT))
# Get RH feedback by subtracting original water vapor kernel times
# atmospheric temperature response from traditional water vapor feedback.
flux['RH'] = flux['lw_q'] - (kernels['wv_lw_toa_all'] * dTA_new)

#  package output into xarray datasets
flux_dataset = xr.Dataset(data_vars=flux)/ dTREFHT.mean('longitude').sel(level=1000)

# %%
keys = list(flux.keys())
for key in keys:
    if key == 'Planck_alt':
        print(key, area_weighted_ave(flux_dataset[key].rename({'latitude':'lat','longitude':'lon'}).sel(level=1000)).mean('month').values)
    else:
        print(key, area_weighted_ave(flux_dataset[key].rename({'latitude':'lat','longitude':'lon'}).sel(level=slice(1000,300)).sum('level')).mean('month').values)

# %%
keys = list(flux.keys())
for key in keys:
    print(key, area_weighted_ave(flux_dataset[key].rename({'latitude':'lat','longitude':'lon'}).sel(lat=slice(90,70),level=slice(1000,300)).sum('level')).mean('month').values)

# %%
keys = list(flux.keys())
for key in keys:
    print(key, area_weighted_ave(flux_dataset[key].rename({'latitude':'lat','longitude':'lon'}).sel(lat=slice(90,70),level=slice(1000,300),month=slice(6,8)).sum('level')).mean('month').values)

# %%
dTREFHT_col = area_weighted_ave(dTREFHT.sel(latitude=slice(90,70)).rename({'latitude':'lat','longitude':'lon'})).sel(month=slice(6,8)).mean('month')
dTREFHT_col

# %%
dTS_col = area_weighted_ave(dTREFHT.sel(latitude=slice(90,70)).rename({'latitude':'lat','longitude':'lon'})).sel(month=slice(6,8)).mean('month')
dTS_col

# %%
dTA_col = area_weighted_ave(dTA_new.sel(latitude=slice(90,70)).rename({'latitude':'lat','longitude':'lon'})).sel(month=slice(6,8)).mean('month')
dTA_col

# %%
dTe_col = area_weighted_ave(DeltaTequiv.sel(latitude=slice(90,70)).rename({'latitude':'lat','longitude':'lon'})).sel(month=slice(6,8)).mean('month')
dTe_col

# %%
files = sorted(glob.glob('/raid/scratch/scratch-itbaxter/exp/cam5-kernels/cam5-kernels/demodata/*nc'))

base = xr.open_dataset(files[0])
change = xr.open_dataset(files[1])
base

# %%
ibase = area_weighted_ave(base['Q'].sel(lat=slice(70,90)))
ichange = area_weighted_ave(change['Q'].sel(lat=slice(70,90)))

# %% [markdown]
# ## ERA5 redo

# %%
surf = 'TOA'

## Create new fixed relative humidity feedback
dir = '/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/ERA5_kernels/thickness_normalized_ta_wv_kernel/'

dp = xr.open_dataset(dir+'dp_era5.nc')['dp']
dp.coords['month'] = np.arange(1,13,step=1)
k_t = xr.open_dataset(dir+f'ERA5_kernel_ta_dp_{surf}.nc')
k_t.coords['month'] = np.arange(1,13,step=1)
k_q_sw = xr.open_dataset(dir+f'ERA5_kernel_wv_sw_dp_{surf}.nc')
k_q_sw.coords['month'] = np.arange(1,13,step=1)
k_q_lw = xr.open_dataset(dir+f'ERA5_kernel_wv_lw_dp_{surf}.nc')
k_q_lw.coords['month'] = np.arange(1,13,step=1)
k_ts = xr.open_dataset(f'/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/ERA5_kernels/ERA5_kernel_ts_{surf}.nc')
k_ts.coords['month'] = np.arange(1,13,step=1)

## Create new fixed relative humidity feedback
dir = '/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/ERA5_kernels/layer_specified_ta_wv_kernel/'

k_t2 = xr.open_dataset(dir+f'ERA5_kernel_ta_nodp_{surf}.nc')
k_t2.coords['month'] = np.arange(1,13,step=1)
k_q_sw2 = xr.open_dataset(dir+f'ERA5_kernel_wv_sw_nodp_{surf}.nc')
k_q_sw2.coords['month'] = np.arange(1,13,step=1)
k_q_lw2 = xr.open_dataset(dir+f'ERA5_kernel_wv_lw_nodp_{surf}.nc')
k_q_lw2.coords['month'] = np.arange(1,13,step=1)
k_ts

# %%
#  actual specific humidity anomalies
DeltaQ = dq 
#  relative humidity in control run (convert from percent to fraction)

small = 0.01
dqsatdT = (qsat(T_ctrl+small, T_ctrl.level) -
           qsat(T_ctrl-small, T_ctrl.level)) / (2*small)


#  Equivalent temperature change
#  (actual humidity change expressed as temperature change at fixed RH)
DeltaTequiv = DeltaQ / (RH_ctrl * dqsatdT )
#DeltaTequiv =  (DeltaQ/Q_ctrl*Rv.magnitude/Lv.magnitude*T_ctrl**2)
#  Scaled by local surface temp. anomaly
#DeltaTequiv_scaled = DeltaTequiv / DeltaTS
 #  But actually we are supposed to be using log(q)

dlogqsatdT = (np.log(qsat(T_ctrl+small, T_ctrl.level)) -
           np.log(qsat(T_ctrl-small, T_ctrl.level))) / (2*small)
    
DeltaLogQ = np.log(dq)
#dlogqsatdT = np.log(dqsatdT)

DeltaTequiv_log = DeltaLogQ / (dlogqsatdT)
#  Interpolated to kernel grid:

# %%
#  create a new dictionary to hold all the feedbacks
flux = {}
#   Compute the feedbacks!
flux['lw_q'] = (k_q_lw[f'{surf}_all'] * DeltaTequiv_log * dp / 100)
#  shortwave water vapor
flux['sw_q'] = (k_q_sw[f'{surf}_all'] * DeltaTequiv_log * dp / 100)

#  longwave temperature  (Planck and lapse rate)
flux['Planck'] =  (k_ts[f'{surf}_all'] +
                   (k_t[f'{surf}_all'])) * dTREFHT * dp /100

flux['lapse'] = k_t[f'{surf}_all'] * (dTA - dTREFHT) * dp / 100
flux['lw_t'] = flux['Planck'] + flux['lapse']

#  Add up the longwave feedbacks
flux['lw_net'] = flux['lw_t'] + flux['lw_q']

flux['q_net'] = flux['lw_q'] + flux['sw_q']

# %%
# temperature kernel for fixed relative humidity is the sum of
#  traditional temperature and traditional water vapor kernels
lw_talt = k_t[f'{surf}_all'] + k_q_lw[f'{surf}_all']
flux['Planck_alt'] =  (k_ts[f'{surf}_all'] +
                       (lw_talt)) * dTREFHT * dp / 100
flux['lapse_alt'] = (lw_talt *
                             (dTA-dTREFHT)) * dp / 100
# Get RH feedback by subtracting original water vapor kernel times
# atmospheric temperature response from traditional water vapor feedback.
flux['RH'] = flux['lw_q'] - (k_q_lw[f'{surf}_all'] * dTA) * dp / 100

#  package output into xarray datasets
era5_flux = xr.Dataset(data_vars=flux)

# %%
era5_flux_dts = era5_flux / dTREFHT.mean('longitude')
era5_flux_dts

# %%
iflux = area_weighted_ave(integral(era5_flux_dts).rename({'latitude':'lat','longitude':'lon'})).mean('month')
iflux

# %% [markdown]
# ## Regional 

# %%
def get_camflux(dq,dqr=None,region=True):
    #  this custom function will compute vertical integrals
    #  taking advantage of xarray named coordinates
    
    flux = {}
    t1=T_ctrl
    dta=dTA_new
    q0 = Q_ctrl

    qs1 = qsat(t1,T_ctrl.level)
    qs2 = qsat(t1+dta,T_ctrl.level)
    dqsdt = (qs2 - qs1)/dta
    rh = 1000*q0/qs1
    dqdt = rh*dqsdt

    dlogqdt=dqdt/(1000*q0)


    # Normalize kernels by the change in moisture for 1 K warming at
    # constant RH (log-q kernel)
    logq_LW_kernel=(kernels[f'wv_lw_{surf}_all']/dlogqdt)
    logq_SW_kernel=(kernels[f'wv_sw_{surf}_all']/dlogqdt)

    dlogq=dq_new/q0

    # Convolve moisture kernel with change in moisture
    dLW_logq=(logq_LW_kernel*dlogq)
    dSW_logq=(logq_SW_kernel*dlogq)
    flux['dLW_logq']=integral(logq_LW_kernel*dlogq)
    flux['dSW_logq']=integral(logq_SW_kernel*dlogq)
    flux['dLWSW_logq'] = flux['dLW_logq'] + flux['dSW_logq']

    slw_talt = kernels[f'ta_{surf}_all'] + kernels[f'wv_lw_{surf}_all']
    flux['dLW_planck_alt'] = (kernels[f'ts_{surf}_all'] +
                           integral(slw_talt)) * dTREFHT.sel(level=1000).squeeze()
    flux['dLW_lapse_alt'] = integral(slw_talt * (dTA_new - dTREFHT))
    flux['dLW_qt'] = integral(kernels[f'wv_lw_{surf}_all'] * dTA_new)
    dLW_qt = (kernels[f'wv_lw_{surf}_all'] * dTA_new)
    flux['dLW_rh'] = integral(dLW_logq) - integral(dLW_qt)
    
    flux['dLW_lapse'] = integral(kernels[f'ta_{surf}_all'] * (dTA_new - dTREFHT))
    flux['dLW_planck'] = (kernels[f'ts_{surf}_all']+integral(kernels[f'ta_{surf}_all'])) * dTREFHT.sel(level=1000).squeeze()
    flux['dt_feedback'] = flux['dLW_lapse'] + flux['dLW_planck']
    
    flux['dLW_ts'] = kernels[f'ts_{surf}_all'] * dTREFHT.sel(level=1000).squeeze()
    flux['dLW_ta'] = integral(kernels[f'ta_{surf}_all'] * dTA_new)
    flux['t_feedback'] = flux['dLW_ts'] + flux['dLW_ta']
    
    return xr.Dataset(flux)

# %%
surf = 'toa'
cam_flux = get_camflux(dq_new,region=False)
cam_flux 

# %%
surf = 'sfc'
scam_flux = get_camflux(dq_new,region=False)
scam_flux 

# %%
surf = 'toa'
tcam_flux = get_camflux(dq_new,region=False)
tcam_flux 

# %%
def get_flux(dq,dqr=None,region=True):
    #  this custom function will compute vertical integrals
    #  taking advantage of xarray named coordinates
    
    DeltaT = dTA_new
    DeltaTS = dTREFHT
    #  actual specific humidity anomalies
    if region == True:
        DeltaQ = dq - dqr
    else:
        DeltaQ = dq
    #  relative humidity in control run (convert from percent to fraction)

    small = 0.01
    dqsatdT = (qsat(T_ctrl+small, T_ctrl.level) -
               qsat(T_ctrl-small, T_ctrl.level)) / (2*small)


    #  Equivalent temperature change
    #  (actual humidity change expressed as temperature change at fixed RH)
    #DeltaTequiv = DeltaQ / (RH_ctrl * dqsatdT )
    DeltaTequiv = DeltaQ / (RH_ctrl * dqsatdT)
    DeltaTequiv_old =  (dq*Rv.magnitude/Lv.magnitude*T_ctrl**2)
    #DeltaTequiv_old = qsat(DeltaQ, DeltaQ.level)
    #  Scaled by local surface temp. anomaly
    #DeltaTequiv_scaled = DeltaTequiv / DeltaTS
     #  But actually we are supposed to be using log(q)

    dlogqsatdT = (np.log(qsat(T_ctrl+small, T_ctrl.level)) -
               np.log(qsat(T_ctrl-small, T_ctrl.level))) / (2*small)
    

    DeltaLogQ = np.log(dq)
    #dlogqsatdT = np.log(dqsatdT)

    DeltaTequiv_log = DeltaLogQ / (dlogqsatdT) 
    #  Interpolated to kernel grid:
    
    #  create a new dictionary to hold all the feedbacks
    flux = {}
    #   Compute the feedbacks!
    flux['lw_q'] = integral(k_q_lw2[f'{surf}_all'] * DeltaTequiv_log) # * dp / 100)
    #  shortwave water vapor
    flux['sw_q'] = integral(k_q_sw2[f'{surf}_all'] * DeltaTequiv_log) # * dp / 100)

    #  longwave temperature  (Planck and lapse rate)
    flux['Planck'] =  (k_ts[f'{surf}_all'] +
                       integral(k_t2[f'{surf}_all'])) * DeltaTS.sel(level=1000) #* dp / 100

    flux['lapse'] = integral(k_t2[f'{surf}_all'] * (DeltaT - DeltaTS)) # * dp / 100
    flux['lw_t'] = flux['Planck'] + flux['lapse']

    #  Add up the longwave feedbacks
    flux['lw_net'] = flux['lw_t'] + flux['lw_q']

    flux['lw_q_trad'] = integral(k_q_lw2[f'{surf}_all'] * DeltaTequiv_old) # * dp / 100)
    flux['sw_q_trad'] = integral(k_q_sw2[f'{surf}_all'] * DeltaTequiv_old) # * dp / 100)
    
    flux['q_net'] = flux['lw_q'] + flux['sw_q']
    flux['q_net_trad'] = flux['lw_q_trad'] + flux['sw_q_trad']
    
    # temperature kernel for fixed relative humidity is the sum of
    #  traditional temperature and traditional water vapor kernels
    lw_talt = k_t2[f'{surf}_all'] + k_q_lw2[f'{surf}_all']
    flux['Planck_alt'] =  (k_ts[f'{surf}_all'] +
                           integral(lw_talt)) * DeltaTS.sel(level=1000).squeeze() #* dp / 100
    flux['lapse_alt'] = integral(lw_talt * 
                                 (DeltaT-DeltaTS)) #* dp / 100
    # Get RH feedback by subtracting original water vapor kernel times
    # atmospheric temperature response from traditional water vapor feedback.
    flux['lw_qt'] = integral(k_q_lw2[f'{surf}_all'] * DeltaT)
    flux['RH'] = flux['lw_q'] - integral(k_q_lw2[f'{surf}_all'] * DeltaT) # * dp / 100)
    flux['RH_trad'] = flux['lw_q_trad'] - integral(k_q_lw2[f'{surf}_all'] * DeltaT)
    #  package output into xarray datasets
    era5_flux = xr.Dataset(data_vars=flux)
    return era5_flux #/ (dTREFHT.mean('longitude')) 

# %%
dTS = dTREFHT.mean('longitude').sel(level=1000).squeeze()
dTSg = area_weighted_ave(dTREFHT.sel(level=1000).rename({'latitude':'lat','longitude':'lon'}))

# %%
def create_flux():
    flux = {}
    t1=T_ctrl
    dta=dTA_new
    q0 = Q_ctrl

    qs1 = qsat(t1,T_ctrl.level)
    qs2 = qsat(t1+dta,T_ctrl.level)
    dqsdt = (qs2 - qs1)/dta
    rh = 1000*q0/qs1
    dqdt = rh*dqsdt

    dlogqdt=dqdt/(1000*q0)


    # Normalize kernels by the change in moisture for 1 K warming at
    # constant RH (log-q kernel)
    logq_LW_kernel=(k_q_lw2[f'{surf}_clr']/dlogqdt)
    logq_SW_kernel=(k_q_sw2[f'{surf}_clr']/dlogqdt)

    dlogq=dq_new/q0

    # Convolve moisture kernel with change in moisture
    dLW_logq=(logq_LW_kernel*dlogq)
    dSW_logq=(logq_SW_kernel*dlogq)
    flux['dLW_logq']=integral(logq_LW_kernel*dlogq)
    flux['dSW_logq']=integral(logq_SW_kernel*dlogq)
    flux['dLWSW_logq'] = flux['dLW_logq'] + flux['dSW_logq']
    
    slw_talt = k_t2[f'{surf}_all'] + k_q_lw2[f'{surf}_all']
    flux['dLW_planck_alt'] = (k_ts[f'{surf}_all'] +
                           integral(slw_talt)) * dTREFHT.sel(level=1000).squeeze()
    flux['dLW_lapse_alt'] = integral(slw_talt * (dTA_new - dTREFHT))
    flux['dLW_qt'] = integral(k_q_lw2[f'{surf}_all'] * dTA_new)
    dLW_qt = (k_q_lw2[f'{surf}_all'] * dTA_new)
    flux['dLW_rh'] = integral(dLW_logq) - integral(dLW_qt)
    
    flux['dLW_lapse'] = integral(k_t2[f'{surf}_all'] * (dTA_new - dTREFHT))
    flux['dLW_planck'] = (k_ts[f'{surf}_all']+integral(k_t2[f'{surf}_all'])) * dTREFHT.sel(level=1000).squeeze()
    flux['dt_feedback'] = flux['dLW_lapse'] + flux['dLW_planck']
    
    flux['dLW_ts'] = k_ts[f'{surf}_all'] * dTREFHT.sel(level=1000).squeeze()
    flux['dLW_ta'] = integral(k_t[f'{surf}_all'] * dTA_new)
    flux['t_feedback'] = flux['dLW_ts'] + flux['dLW_ta']
    
    return xr.Dataset(flux)

# %%
surf = 'TOA'

## Create new fixed relative humidity feedback
dir = '/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/ERA5_kernels/thickness_normalized_ta_wv_kernel/'

dp = xr.open_dataset(dir+'dp_era5.nc')['dp']
dp.coords['month'] = np.arange(1,13,step=1)
k_t = xr.open_dataset(dir+f'ERA5_kernel_ta_dp_{surf}.nc')
k_t.coords['month'] = np.arange(1,13,step=1)
k_q_sw = xr.open_dataset(dir+f'ERA5_kernel_wv_sw_dp_{surf}.nc')
k_q_sw.coords['month'] = np.arange(1,13,step=1)
k_q_lw = xr.open_dataset(dir+f'ERA5_kernel_wv_lw_dp_{surf}.nc')
k_q_lw.coords['month'] = np.arange(1,13,step=1)
k_ts = xr.open_dataset(f'/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/ERA5_kernels/ERA5_kernel_ts_{surf}.nc')
k_ts.coords['month'] = np.arange(1,13,step=1)

## Create new fixed relative humidity feedback
dir = '/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/ERA5_kernels/layer_specified_ta_wv_kernel/'

k_t2 = xr.open_dataset(dir+f'ERA5_kernel_ta_nodp_{surf}.nc')
k_t2.coords['month'] = np.arange(1,13,step=1)
k_q_sw2 = xr.open_dataset(dir+f'ERA5_kernel_wv_sw_nodp_{surf}.nc')
k_q_sw2.coords['month'] = np.arange(1,13,step=1)
k_q_lw2 = xr.open_dataset(dir+f'ERA5_kernel_wv_lw_nodp_{surf}.nc')
k_q_lw2.coords['month'] = np.arange(1,13,step=1)
k_ts

# %%
toa_flux = create_flux()
toa_flux

# %%
## SURFACE

# %%
surf = 'SFC'

## Create new fixed relative humidity feedback
dir = '/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/ERA5_kernels/thickness_normalized_ta_wv_kernel/'

dp = xr.open_dataset(dir+'dp_era5.nc')['dp']
dp.coords['month'] = np.arange(1,13,step=1)
k_t = xr.open_dataset(dir+f'ERA5_kernel_ta_dp_{surf}.nc')
k_t.coords['month'] = np.arange(1,13,step=1)
k_q_sw = xr.open_dataset(dir+f'ERA5_kernel_wv_sw_dp_{surf}.nc')
k_q_sw.coords['month'] = np.arange(1,13,step=1)
k_q_lw = xr.open_dataset(dir+f'ERA5_kernel_wv_lw_dp_{surf}.nc')
k_q_lw.coords['month'] = np.arange(1,13,step=1)
k_ts = xr.open_dataset(f'/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/ERA5_kernels/ERA5_kernel_ts_{surf}.nc')
k_ts.coords['month'] = np.arange(1,13,step=1)

## Create new fixed relative humidity feedback
dir = '/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/ERA5_kernels/layer_specified_ta_wv_kernel/'

k_t2 = xr.open_dataset(dir+f'ERA5_kernel_ta_nodp_{surf}.nc')
k_t2.coords['month'] = np.arange(1,13,step=1)
k_q_sw2 = xr.open_dataset(dir+f'ERA5_kernel_wv_sw_nodp_{surf}.nc')
k_q_sw2.coords['month'] = np.arange(1,13,step=1)
k_q_lw2 = xr.open_dataset(dir+f'ERA5_kernel_wv_lw_nodp_{surf}.nc')
k_q_lw2.coords['month'] = np.arange(1,13,step=1)
k_ts

# %%
surf_flux = create_flux()
surf_flux

# %%
fig = plt.figure(figsize=(7.5,3))

ax = fig.add_subplot(121)

ax.text(-0.03, 1.05, 'a', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

ax.set_title('Global Mean Local Feedback')
#ax.set_xlim([-90,90])
#ax.set_ylim([-6.5,6.5])
ax.set_ylim([-3.5,3.5])
#ax.yaxis.set_label_position("right")
#ax.yaxis.tick_right()
ax.set_ylabel(r'Feedback Parameter [$\mathrm{W\ m^{-2}\ K^{-1}}$]')
#ax.set_xlabel('latitude')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.25)

    
xlib = {'dLWSW_logq':r'$\mathrm{{\lambda}_{wv}}$',
        'dLW_rh':r'$\mathrm{\tilde{\lambda}_{rh}}$',
        'dLW_planck':r'$\mathrm{{\lambda}_{lr}}$',
        'dLW_lapse':r'$\mathrm{{\lambda}_{p}}$',
        'dLW_planck_alt':r'$\mathrm{\tilde{\lambda}_{p}}$',
        'dLW_lapse_alt':r'$\mathrm{\tilde{\lambda}_{lr}}$',
       }

xpos = {'dLWSW_logq':0,
        'dLW_rh':1,
        'dLW_planck':2,
        'dLW_lapse':3,
        'dLW_planck_alt':4,
        'dLW_lapse_alt':5,
        'dLW_logq':0,
        'dSW_logq':0,
       }

for i in range(5):
    ax.axvline(i+0.5,linestyle='--',linewidth=0.7,c='k')

ax.yaxis.grid(linestyle='--',linewidth=0.7,c='silver')

ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xticklabels(['WV', 'RH', 'Planck', 'LR', 'Planck alt', 'LR alt'])

plt.axhline(0,linestyle='--',linewidth=0.7,c='k')

# Loop through the variables and plot the data
for key in xlib:
    ax.scatter(xpos[key] - 0.2, area_weighted_ave((surf_flux[key] / dTS).sel(latitude=slice(90,-90),month=slice(6,8)).rename({'latitude':'lat','longitude':'lon'})).mean('month'), 
               c='k',s=50,
               edgecolor='k',
               zorder=100,
               label='JJA SFC')
    ax.scatter(xpos[key] + 0.2, area_weighted_ave((toa_flux[key] / dTS).sel(latitude=slice(90,-90),month=slice(6,8)).rename({'latitude':'lat','longitude':'lon'})).mean('month'), 
               c='tab:blue',s=50,
               edgecolor='tab:blue',
               zorder=100,
               label='JJA TOA')
    ax.scatter(xpos[key] - 0.2, area_weighted_ave((surf_flux[key] / dTS).sel(latitude=slice(90,-90),month=slice(1,12)).rename({'latitude':'lat','longitude':'lon'})).mean('month'), 
               c='None',s=50,
               edgecolor='k',
               zorder=100,
               label='Annual SFC')
    ax.scatter(xpos[key] + 0.2, area_weighted_ave((toa_flux[key] / dTS).sel(latitude=slice(90,-90),month=slice(1,12)).rename({'latitude':'lat','longitude':'lon'})).mean('month'),
               c='None',s=50,
               edgecolor='tab:blue',
               zorder=100,
               label='Annual TOA')

key = 'dLW_logq'
sc1 = ax.scatter(xpos[key]-0.2,area_weighted_ave((surf_flux[key]/dTS).rename({'latitude':'lat','longitude':'lon'}).sel(lat=slice(90,-90),month=slice(6,8))).mean('month').values,
                 facecolors='tab:red', 
                 edgecolors='tab:red',
                 linewidth=1.5,
                 zorder=101,
                 c='tab:red',
                 label=r'$\mathrm{\lambda_{WV\ LW}}$ JJA')

key = 'dLW_logq'
sc2 = ax.scatter(xpos[key]-0.2,area_weighted_ave((surf_flux[key]/dTS).rename({'latitude':'lat','longitude':'lon'}).sel(lat=slice(90,-90),month=slice(1,12))).mean('month').values,
                 facecolors='None', 
                 edgecolors='tab:red',
                 linewidth=1.5,
                 zorder=101,
                 c='None',
                 label=r'$\mathrm{\lambda_{WV\ LW}}$ Annual')

key = 'dLW_logq'
ax.scatter(xpos[key]+0.2,area_weighted_ave((toa_flux[key]/dTS).rename({'latitude':'lat','longitude':'lon'}).sel(lat=slice(90,-90),month=slice(6,8))).mean('month').values,
           facecolors='tab:red', edgecolors='tab:red',
           linewidth=1.5,zorder=101)

key = 'dLW_logq'
scatter = ax.scatter(xpos[key]+0.2,area_weighted_ave((toa_flux[key]/dTS).rename({'latitude':'lat','longitude':'lon'}).sel(lat=slice(90,-90),month=slice(1,12))).mean('month').values,
                     c='None',facecolors='None', 
                     edgecolors='tab:red',
                     linewidth=1.5,zorder=101)
    
handles, labels = ax.get_legend_handles_labels()
unique_labels = list(set(labels))
handles0 = handles[0:4]
unique_labels0 = [unique_labels[1],unique_labels[2],unique_labels[0],unique_labels[3]]
legend = ax.legend(handles0, unique_labels0, 
                   fontsize=7, loc='lower left', 
                   bbox_to_anchor=[0.2,0.7],
                   frameon=True, edgecolor='white', 
                   framealpha=1.0)

ax.add_artist(legend)  

handles1 = [handles[-2],handles[-1]]
print(unique_labels)
unique_labels1 = [unique_labels[-2],unique_labels[-1]]
legend2 = ax.legend(handles1, unique_labels1, 
                    bbox_to_anchor=[0.55,0.8],
                    fontsize=7, loc='lower left', 
                    frameon=True, edgecolor='white', 
                    framealpha=1.0)
ax.add_artist(legend2)

ax.text(0.025, 0.025, 'Surface', weight='bold',
        rotation='vertical',
        fontsize=7,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

ax.text(0.11, 0.025, 'Top of atmosphere', weight='bold',
        rotation='vertical',
        fontsize=7,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)
        

#-------------------------------------------------------------
ax = fig.add_subplot(122)

ax.text(-0.03, 1.05, 'b', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

ax.set_title('Arctic Local Feedback')
#ax.set_xlim([-90,90])
ax.set_ylim([-3.5,3.5])
#ax.yaxis.set_label_position("right")
#ax.yaxis.tick_right()
#ax.set_ylabel(r'Feedback Parameter [$\mathrm{W\ m^{-2}\ K^{-1}}$]')
#ax.set_xlabel('latitude')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.25)
    
for i in range(5):
    ax.axvline(i+0.5,linestyle='--',linewidth=0.7,c='k')

ax.yaxis.grid(linestyle='--',linewidth=0.7,c='silver')
    
xlib = {'dLWSW_logq':r'$\mathrm{{\lambda}_{wv}}$',
        'dLW_rh':r'$\mathrm{\tilde{\lambda}_{rh}}$',
        'dLW_planck':r'$\mathrm{{\lambda}_{lr}}$',
        'dLW_lapse':r'$\mathrm{{\lambda}_{p}}$',
        'dLW_planck_alt':r'$\mathrm{\tilde{\lambda}_{p}}$',
        'dLW_lapse_alt':r'$\mathrm{\tilde{\lambda}_{lr}}$',
       }

xpos = {'dLWSW_logq':0,
        'dLW_rh':1,
        'dLW_planck':2,
        'dLW_lapse':3,
        'dLW_planck_alt':4,
        'dLW_lapse_alt':5,
        'dLW_logq':0,
        'dSW_logq':0,
       }

ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xticklabels(['WV', 'RH', 'Planck', 'LR', 'Planck alt', 'LR alt'])

# Loop through the variables and plot the data
for key in xlib:
    ax.scatter(xpos[key] - 0.2, area_weighted_ave((surf_flux[key] / dTS).sel(latitude=slice(90,70),month=slice(6,8)).rename({'latitude':'lat','longitude':'lon'})).mean('month'), 
               c='k',s=50,
               edgecolor='k',
               zorder=100,
               label='JJA SFC')
    ax.scatter(xpos[key] + 0.2, area_weighted_ave((toa_flux[key] / dTS).sel(latitude=slice(90,70),month=slice(6,8)).rename({'latitude':'lat','longitude':'lon'})).mean('month'), 
               c='tab:blue',s=50,
               edgecolor='tab:blue',
               zorder=100,
               label='JJA TOA')
    ax.scatter(xpos[key] - 0.2, area_weighted_ave((surf_flux[key] / dTS).sel(latitude=slice(90,70),month=slice(1,12)).rename({'latitude':'lat','longitude':'lon'})).mean('month'), 
               c='None',s=50,
               edgecolor='k',
               zorder=100,
               label='Annual SFC')
    ax.scatter(xpos[key] + 0.2, area_weighted_ave((toa_flux[key] / dTS).sel(latitude=slice(90,70),month=slice(1,12)).rename({'latitude':'lat','longitude':'lon'})).mean('month'),
               c='None',s=50,
               edgecolor='tab:blue',
               zorder=100,
               label='Annual TOA')

key = 'dLW_logq'
sc1 = ax.scatter(xpos[key]-0.2,area_weighted_ave((surf_flux[key]/dTS).rename({'latitude':'lat','longitude':'lon'}).sel(lat=slice(90,70),month=slice(6,8))).mean('month').values,
                 facecolors='tab:red', 
                 edgecolors='tab:red',
                 linewidth=1.5,
                 zorder=101,
                 c='tab:red',
                 label=r'$\mathrm{\lambda_{WV\ LW}}$ JJA')

key = 'dLW_logq'
sc2 = ax.scatter(xpos[key]-0.2,area_weighted_ave((surf_flux[key]/dTS).rename({'latitude':'lat','longitude':'lon'}).sel(lat=slice(90,70),month=slice(1,12))).mean('month').values,
                 facecolors='None', 
                 edgecolors='tab:red',
                 linewidth=1.5,
                 zorder=101,
                 c='None',
                 label=r'$\mathrm{\lambda_{WV\ LW}}$ Annual')

key = 'dLW_logq'
ax.scatter(xpos[key]+0.2,area_weighted_ave((toa_flux[key]/dTS).rename({'latitude':'lat','longitude':'lon'}).sel(lat=slice(90,70),month=slice(6,8))).mean('month').values,
           facecolors='tab:red', edgecolors='tab:red',
           linewidth=1.5,zorder=101)

key = 'dLW_logq'
scatter = ax.scatter(xpos[key]+0.2,area_weighted_ave((toa_flux[key]/dTS).rename({'latitude':'lat','longitude':'lon'}).sel(lat=slice(90,70),month=slice(1,12))).mean('month').values,
                     c='None',facecolors='None', 
                     edgecolors='tab:red',
                     linewidth=1.5,zorder=10)
"""   
handles, labels = ax.get_legend_handles_labels()
unique_labels = list(set(labels))
handles0 = handles[0:4]
unique_labels0 = unique_labels[0:4]
legend = ax.legend(handles0, unique_labels0, 
                   fontsize=7, loc='lower left', 
                   bbox_to_anchor=[0.6,0.7],
                   frameon=True, edgecolor='white', 
                   framealpha=1.0)
ax.add_artist(legend)  

handles1 = [handles[-2],handles[-1]]
print(unique_labels)
unique_labels1 = [unique_labels[-2],unique_labels[-1]]
legend2 = ax.legend(handles1, unique_labels1, 
                    bbox_to_anchor=[0.2,0.8],
                    fontsize=7, loc='lower left', 
                    frameon=True, edgecolor='white', 
                    framealpha=1.0)
ax.add_artist(legend2)
"""
ax.text(0.025, 0.025, 'Surface', weight='bold',
        rotation='vertical',
        fontsize=7,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

ax.text(0.11, 0.025, 'Top of atmosphere', weight='bold',
        rotation='vertical',
        fontsize=7,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

plt.axhline(0,linestyle='--',linewidth=0.7,c='k')
plt.savefig('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/moisture_tagging_paper/Figures/plots/moisture_tagging-supfigure_10-era5_feedbacks_local.png',dpi=600)

# %%
fbs = area_weighted_ave((toa_flux/dTS).rename({'latitude':'lat','longitude':'lon'}).sel(lat=slice(90,70),month=slice(1,12))).mean('month')
fbs

# %%
def get_flux_region(dq,dqr=None,region=True):
    #  this custom function will compute vertical integrals
    #  taking advantage of xarray named coordinates
    def integral(field):
        return (field * dp_masked).sum(dim='pfull')
    
    #  actual specific humidity anomalies
    if region == True:
        DeltaQ = dq_new - dqr
    else:
        DeltaQ = dq_new
    #  relative humidity in control run (convert from percent to fraction)

    small = 0.01
    dqsatdT = (qsat(T_ctrl+small, T_ctrl.level) -
               qsat(T_ctrl-small, T_ctrl.level)) / (2*small)


    #  Equivalent temperature change
    #  (actual humidity change expressed as temperature change at fixed RH)
    #DeltaTequiv = DeltaQ / (RH_ctrl * dqsatdT )
    DeltaTequiv =  (DeltaQ/Q_ctrl*Rv.magnitude/Lv.magnitude*T_ctrl**2)
    #  Scaled by local surface temp. anomaly
    #DeltaTequiv_scaled = DeltaTequiv / DeltaTS
     #  But actually we are supposed to be using log(q)

    dlogqsatdT = (np.log(qsat(T_ctrl+small, T_ctrl.level)) -
               np.log(qsat(T_ctrl-small, T_ctrl.level))) / (2*small)

    DeltaLogQ = np.log(DeltaQ)
    dlogqsatdT = np.log(dqsatdT)

    DeltaTequiv_log = DeltaLogQ / (dlogqsatdT) 
    #  Interpolated to kernel grid:
    
    #  create a new dictionary to hold all the feedbacks
    flux = {}
    #   Compute the feedbacks!
    flux['lw_q'] = (k_q_lw2[f'{surf}_all'] * DeltaTequiv_log) # * dp / 100)
    #  shortwave water vapor
    flux['sw_q'] = (k_q_sw2[f'{surf}_all'] * DeltaTequiv_log) # * dp / 100)

    #  longwave temperature  (Planck and lapse rate)
    #flux['Planck'] =  (k_ts[f'{surf}_all'] +
    #                   (k_t2[f'{surf}_all'])) * dTREFHT #* dp / 100

    #flux['lapse'] = k_t2[f'{surf}_all'] * (dTA_new - dTREFHT) # * dp / 100
    #flux['lw_t'] = flux['Planck'] + flux['lapse']

    #  Add up the longwave feedbacks
    #flux['lw_net'] = flux['lw_t'] + flux['lw_q']

    flux['lw_q_trad'] = (k_q_lw2[f'{surf}_all'] * DeltaTequiv) # * dp / 100)
    flux['sw_q_trad'] = (k_q_sw2[f'{surf}_all'] * DeltaTequiv) # * dp / 100)
    
    flux['q_net'] = flux['lw_q'] + flux['sw_q']
    flux['q_net_trad'] = flux['lw_q_trad'] + flux['sw_q_trad']
    
    # temperature kernel for fixed relative humidity is the sum of
    #  traditional temperature and traditional water vapor kernels
    #lw_talt = k_t2[f'{surf}_all'] + k_q_lw2[f'{surf}_all']
    #flux['Planck_alt'] =  (k_ts[f'{surf}_all'] +
    #                       (lw_talt)) * dTREFHT #* dp / 100
    #flux['lapse_alt'] = (lw_talt *
    #                             (dTA_new-dTREFHT)) #* dp / 100
    # Get RH feedback by subtracting original water vapor kernel times
    # atmospheric temperature response from traditional water vapor feedback.
    flux['RH'] = flux['lw_q'] - (k_q_lw2[f'{surf}_all'] * dTA) # * dp / 100)

    #  package output into xarray datasets
    era5_flux = xr.Dataset(data_vars=flux)
    return era5_flux #/ dTREFHT.mean('longitude')

# %%
ntot = get_flux_region(dq_new,region=False)
ntot

# %%
test = [get_flux_region(dq,dqs.slope.sel(region=i).rename({'plev':'level'}),region=True) for i in dqs.region]
test = xr.concat(test,dim='region')
test

# %% [markdown]
# ## Main Plot Start Here

# %%
surf = 'TOA'

## Create new fixed relative humidity feedback
dir = '/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/ERA5_kernels/thickness_normalized_ta_wv_kernel/'

dp = xr.open_dataset(dir+'dp_era5.nc')['dp']
dp.coords['month'] = np.arange(1,13,step=1)
k_t = xr.open_dataset(dir+f'ERA5_kernel_ta_dp_{surf}.nc')
k_t.coords['month'] = np.arange(1,13,step=1)
k_q_sw = xr.open_dataset(dir+f'ERA5_kernel_wv_sw_dp_{surf}.nc')
k_q_sw.coords['month'] = np.arange(1,13,step=1)
k_q_lw = xr.open_dataset(dir+f'ERA5_kernel_wv_lw_dp_{surf}.nc')
k_q_lw.coords['month'] = np.arange(1,13,step=1)
k_ts = xr.open_dataset(f'/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/ERA5_kernels/ERA5_kernel_ts_{surf}.nc')
k_ts.coords['month'] = np.arange(1,13,step=1)

## Create new fixed relative humidity feedback
dir = '/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/ERA5_kernels/layer_specified_ta_wv_kernel/'

k_t2 = xr.open_dataset(dir+f'ERA5_kernel_ta_nodp_{surf}.nc')
k_t2.coords['month'] = np.arange(1,13,step=1)
k_q_sw2 = xr.open_dataset(dir+f'ERA5_kernel_wv_sw_nodp_{surf}.nc')
k_q_sw2.coords['month'] = np.arange(1,13,step=1)
k_q_lw2 = xr.open_dataset(dir+f'ERA5_kernel_wv_lw_nodp_{surf}.nc')
k_q_lw2.coords['month'] = np.arange(1,13,step=1)
k_ts

# %%
def r_create_flux(dq_in):
    flux = {}
    t1=T_ctrl
    dta=dTA_new
    q0 = Q_ctrl

    qs1 = qsat(t1,T_ctrl.level)
    qs2 = qsat(t1+dta,T_ctrl.level)
    dqsdt = (qs2 - qs1)/dta
    rh = 1000*q0/qs1
    dqdt = rh*dqsdt

    dlogqdt=dqdt/(1000*q0)


    # Normalize kernels by the change in moisture for 1 K warming at
    # constant RH (log-q kernel)
    logq_LW_kernel=(k_q_lw2[f'{surf}_clr']/dlogqdt)
    logq_SW_kernel=(k_q_sw2[f'{surf}_clr']/dlogqdt)

    dlogq=dq_in/q0

    # Convolve moisture kernel with change in moisture
    dLW_logq=(logq_LW_kernel*dlogq)
    dSW_logq=(logq_SW_kernel*dlogq)
    flux['dLW_logq']=integral(logq_LW_kernel*dlogq)
    flux['dSW_logq']=integral(logq_SW_kernel*dlogq)
    flux['dLWSW_logq'] = integral(dLW_logq + dSW_logq)
    
    lw_talt = k_t2[f'{surf}_all'] + k_q_lw2[f'{surf}_all']
    flux['dLW_planck_alt'] = (k_ts[f'{surf}_all'] +
                           integral(lw_talt)) * dTREFHT.sel(level=1000).squeeze()
    flux['dLW_lapse_alt'] = integral(lw_talt * (dTA_new - dTREFHT))
    flux['dLW_qt'] = integral(k_q_lw2[f'{surf}_all'] * dTA_new)
    dLW_qt = (k_q_lw2[f'{surf}_all'] * dTA_new)
    flux['dLW_rh'] = integral(dLW_logq) - integral(dLW_qt)
    
    flux['dLW_lapse'] = integral(k_t2[f'{surf}_all'] * (dTA_new - dTREFHT))
    flux['dLW_planck'] = (k_ts[f'{surf}_all']+integral(k_t2[f'{surf}_all'])) * dTREFHT.sel(level=1000).squeeze()
    flux['dt_feedback'] = flux['dLW_lapse'] + flux['dLW_planck']
    
    flux['dLW_ts'] = k_ts[f'{surf}_all'] * dTREFHT.sel(level=1000).squeeze()
    flux['dLW_ta'] = integral(k_t[f'{surf}_all'] * dTA_new)
    flux['t_feedback'] = flux['dLW_ts'] + flux['dLW_ta']
    
    return xr.Dataset(flux)

# %%
def wv_feedback(dq_in):
    t1=T_ctrl
    dta=dTA_new
    q0 = Q_ctrl

    qs1 = qsat(t1,T_ctrl.level)
    qs2 = qsat(t1+dta,T_ctrl.level)
    dqsdt = (qs2 - qs1)/dta
    rh = 1000*q0/qs1
    dqdt = rh*dqsdt

    dlogqdt=dqdt/(1000*q0)


    # Normalize kernels by the change in moisture for 1 K warming at
    # constant RH (log-q kernel)
    logq_LW_kernel=(k_q_lw2[f'{surf}_clr']/dlogqdt)
    logq_SW_kernel=(k_q_sw2[f'{surf}_clr']/dlogqdt)

    dlogq=dq_in/q0

    # Convolve moisture kernel with change in moisture
    dLW_logq=(logq_LW_kernel*dlogq)
    dSW_logq=(logq_SW_kernel*dlogq)
    flux['dLW_logq']=integral(logq_LW_kernel*dlogq)
    flux['dSW_logq']=integral(logq_SW_kernel*dlogq)
    flux['dLWSW_logq'] = flux['dLW_logq'] + flux['dSW_logq']
    
    return dLW_logq + dSW_logq

# %%
## Create new fixed relative humidity feedback
k_ts = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/ERA5_kernels/ERA5_kernel_ts_TOA.nc')

dir = '/raid/scratch/scratch-itbaxter/exp/ERA5_radiative_kernels/Data_2024/other_kernels/kernel_cam5_res2.5_level37_toa_sfc.nc'

kernels = xr.open_dataset(dir)
kernels = kernels.rename({'time':'month'})
kernels.coords['month'] = np.arange(1,13,step=1)
kernels

# %%
def cam_wv_feedback(dq_in,surf='toa'):
    t1=T_ctrl
    dta=dTA_new
    q0 = Q_ctrl

    qs1 = qsat(t1,T_ctrl.level)
    qs2 = qsat(t1+dta,T_ctrl.level)
    dqsdt = (qs2 - qs1)/dta
    rh = 1000*q0/qs1
    dqdt = rh*dqsdt

    dlogqdt=dqdt/(1000*q0)


    # Normalize kernels by the change in moisture for 1 K warming at
    # constant RH (log-q kernel)
    logq_LW_kernel=(kernels[f'wv_lw_{surf}_all']/dlogqdt)
    logq_SW_kernel=(kernels[f'wv_sw_{surf}_all']/dlogqdt)

    dlogq=dq_in/q0

    # Convolve moisture kernel with change in moisture
    dLW_logq=(logq_LW_kernel*dlogq)
    dSW_logq=(logq_SW_kernel*dlogq)
    flux['dLW_logq']=integral(logq_LW_kernel*dlogq)
    flux['dSW_logq']=integral(logq_SW_kernel*dlogq)
    flux['dLWSW_logq'] = flux['dLW_logq'] + flux['dSW_logq']
    
    return dLW_logq + dSW_logq

# %%
areacella = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/CESM2-LE/CESM2.areacella.nc')['areacella'].squeeze().load()


rlist = np.arange(1,54,step=1)

latbounds = np.arange(-90,90.1,step=20)
lonbounds = np.arange(0,360.1,step=60)


arc_weighted = (areacella).sel(lat=slice(70,90)).sum(dim=['lat','lon'])

def weight_trends(arcealla,i):
    ds_w = np.zeros(6)
    for y in range(6):
        ds_w[y] = arcealla.sel(lat=slice(latbounds[i],latbounds[i+1]),lon=slice(lonbounds[y],lonbounds[y+1])).sum(dim=['lat','lon'])
    return ds_w
    
reg_weights = [weight_trends(areacella,i) for i in range(9)]
reg_weights = np.concatenate(reg_weights)

xreg_weights = xr.DataArray(np.std(reg_weights)/reg_weights,
                            dims=dqs.region.dims,
                            coords=dqs.region.coords)

#(reg_weights[i]/arc_weighted.values)
xreg_weights

# %%
dqs_norm = dqs * xreg_weights
dqs_norm

# %%
idqs_norm = area_weighted_ave(integral((dqs_norm['slope']).sel(latitude=slice(90,70)).rename({'plev':'level'})).rename({'longitude':'lon','latitude':'lat'}))
idqs_norm

# %%
new_fb = [wv_feedback(dq_new-dqs.slope.sel(region=i).rename({'plev':'level'})) for i in dqs.region]
new_fb = xr.concat(new_fb,dim='region')
new_fb

# %%
inew_fb = area_weighted_ave((new_fb/dTS).sel(latitude=slice(90,70)).rename({'latitude':'lat','longitude':'lon'}))
inew_fb

# %%
tot_fb = r_create_flux(dq_new)
itot_fb = area_weighted_ave((tot_fb/dTS).sel(latitude=slice(90,70)).rename({'latitude':'lat','longitude':'lon'}))
itot_fb

# %%
new_test = [wv_feedback(dq_new-dqs.slope.sel(region=i).rename({'plev':'level'})) for i in dqs.region]
new_test = xr.concat(new_test,dim='region')
new_test

# %%
surf = 'toa'
cam_test = [cam_wv_feedback(dq_new-dqs.slope.sel(region=i).rename({'plev':'level'})) for i in dqs.region]
cam_test = xr.concat(cam_test,dim='region')
cam_test

# %%
norm_test = new_test * xreg_weights

# %%
inorm_test = area_weighted_ave((integral(norm_test)/dTS).sel(latitude=slice(90,70)).rename({'latitude':'lat','longitude':'lon'}))
inorm_test

# %%
surf = 'TOA'
new_tot = wv_feedback(dq_new)
new_tot

# %%
surf = 'toa'

cam_tot = cam_wv_feedback(dq_new)
cam_tot

# %%
# Number of boxes
num_boxes  = 54
xnum_boxes = 6
ynum_boxes = 9

box = ((280,310),(60,75))

# Define latitude and longitude bounds
lat_min, lat_max = -90.0, 90.0
lon_min, lon_max = -180.0, 180.0

# Generate equally spaced latitude and longitude values
latitudes = np.linspace(lat_min, lat_max, ynum_boxes)
longitudes = np.linspace(lon_min, lon_max, xnum_boxes)

# Create a meshgrid of latitudes and longitudes
lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

# Assign unique integer values to each grid cell
grid_values = np.arange(1, num_boxes + 1).reshape(lat_grid.shape)
print(grid_values.shape,lat_grid.shape)

itest = area_weighted_ave((((new_tot-new_test))/dTS).sel(month=slice(6,8),latitude=slice(90,70)).mean('month').rename({'latitude':'lat','longitude':'lon'}))
itest_new = area_weighted_ave(((integral(new_tot-new_test))/dTS).sel(month=slice(6,8),latitude=slice(90,70)).mean('month').rename({'latitude':'lat','longitude':'lon'}))

grid = xr.DataArray(itest_new.values.reshape(lat_grid.shape),
                    name=r'Contribution to Arctic WV feedback',
                    dims=('lat','lon'),
                    coords={'lat': latitudes[:], 'lon': longitudes[:]},
                   )

# %%
itest_norm = area_weighted_ave(((integral(new_tot-new_test))/dTS).sel(month=slice(6,8),latitude=slice(90,70)).mean('month').rename({'latitude':'lat','longitude':'lon'}))

# %%
inew_tot = area_weighted_ave(((integral(new_tot))/dTS).sel(month=slice(6,8),latitude=slice(90,70)).mean('month').rename({'latitude':'lat','longitude':'lon'}))

# %%

# Create a custom RdYlBu colormap with white in the middle
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 7
mpl.rcParams['hatch.color'] = 'silver'

fig = plt.figure(figsize=(7.5,7))
# Panel 1--------------------------------------------------
ax = fig.add_subplot(211,projection=ccrs.Robinson(central_longitude=180))
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')

ax.text(-0.03, 1.05, 'a', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

num_rows = 9
num_cols = 6

# Generate evenly spaced squares in a grid
cnt = 1
for i in range(num_rows):
    for j in range(num_cols):
        lon1 = 0 + j * (360 / num_cols)
        lat1 = -90 + i * (180 / num_rows)
        lon2 = lon1 + (360 / num_cols)
        lat2 = lat1 + (180 / num_rows)

        square = {'name': f'{cnt}', 'coords': (lon1, lat1, lon2, lat2)}

        # Plot the squares
        rectangle = Rectangle((lon1, lat1), lon2 - lon1, lat2 - lat1, edgecolor='None', facecolor='none')
        ax.add_patch(rectangle)
        if abs(grid.values[i,j]) >= 1:
            ax.text(lon1+25, lat2-15, square['name'], color='white', fontsize=8, va='bottom', ha='left', transform=ccrs.PlateCarree())
        else:
            ax.text(lon1+25, lat2-15, square['name'], color='black', fontsize=8, va='bottom', ha='left', transform=ccrs.PlateCarree())
        cnt += 1

# Create a grid of values corresponding to the squares
x = np.linspace(0, 360, num_cols + 1)
y = np.linspace(-90, 90, num_rows + 1)
X, Y = np.meshgrid(x, y)
Z = np.random.rand(num_rows, num_cols)  # Replace this with your actual data

cmap = colormaps.BlueWhiteOrangeRed
levels = np.arange(-0.35,0.351,step=0.025)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

p = ax.pcolormesh(X, Y, grid.values, 
                        cmap=cmap,
                        #shading='auto', 
                        alpha=1.0, 
                        norm=norm,
                        transform=ccrs.PlateCarree(),
                       )
# Colorbar

cax = fig.add_axes([0.85,0.55,0.02,0.42])
cb = plt.colorbar(p,orientation='vertical', 
                  cax=cax,
                  drawedges=True,
                  #ticks=levels[::2]
                 )
cb.set_label(r'JJA Arctic WV feedback [W $\mathrm{m^{-2}\ K^{-1}}$] ')
cb.outline.set_color('k')
cb.outline.set_linewidth(0.8)
cb.dividers.set_color('k')
cb.dividers.set_linewidth(0.8)
cb.ax.tick_params(size=0)
cb.ax.minorticks_off()
cb.ax.yaxis.set_label_position('left')

cb.ax.set_aspect('auto')
pos = cb.ax.get_position()

# create a second axes instance and set the limits you need
cax2 = cb.ax.twinx()
cax2.set_position(pos)

fmt = lambda x, pos: '{:.02f}'.format(x)

cb2 = plt.colorbar(p,cax2,orientation='vertical',
                  drawedges=True,
                  ticks=levels[::2],
                 )
ticklabels = 100*(levels[::2]/np.sum(grid.values))
cb2.ax.set_yticklabels(["{:.02f}".format(i) for i in ticklabels])
cb2.outline.set_color('k')
cb2.outline.set_linewidth(0.8)
cb2.dividers.set_color('k')
cb2.dividers.set_linewidth(0.8)
cb2.ax.tick_params(size=0)
cb2.ax.minorticks_off()
cb2.set_label(r'Relative Contribution to Arctic WV feedback [%]')
cb2.ax.yaxis.set_label_position('right')

ax.spines['geo'].set_linewidth(1.25)

# Add gridlines
gl = ax.gridlines(draw_labels=True, linewidth=1.0,  
                  x_inline=False,
                  y_inline=False,
                  rotate_labels=False, color='k', 
                  linestyle='-', alpha=1.0)
gl.top_labels = False  # Turn off top labels
gl.left_labels = True  # Turn off left labels
gl.right_labels = False  # Turn off left labels
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([-90, -70, -50, -30, -10, 10, 30, 50, 70, 90])

#####################################################
## Panel b-------------------------------------------
#####################################################

ax = fig.add_subplot(212)
ax.set_xlim([0.5,54.5])
ax.set_ylim([100,1000])
ax.set_ylabel('Pressure [hPa]')
ax.set_xlabel('Region')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.25)
ax.invert_yaxis()
ax.set_yscale("log")
ax.yaxis.set_major_locator(FixedLocator([100,200,300,400,600,700,800,900,1000])) #AutoLocator())
ax.yaxis.set_major_formatter('{x:3.0f}')
ax.tick_params(right=True)
ax.xaxis.set_major_locator(MultipleLocator(5)) #AutoLocator())
ax.xaxis.set_minor_locator(MultipleLocator(1))

ax.grid(linestyle='--',alpha=0.7,linewidth=0.35,
        color='silver')

ax.text(-0.03, 1.05, 'b', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

cmap = colormaps.BlueWhiteOrangeRed
bounds = np.arange(-0.05,0.051,step=0.005)
p = ax.contourf(np.arange(1,55,step=1),itest['level'].values,np.array(itest),
                cmap=cmap,
                #extend='both',
                levels=bounds,
                )

ln1 = ax.contour(np.arange(1,55,step=1),itest['level'].values,np.array(itest),
               levels=np.concatenate([bounds[:10],bounds[11:]]),
                linewidths=0.2,
               colors='k',
              )

cax = fig.add_axes([0.85,0.08,0.02,0.42])
cb = plt.colorbar(p,orientation='vertical', 
                  cax=cax,
                  drawedges=True,
                 )
cb.set_label(r'JJA Arctic WV feedback [W $\mathrm{m^{-2}\ K^{-1}}$] ')
cb.outline.set_color('k')
cb.outline.set_linewidth(0.8)
cb.dividers.set_color('k')
cb.dividers.set_linewidth(0.8)
cb.ax.tick_params(size=0)
cb.ax.minorticks_off()


fig.subplots_adjust(top=0.92,bottom=0.11,left=0.08,right=0.78,hspace=0.3,wspace=0.1)
#plt.savefig('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/moisture_tagging_paper/Figures/plots/moisture_tagging-figure_4-era5.blue2-toa.png',dpi=600)




