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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

#from utils import *
from imods.stats import *
# %% [markdown]
# ## iCESM1

# %%
def index(ds):
    return (ds*areacella).sel(lat=slice(70,90)).sum(dim=['lat','lon'])/areacella.sel(lat=slice(70,90)).sum(dim=['lat','lon'])

areacella = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/CESM2-LE/CESM2-FV2.areacella.nc')['areacella'].squeeze().load()

# %%
file = '/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/isotope-nudging2/atm/isotope-nudging2.cam.h0.Q.197901-202212.plev.nc'

q = xr.open_dataset(file)['Q']
q.coords['time'] = np.arange('1979-01-01','2023-01-01',dtype='datetime64[M]')
q.coords['plev'] = q['plev']/100.
q

# %%
file = '/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/isotope-nudging2/atm/isotope-nudging2.cam.h0.TMQ.197901-202212.nc'

tmq = xr.open_dataset(file)['TMQ']
tmq.coords['time'] = np.arange('1979-01-01','2023-01-01',dtype='datetime64[M]')
tmq

# %%
file = '/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/isotope-nudging2/atm/isotope-nudging2.cam.h0.PRECL.197901-202212.nc'

icesm_precl = xr.open_dataset(file)['PRECL']
icesm_precl.coords['time'] = np.arange('1979-01-01','2023-01-01',dtype='datetime64[M]')
icesm_precl

# %%
file = '/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/isotope-nudging2/atm/isotope-nudging2.cam.h0.PRECC.197901-202212.nc'

icesm_precc = xr.open_dataset(file)['PRECC']
icesm_precc.coords['time'] = np.arange('1979-01-01','2023-01-01',dtype='datetime64[M]')
icesm_precc

# %%
icesm_prect = (icesm_precl+icesm_precc)*8.64e+7
icesm_prect_i = index(jja(icesm_prect)).sel(year=slice(1979,2022))
icesm_prect_i

# %%
icesm_qi = index(jja(q))
icesm_qi

# %%
ds_y = (icesm_qi.sel(plev=slice(1000,300)).integrate('plev')/-9.81).sel(year=slice(1981,2022))

r = linregress(ds_y,ds_y.year)
r.slope.values*10

# %%
icesm_tmqi = index(jja(tmq))
icesm_tmqi

# %%
ds_y = jja(tmq).sel(year=slice(1981,2022))
ds_x = ds_y.year

rtmq = linregress(ds_y,ds_x)
rtmq

# %%
ds_y = jja(q.mean('lon')).sel(year=slice(1981,2022))
ds_x = ds_y.year

rq_zon = linregress(ds_y,ds_x)
rq_zon

# %%
ds_y = jja(zonal_integration(q)).sel(year=slice(1981,2022))
ds_x = ds_y.year

rq_zon_int = linregress(ds_y,ds_x)
rq_zon_int

# %%
file = '/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/isotope-nudging2/atm/isotope-nudging2.cam.h0.Z3.197901-202212.plev.nc'

icesm_z3 = xr.open_dataset(file)
icesm_z3.coords['time'] = np.arange('1979-01-01','2023-01-01',dtype='datetime64[M]')
icesm_z3.coords['plev'] = icesm_z3['plev'] /100
icesm_z3

# %%
gpcp = xr.open_dataset('/raid/scratch/scratch-polar/obs/gpcp-rain/precip.mon.mean.new.nc')
gpcp_i = jja(area_weighted_ave(gpcp['precip'].sel(lat=slice(70,90))))
gpcp

# %%
gpcp_ea = mon(area_weighted_ave(gpcp['precip'].sel(lat=slice(50,70),lon=slice(0,120))),month=6)

# %%
linregress(gpcp_ea.sel(year=slice(1981,2022)),gpcp_ea.sel(year=slice(1981,2022)).year)*10

# %%
dware = xr.open_dataset('/raid/scratch/scratch-polar/obs/Delaware/air.mon.mean.v501.nc')
dware_i = jja(area_weighted_ave(dware['air'].sel(lat=slice(70,90))))
dware

# %%
ds_y = jja(icesm_z3['Z3'].mean('lon')).sel(year=slice(1981,2022))
ds_x = ds_y.year

rz3_zon = linregress(ds_y,ds_x)
rz3_zon

# %%
ds_y = jja(icesm_z3['Z3']).sel(plev=200,year=slice(1981,2022))
ds_x = ds_y.year

rz3 = linregress(ds_y,ds_x)
rz3

# %%
def icesm_reader(var):
    file = f'/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/isotope-nudging2/atm/isotope-nudging2.cam.h0.{var}.197901-202212.plev.nc'

    ds = xr.open_dataset(file)
    ds.coords['time'] = np.arange('1979-01-01','2023-01-01',dtype='datetime64[M]')
    return ds

# %%
icesm_u = icesm_reader('U')
icesm_v = icesm_reader('V')
icesm_q = icesm_reader('Q')

# %%
icesm_vivt = (icesm_v['V']*icesm_q['Q']).integrate('plev')/(-9.81)
icesm_uivt = (icesm_u['U']*icesm_q['Q']).integrate('plev')/(-9.81)

# %%
def zonal_integration(ds):
    REARTH = 6.37122e6 #km
    rad2deg = 180/np.pi
    dx = ds.lon.diff('lon')
    area = (dx*3.1415926*REARTH/360.*2.)

    index = (ds * np.cos(ds.lat / 180. * 3.1415926) * area).sum(('lon'),skipna=True)
    return index

# %%
vivt_70 = zonal_integration(icesm_vivt).sel(lat=70,method='nearest')

# %%
ds_y = jja(icesm_vivt).sel(year=slice(1981,2022))
ds_x = ds_y.year

rv = linregress(ds_y,ds_x)

# %%
ds_y = jja(icesm_uivt).sel(year=slice(1981,2022))
ds_x = ds_y.year

ru = linregress(ds_y,ds_x)

# %%
ds_y = jja(np.sqrt(icesm_uivt*icesm_uivt+icesm_vivt*icesm_vivt)).sel(year=slice(1981,2022))
ds_x = ds_y.year

rt = linregress(ds_y,ds_x)

# %% [markdown]
# ## ERA5

# %%
file = '/raid/scratch/scratch-polar/obs/ERA5/precipitaion/download.1979-2022.nc'

era5_precip = xr.open_dataset(file)
era5_precip

# %%
era5_prect_i = 32.8767*area_weighted_ave(jja(era5_precip['tp']).sel(latitude=slice(90,70)).rename({'latitude':'lat','longitude':'lon'})).sel(year=slice(1979,2022))
era5_prect_i

# %% [markdown]
# files = sorted(glob.glob('/raid/scratch/scratch-polar/obs/ERA5/pressure/all/era5*nc'))
# 
# def preprocess(ds):
#     return ds.integrate('level')/9.81
# 
# q = [preprocess(xr.open_dataset(file)) for file in files]
# q = xr.concat(q,dim='time')
# q = q.rename({'latitude':'lat','longitude':'lon'})
# q

# %%
files = sorted(glob.glob('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/ERA5/q/*nc'))

era5_q = xr.open_mfdataset(files,combine='nested',concat_dim='time')['q']
era5_q.coords['latitude'] = icesm_q['lat'].values
era5_q.coords['longitude'] = icesm_q['lon'].values
era5_q = era5_q.rename({'latitude':'lat','longitude':'lon','level':'lev'})
era5_q

# %%
ds_y = jja(era5_q.integrate('lev')/9.81).sel(year=slice(1981,2022)).load()
ds_x = ds_y.year

erq = linregress(ds_y,ds_x)
erq

# %%
ds_y = jja(era5_q.mean('lon')).sel(year=slice(1981,2022)).load()
ds_x = ds_y.year

erq_zon = linregress(ds_y,ds_x)
erq_zon

# %%
ds_y = jja(zonal_integration(era5_q)).sel(year=slice(1981,2022)).load()
ds_x = ds_y.year

erq_zon_int = linregress(ds_y,ds_x)
erq_zon_int

# %%
erq_i = index(jja((era5_q.integrate('lev')/9.81).sel(lat=slice(70,90))))
erq_i

# %%
files = sorted(glob.glob('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/ERA5/z3/*nc'))

era5_z3 = xr.open_mfdataset(files,combine='nested',concat_dim='time')['z']
era5_z3.coords['latitude'] = icesm_q['lat'].values
era5_z3.coords['longitude'] = icesm_q['lon'].values
era5_z3 = era5_z3.rename({'latitude':'lat','longitude':'lon','level':'lev'})
era5_z3

# %%
files = sorted(glob.glob('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/ERA5/u/*nc'))

era5_u = xr.open_mfdataset(files,combine='nested',concat_dim='time')['u']
era5_u.coords['latitude'] = icesm_q['lat'].values
era5_u.coords['longitude'] = icesm_q['lon'].values
era5_u = era5_u.rename({'latitude':'lat','longitude':'lon','level':'lev'})
era5_u

# %%
files = sorted(glob.glob('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/ERA5/v/*nc'))

era5_v = xr.open_mfdataset(files,combine='nested',concat_dim='time')['v']
era5_v.coords['latitude'] = icesm_q['lat'].values
era5_v.coords['longitude'] = icesm_q['lon'].values
era5_v = era5_v.rename({'latitude':'lat','longitude':'lon','level':'lev'})
era5_v

# %%
era5_vivt = (era5_v*era5_q).integrate('lev')/9.81
era5_uivt = (era5_u*era5_q).integrate('lev')/9.81

# %%
ds_y = jja(era5_vivt).sel(year=slice(1981,2022)).load()
ds_x = ds_y.year

ervivt = linregress(ds_y,ds_x)
ervivt

# %%
ds_y = jja(era5_uivt).sel(year=slice(1981,2022)).load()
ds_x = ds_y.year

eruivt = linregress(ds_y,ds_x)
eruivt

# %%
ds_y = jja(era5_z3).sel(lev=200,year=slice(1981,2022)).load()
ds_x = ds_y.year

erz3 = linregress(ds_y,ds_x)
erz3

# %%
ds_y = jja(era5_z3.mean('lon')).sel(year=slice(1981,2022)).load()
ds_x = ds_y.year

erz3_zon = linregress(ds_y,ds_x)
erz3_zon

# %%
cesm2_areacella = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/CESM2-LE/CESM2.areacella.nc')['areacella'].squeeze()

# %%
def preprocess(ds):
    cesm2_areacella.coords['lat'] = ds['lat']
    cesm2_areacella.coords['lon'] = ds['lon']
    index = (ds['PRECT']*cesm2_areacella).sel(lat=slice(70,90)).sum(dim=['lat','lon'])/cesm2_areacella.sel(lat=slice(70,90)).sum(dim=['lat','lon'])
    return index.squeeze()

def process(i):
    files = sorted(glob.glob(f'/home/scratch-qinghua3/exp/cesm-len2/rain/his/cmip6/1001/b.e21.BHISTcmip6.f09_g17.LE2*.{i:03.0f}.cam.h0.PRECT*nc'))

    cesm2_prect = xr.open_mfdataset(files[-5:],combine='nested',concat_dim='time',preprocess=preprocess)
    cesm2_prect.coords['time'] = np.arange('1970-01-01','2015-01-01',dtype='datetime64[M]')
    
    files = sorted(glob.glob(f'/home/scratch-qinghua3/exp/cesm-len2/rain/ssp/cmip6/1001/b.e21.BSSP370cmip6.f09_g17.LE2*.{i:03.0f}.cam.h0.PRECT.201501-202412.nc'))

    ssp = xr.open_mfdataset(files[0],combine='nested',concat_dim='time',preprocess=preprocess)
    ssp.coords['time'] = np.arange('2015-01-01','2025-01-01',dtype='datetime64[M]')
    
    return xr.concat([cesm2_prect,ssp],dim='time')

cesm2_prect = [process(i) for i in range(1,11)]
cesm2_prect = xr.concat(cesm2_prect,dim='ens')
cesm2_prect.coords['ens'] = np.arange(1,11,step=1)
cesm2_prect_i = jja(cesm2_prect*8.64e+7).sel(year=slice(1979,2022))
cesm2_prect_i

# %%
cesm2_tmq = jja(xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/CESM2-LE/cesm2-le.cmip6.TMQ.arctic.month.nc')['TMQ']).sel(year=slice(1979,2014))
cesm2_ssp = jja(xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/CESM2-LE/cesm2-le.ssp370.TMQ.arctic.month.nc')['TMQ']).sel(year=slice(2015,2022))
cesm2_tmq = xr.concat([cesm2_tmq,cesm2_ssp],dim='year')
cesm2_tmq

# %%
ds_y = erq_i.sel(year=slice(1981,2022))*1000
ds_x = ds_y.year

r = linregress(ds_y,ds_x)
r.slope.values

# %%
ds_y =(1000*icesm_qi.integrate('plev')/-9.81).sel(year=slice(1981,2022))
ds_x = ds_y.year

r = linregress(ds_y,ds_x)
r.slope.values

# %%
ds_y = cesm2_tmq.mean('member_id').sel(year=slice(1981,2022))*10
ds_x = ds_y.year

r = linregress(ds_y,ds_x)
r.slope.values

# %% [markdown]
# ## OBS RAIN

# %%
file = '/raid/scratch/scratch-itbaxter/obs/GPCC/precip.comb.v2020to2019-v2020monitorafter.total.nc'

gpcc = xr.open_dataset(file)
gpcc

# %%
igpcc = jja(area_weighted_ave(gpcc['precip'].sel(lat=slice(90,70)))) #/30 #.sel(year=slice(1979,2022))
igpcc

# %%
gpm = jja(xr.open_dataset('/raid/scratch/scratch-itbaxter/obs/GPM/gpm.prec.arctic.70-90N.month.2000-2023.nc')['__xarray_dataarray_variable__'])
gpm

# %%
file = '/raid/scratch/scratch-polar/obs/cmap-rain/precip.mon.mean.nc'

cmap = jja(xr.open_dataset(file))
cmap

# %%
icmap = area_weighted_ave(cmap['precip'].sel(lat=slice(90,70))).sel(year=slice(1979,2022))
icmap

# %%
files = sorted(glob.glob('/raid/scratch/scratch-polar/obs/Delaware/precip.mon.total.*.nc'))

dware1 = xr.open_dataset(files[0])['precip'].resample(time='QS-DEC').sum('time')
dware = dware1.where(dware1['time.month'] == 6).groupby('time.year').mean('time').sel(year=slice(1979,2022))
idware = area_weighted_ave(dware.sel(lat=slice(90,70)))
idware

# %%
idware = area_weighted_ave(dware.sel(lat=slice(90,70)))
idware

# %% [markdown]
# ## PLOT

# %%
# Create a custom RdYlBu colormap with white in the middle
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 7
mpl.rcParams['hatch.color'] = 'silver'

fig = plt.figure(figsize=(7.5,8))
## Panel 1 --------------------------------------------------
ax = fig.add_subplot(321)
ax.set_xlim([1978,2023])
#ax.set_ylim([100,1000])
ax.set_ylabel(r'JJA Total Precipitable Water [kg $\mathrm{m^{-2}}$]')
ax.set_xlabel('year')
for axis in ['bottom','left']:
    ax.spines[axis].set_linewidth(1.25)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    
ax.text(-0.03, 1.05, 'a', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

fills = ax.fill_between(np.arange(1979,2023,step=1),cesm2_tmq.min('member_id'),cesm2_tmq.max('member_id'),color='silver',alpha=0.4)

icesm_qi_sel = (100*icesm_qi.integrate('plev')/-9.81).sel(year=slice(1979,2022))
ln2 = ax.plot(icesm_qi_sel['year'],icesm_qi_sel,
              color='k',
              linewidth=1.5,
              zorder=11,
              label='iCESM1')
erq_i_sel = erq_i.sel(year=slice(1979,2022))
ln3 = ax.plot(erq_i_sel['year'],erq_i_sel*100,
              color='#AA4499',
              linewidth=1.5,
              zorder=11,
              label='ERA5')

ln1 = ax.plot(np.arange(1979,2023,step=1),cesm2_tmq.mean('member_id').values,
              c='#6699CC',linewidth=1.5,label='CESM2-LE',zorder=10)

plt.legend(frameon=False,fontsize=7)

## Panel 2 --------------------------------------------------
ax = fig.add_subplot(322)
ax.set_xlim([1978,2023])
ax.set_ylim([0.62,1.5])
ax.set_ylabel(r'JJA Total Precipitation [mm $\mathrm{day^{-1}}$]')
ax.yaxis.set_label_position("right")
ax.set_xlabel('year')
ax.yaxis.tick_right()
for axis in ['bottom','right']:
    ax.spines[axis].set_linewidth(1.25)
ax.spines[['left', 'top']].set_visible(False)
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.02))    

ax.text(-0.03, 1.05, 'b', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

fills = ax.fill_between(np.arange(1979,2023,step=1),cesm2_prect_i.min('ens'),cesm2_prect_i.max('ens'),
                        zorder=10,
                          label='CESM2-LE Spread',
                        color='silver',alpha=0.4)

obs = xr.concat([idware*10/92,icmap,gpcp_i,igpcc.sel(year=slice(1979,2022))/30,24*gpm.sel(year=slice(1979,2022))],dim='product').sel(year=slice(1979,2022))

lnobs = ax.fill_between(np.arange(1979,2023,step=1),obs.min('product'),obs.max('product'),
                          zorder=11,
                          label='Obs Spread',
                          color='tab:green',alpha=0.4)

leg2 = plt.legend(loc='lower left', 
                 bbox_to_anchor=[0.4, 0.9], 
                 frameon=False,fontsize=7)
ax.add_artist(leg2)

ln2, = ax.plot(icesm_prect_i['year'],icesm_prect_i,zorder=20,linewidth=1.5,c='k',label='iCESM1')
ln3, = ax.plot(era5_prect_i['year'],era5_prect_i*30,zorder=20,linewidth=1.5,c='#AA4499',label='ERA5')                 
                 
ln10, = ax.plot(np.arange(1979,2023,step=1),cesm2_prect_i.mean('ens').sel(year=slice(1979,2022)).values,
               zorder=11,
               c='#6699CC',linewidth=1.5,label='CESM2-LE')

leg1 = plt.legend(handles=[ln2,ln3,ln10],loc='lower left', 
                 bbox_to_anchor=[0.1, 0.85], 
                 frameon=False,fontsize=7)

# Panel 3--------------------------------------------------
ax = fig.add_subplot(323,projection=ccrs.NorthPolarStereo())
#ax = fig.add_axes([0,0.35,0.5,0.35],ccrs.NorthPolarStereo())
ax.set_extent([-180,180,40,90],ccrs.PlateCarree())
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')

ax.text(-0.4, 0.35, 'ERA5', weight='bold',
        fontsize=17,
        rotation=90,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

ax.text(-0.03, 1.05, 'c', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

colors1 = plt.cm.GnBu(np.linspace(0, 1, 256))
colors2 = plt.cm.YlOrBr(np.linspace(0, 1, 256))[::-1]
new_colors = np.vstack((colors2[:], np.ones((40, 4)), colors1[:]))
new_cmap = LinearSegmentedColormap.from_list('RdYlBu_r_white', new_colors, N=16)

cmap = colormaps.BlueWhiteOrangeRed
bounds = np.arange(-9,9.1,1)/10
erqs,lons = add_cyclic_point(erq.slope,erq['lon'])
p = ax.contourf(lons,erq['lat'],erqs*1000,
                cmap=new_cmap,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

s = ax.contourf(erq['lon'],erq['lat'],erq.slope.where(erq.p > 0.05),
                cmap=new_cmap,
                hatches=['//'],
                alpha=0,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())


lats,_ = xr.broadcast(rz3['lat'],rz3.slope)
latline = ax.contour(rz3['lon'],rz3['lat'],lats,
                     levels=[70],
                     colors='tab:purple',
                     linewidths=1.0,
                     alpha=0.5,
                     #linestyle='--',
                     transform=ccrs.PlateCarree()
                    )

# Geopotential Height
c = ax.contour(erz3['lon'],erz3['lat'],erz3.slope,
               levels=[-6,-4,6,10,14,18],
               linewidths=0.75,
               colors='k',
               transform=ccrs.PlateCarree()
              )

z = ax.contour(erz3['lon'],erz3['lat'],erz3.slope,
               levels=[0],
               linewidths=1.5,
               colors='k',
               transform=ccrs.PlateCarree()
              )

def fmt(x):
    return f'{x:2.0f}'

clab1 = ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=7)
clab2 = ax.clabel(z, z.levels, inline=True, fmt=fmt, fontsize=7)
for l in clab1+clab2:
    l.set_rotation(0)

rvs = ervivt.slope #.where(rt.p < 0.05)
rus = eruivt.slope #.where(rt.p < 0.05)
Q = ax.quiver(rvs['lon'], rvs['lat'], rus*10, rvs*10, 
          color='red',
          edgecolor='k',
          transform=ccrs.PlateCarree(), 
          regrid_shape=15)

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

ax.set_boundary(circle, transform=ax.transAxes)
ax.spines['geo'].set_linewidth(1.25)

# Add gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.35,  
                  x_inline=False,
                  y_inline=False,
                  rotate_labels=False, color='k', 
                  linestyle='--', alpha=0.4)

gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

#cmap = colormaps.precip_diff_12lev

## Panel 4 --------------------------------------------------
ax = fig.add_subplot(324)
ax.set_xlim([-90,90])
ax.set_ylim([100,1000])
ax.set_ylabel('Pressure [hPa]')
ax.set_xlabel('Latitude')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.25)
ax.invert_yaxis()
ax.set_yscale("log")
ax.yaxis.set_major_locator(FixedLocator([100,200,300,400,600,800,1000])) #AutoLocator())
ax.yaxis.set_major_formatter('{x:3.0f}')
ax.set_xticks(np.arange(-90, 91, 30), [r'90°S', r'60°S', r'30°S', r'0°', r'30°N', r'60°N', r'90°N'])
ax.tick_params(right=True)

ax.text(-0.03, 1.05, 'd', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

cmap = colormaps.BlueWhiteOrangeRed
bounds = np.arange(-0.16,0.161,0.02)
#rtmqs,lons = add_cyclic_point(rtmq.slope,rtmq['lon'])
p = ax.contourf(erq_zon['lat'],erq_zon['lev'],erq_zon.slope*10000,
                cmap=new_cmap,
                extend='both',
                levels=bounds,
                )

s = ax.contourf(erq_zon['lat'],erq_zon['lev'],erq_zon.slope.where(erq_zon.p > 0.05)*10000,
                cmap=new_cmap,
                hatches=['//'],
                alpha=0,
                extend='both',
                levels=bounds,
                )

ax.axvline(70,
           linestyle='--',
           linewidth=1.0,
           c='tab:purple',
           alpha=0.7,
          )

# Geopotential Height
c = ax.contour(erz3_zon['lat'],erz3_zon['lev'],erz3_zon.slope,
               levels=[-6,-3,3,6,9,12,15],
               linewidths=0.75,
               colors='k',
              )

z = ax.contour(erz3_zon['lat'],erz3_zon['lev'],erz3_zon.slope,
               levels=[0],
               linewidths=1.5,
               colors='k',
              )
def fmt(x):
    return f'{x:2.0f}'

clab1 = ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=7)
clab2 = ax.clabel(z, z.levels, inline=True, fmt=fmt, fontsize=7)
for l in clab1+clab2:
    l.set_rotation(0)

# Panel 5--------------------------------------------------
ax = fig.add_subplot(325,projection=ccrs.NorthPolarStereo())
ax.set_extent([-180,180,40,90],ccrs.PlateCarree())
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')

ax.text(-0.4, 0.25, 'iCESM1', weight='bold',
        fontsize=17,
        rotation=90,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

ax.text(-0.03, 1.05, 'e', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

cmap = colormaps.BlueWhiteOrangeRed
bounds = np.arange(-9,9.1,1)/10
rtmqs,lons = add_cyclic_point(rtmq.slope,rtmq['lon'])
p = ax.contourf(lons,rtmq['lat'],rtmqs*10,
                cmap=new_cmap,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

s = ax.contourf(rtmq['lon'],rtmq['lat'],rtmq.slope.where(rtmq.p > 0.05),
                cmap=new_cmap,
                hatches=['//'],
                alpha=0,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())


lats,_ = xr.broadcast(rz3['lat'],rz3.slope)
latline = ax.contour(rz3['lon'],rz3['lat'],lats,
                     levels=[70],
                     colors='tab:purple',
                     linewidths=1.0,
                     alpha=0.5,
                     #linestyle='--',
                     transform=ccrs.PlateCarree()
                    )

# Geopotential Height
c = ax.contour(rz3['lon'],rz3['lat'],rz3.slope*10,
               levels=[-6,-4,6,10,14,18],
               linewidths=0.75,
               colors='k',
               transform=ccrs.PlateCarree()
              )

z = ax.contour(rz3['lon'],rz3['lat'],rz3.slope*10,
               levels=[0],
               linewidths=1.5,
               colors='k',
               transform=ccrs.PlateCarree()
              )

def fmt(x):
    return f'{x:2.0f}'

clab1 = ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=7)
clab2 = ax.clabel(z, z.levels, inline=True, fmt=fmt, fontsize=7)
for l in clab1+clab2:
    l.set_rotation(0)

rvs = rv.slope 
rus = ru.slope 
Q = ax.quiver(rv['lon'], rv['lat'], rus*10, rvs*10, 
          color='red',
          edgecolor='k',
          transform=ccrs.PlateCarree(), 
          regrid_shape=15)

qk = ax.quiverkey(Q, 0.65, -0.1, 10, r'IVT: $10\ \mathrm{\frac{kg}{m\ s\ decade}}}$', 
                  labelpos='E',
                 )

# Colorbar

cax = fig.add_axes([0.1,0.05,0.35,0.01])
cb = plt.colorbar(p, cax=cax,orientation='horizontal', 
                  drawedges=True,
                  ticks=bounds[::3]
                 )
cb.set_ticklabels(bounds[::3])
cb.set_label(r'JJA Total Precipitable Water [kg $\mathrm{m^{-2}\ decade^{-1}}$] ')
cb.outline.set_color('k')
cb.outline.set_linewidth(0.8)
cb.dividers.set_color('k')
cb.dividers.set_linewidth(0.8)
cb.ax.tick_params(size=0)

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

ax.set_boundary(circle, transform=ax.transAxes)
ax.spines['geo'].set_linewidth(1.25)

# Add gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.35,  
                  x_inline=False,
                  y_inline=False,
                  rotate_labels=False, color='k', 
                  linestyle='--', alpha=0.4)
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

## Panel 6 --------------------------------------------------
ax = fig.add_subplot(326)
ax.set_xlim([-90,90])
ax.set_ylim([100,1000])
ax.set_ylabel('Pressure [hPa]')
ax.set_xlabel('Latitude')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.25)
ax.invert_yaxis()
ax.set_yscale("log")
ax.yaxis.set_major_locator(FixedLocator([100,200,300,400,600,800,1000])) #AutoLocator())
ax.yaxis.set_major_formatter('{x:3.0f}')
ax.set_xticks(np.arange(-90, 91, 30), [r'90°S', r'60°S', r'30°S', r'0°', r'30°N', r'60°N', r'90°N'])
ax.tick_params(right=True)

ax.text(-0.03, 1.05, 'f', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

cmap = colormaps.BlueWhiteOrangeRed
bounds = np.arange(-0.16,0.161,0.02)
p = ax.contourf(rq_zon['lat'],rq_zon['plev'],rq_zon.slope*10000,
                cmap=new_cmap,
                extend='both',
                levels=bounds,
                )

s = ax.contourf(rq_zon['lat'],rq_zon['plev'],rq_zon.slope.where(rq_zon.p > 0.05)*10000,
                cmap=new_cmap,
                hatches=['//'],
                alpha=0,
                extend='both',
                levels=bounds,
                )

ax.axvline(70,
           linestyle='--',
           linewidth=1.0,
           c='tab:purple',
           alpha=0.7,
          )

# Geopotential Height
c = ax.contour(rz3_zon['lat'],rz3_zon['plev'],rz3_zon.slope*10,
               levels=[-6,-3,3,6,9,12,15],
               linewidths=0.75,
               colors='k',
              )

z = ax.contour(rz3_zon['lat'],rz3_zon['plev'],rz3_zon.slope*10,
               levels=[0],
               linewidths=1.5,
               colors='k',
              )
def fmt(x):
    return f'{x:2.0f}'

clab1 = ax.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=7)
clab2 = ax.clabel(z, z.levels, inline=True, fmt=fmt, fontsize=7)
for l in clab1+clab2:
    l.set_rotation(0)

# Colorbar
cax = fig.add_axes([0.535,0.05,0.385,0.01])
cb = plt.colorbar(p, cax=cax,orientation='horizontal', 
                  drawedges=True,
                 )
cb.set_label(r'JJA Spec. Hum. [g $\mathrm{kg^{-1}\ decade^{-1}}$] ')
cb.outline.set_color('k')
cb.outline.set_linewidth(0.8)
cb.dividers.set_color('k')
cb.dividers.set_linewidth(0.8)
cb.ax.tick_params(size=0)

fig.subplots_adjust(top=0.96,bottom=0.11,left=0.07,right=0.93,hspace=0.3,wspace=0.1)
fig.savefig('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/moisture_tagging_paper/plots/moisture_tagging-figure_1-6panel_blue-2shaded.png',dpi=600)

# %%
# Create a custom RdYlBu colormap with white in the middle
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 7
mpl.rcParams['hatch.color'] = 'silver'

colors = ['#332288','#117733','#44AA99','#88CCEE','#DDCC77','#CC6677','#AA4499','#882255']
colors = ['#000000','#E69F00','#56B4E9','#009E73','#F0E442','#0072B2','#D55E00','#CC79A7']

fig = plt.figure(figsize=(6,4))

## Panel 2 --------------------------------------------------
ax = fig.add_subplot(111)
ax.set_xlim([1978,2023])
ax.set_ylim([0.62,1.4])
ax.set_ylabel(r'JJA Total Precipitation [mm $\mathrm{day^{-1}}$]')
ax.yaxis.set_label_position("left")
ax.set_xlabel('year')
#ax.yaxis.tick_right()
for axis in ['bottom','left']:
    ax.spines[axis].set_linewidth(1.25)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.02))    

ax.text(-0.03, 1.05, 'a', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

leg2 = plt.legend(loc='lower left', 
                 bbox_to_anchor=[0.4, 0.85], 
                 frameon=False,fontsize=7)
ax.add_artist(leg2)

lnd, = ax.plot(idware['year'],idware*10/92,linewidth=1.5,zorder=20,c=colors[1],label='UDel-TS')
ln0, = ax.plot(icmap['year'],icmap,linewidth=1.5,zorder=20,c=colors[3],label='CMAP')
ln1, = ax.plot(gpcp_i['year'],gpcp_i,linewidth=1.5,zorder=20,c=colors[5],label='GPCP') # (jja(igpcc)/30)
ln4, = ax.plot(igpcc.sel(year=slice(1979,2022))['year'],igpcc.sel(year=slice(1979,2022))/30,linewidth=1.5,zorder=20,c=colors[6],linestyle='-',label='GPCC') 
ln5, = ax.plot(gpm.sel(year=slice(1979,2022))['year'],24*gpm.sel(year=slice(1979,2022)),linewidth=1.5,zorder=20,c=colors[7],linestyle='-',label='GPM-IMERG') 
ln2, = ax.plot(icesm_prect_i['year'],icesm_prect_i,zorder=20,linewidth=1.5,c='k',label='iCESM1')

leg2 = plt.legend(handles=[ln0,ln1,lnd],loc='lower left', 
                 bbox_to_anchor=[0.3, 0.8], 
                 frameon=False,fontsize=7)
ax.add_artist(leg2)

leg3 = plt.legend(handles=[ln4,ln5,ln2],loc='lower left', 
                 bbox_to_anchor=[0.45, 0.8], 
                 frameon=False,fontsize=7)


fig.subplots_adjust(top=0.90,bottom=0.11,left=0.1,right=0.97,hspace=0.3,wspace=0.1)
fig.savefig('./plots/moisture_tagging-figure_1-6panel_blue-precip.png',dpi=600)



# %%
