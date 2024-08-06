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
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.patches as mpatches

from fastloop import f_tkji2tpji

from imods.stats import *


# %%
file = '/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/aar/out.iCESM1-AR.daily.1979-2022.nc'

nud = xr.open_dataset(file)
nud

# %%
nud_freq = xr.ones_like(nud['kstatusmap']).where(nud['kstatusmap'] > 0)
nud_freq = (nud_freq.where(nud_freq['time.season'] == 'JJA').groupby('time.year').sum('time')/92.).squeeze()
nud_freq

# %%
icesm = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/moisture_tagging_paper/icesm.ar_freq.jja.nc')['kstatusmap'].squeeze()
icesm

# %%
def process(file):
    ds = xr.open_dataset(file)['kstatusmap']
    ones = xr.ones_like(ds).where(ds > 0)
    ds_jja = (ones.where(ds['time.season'] == 'JJA').groupby('time.year').sum('time')/(92*4))
    return ds_jja.squeeze()

# %%
files = sorted(glob.glob('/home/scratch-qinghua3/exp/cesm-len2/ivt-6hour/u/his/*uIVT.2010010100-2014123100.nc'))
mems = [f'{file.split(".")[4]}-{file.split(".")[5]}' for file in files]
[print(mem,i+1) for i,mem in enumerate(mems)];

# %%
new_mems = []
for idx,mem in enumerate(mems):
    _,ic,num = mem.split('-')
    
    if int(ic) < 1200:
        icn = int(ic) #-10
    else:
        icn = int(ic)
    
    new_mems.append(f'r{int(num)}i{icn}p1f2')
        
print(new_mems)

# %%
ds = xr.open_mfdataset('/raid/scratch/scratch-itbaxter/exp/CESM2-LE/ivt/jja_ar*nc',combine='nested',concat_dim='member_id')['kstatusmap'].squeeze()/4.
ds.coords['member_id'] = new_mems #np.arange(1,41,step=1)
ds.load()

# %%
cesm2_jja_mean = ds.sel(year=slice(1979,2018)).mean('member_id')
ds_x = cesm2_jja_mean.year

c2r = linregress(cesm2_jja_mean,ds_x,dim='year')
c2r

# %%
cesm2_jja = ds.sel(year=slice(1979,2022))
ds_x = cesm2_jja.year

spread = linregress(cesm2_jja,ds_x,dim='year')
spread

# %%
areacella2 = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/CESM2-LE/CESM2.areacella.nc')['areacella'].squeeze()
areacella2

# %%
index_cesm2 = area_weighted_ave(cesm2_jja.sel(lat=slice(70,90)))

# %%
index_green = area_weighted_ave(cesm2_jja.sel(lat=slice(60,85),lon=slice(270,310)))

# %%
index_r2 = linregress(index_cesm2,index_cesm2.year)
index_r2

# %%
era5 = xr.open_dataset('/raid/scratch/scratch-polar/obs/AR/ERA5/ERA5_AR_v2.nc')
era5

# %%
freq  = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/Moisture-tagging/notebooks/era5.ar.freq.jja.nc')
freq

# %%
ds_y = freq['ar_binary_tag'].sel(year=slice(1981,2019))
r = linregress(ds_y,ds_y.year)
r

# %%
out_08 = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/aar/out.iCESM1-AR.daily.2008.nc')

ones_08 = xr.ones_like(out_08['kstatusmap']).where(out_08['kstatusmap'] >= 1).fillna(0)
freq_08 = (ones_08.sel(time=slice('2008-06-01','2008-08-31')).sum('time')/92).squeeze() #.plot()
freq_08['year'] = 2008

icesm_freq = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/moisture_tagging_paper/icesm.ar_freq.jja.nc')
icesm_freq2 = icesm_freq.where(icesm_freq['year'] == 2008, freq_08)

ir = linregress(icesm_freq['kstatusmap'].sel(year=slice(1981,2022)),icesm_freq.sel(year=slice(1981,2022)).year)
ir

# %%
# Indices
#ERA5
index_era5 = area_weighted_ave(freq['ar_binary_tag'].sel(year=slice(1979,2022)).sel(lat=slice(90,70)))

# CESM2-LE
index_cesm2 = area_weighted_ave(ds.sel(year=slice(1979,2022)).sel(lat=slice(70,90)))

# iCESM1
areacella = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/CESM2-LE/CESM2-FV2.areacella.nc')['areacella'].squeeze().load()

index_08 = (freq_08*areacella).sel(lat=slice(70,90)).sum(dim=['lat','lon'])/areacella.sel(lat=slice(70,90)).sum(dim=['lat','lon'])
index_freq = (icesm_freq*areacella).sel(lat=slice(70,90)).sum(dim=['lat','lon'])/areacella.sel(lat=slice(70,90)).sum(dim=['lat','lon'])
index_freq = index_freq['kstatusmap'].where(index_freq['year'] != 2008, index_08.values)
index_freq

# %%
freq_80s = icesm_freq['kstatusmap'].sel(year=slice(1985,1995)).mean('year')
freq_00s = icesm_freq['kstatusmap'].sel(year=slice(1995,2012)).mean('year')
freq_10s = icesm_freq['kstatusmap'].sel(year=slice(2015,2022)).mean('year')
freq_mean = icesm_freq['kstatusmap'].mean('year')

p = (100*(freq_00s-freq_mean)).plot(transform=ccrs.PlateCarree(),subplot_kws={'projection':ccrs.NorthPolarStereo()})
p.axes.coastlines()
p.axes.set_extent([-180,180,40,90],ccrs.PlateCarree())

# %%
index_08 = (freq_08*areacella).sel(lat=slice(60,85),lon=slice(270,310)).sum(dim=['lat','lon'])/areacella.sel(lat=slice(60,85),lon=slice(270,310)).sum(dim=['lat','lon'])
index_freq_green = (icesm_freq*areacella).sel(lat=slice(60,85),lon=slice(270,310)).sum(dim=['lat','lon'])/areacella.sel(lat=slice(60,85),lon=slice(270,310)).sum(dim=['lat','lon'])
index_freq_green = index_freq_green['kstatusmap'].where(index_freq_green['year'] != 2008, index_08.values)
index_freq_green

# %%
slope_c2i = linregress(index_cesm2,index_cesm2.year).slope
slope_green = linregress(index_green,index_green.year).slope

slope_freq = linregress(index_freq.sel(year=slice(1981,2022)),index_freq.sel(year=slice(1981,2022)).year).slope
slope_freq_g = linregress(index_freq_green.sel(year=slice(1981,2022)),index_freq_green.sel(year=slice(1981,2022)).year).slope

# %%
green_sel = spread.where(slope_green > slope_freq_g.values,drop=True)
green_sel

# %%
ds_new = ds
ds_new.coords['member_id'] = new_mems
ds_new

# %%
cesm2_his = jja(xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/CESM2-LE/cesm2-le.hist_smbb.TMQ.arctic.month_new.nc')['TMQ']).sel(year=slice(1979,2014))
cesm2_ssp = jja(xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/CESM2-LE/cesm2-le.ssp370_smbb.TMQ.arctic.month_new.nc')['TMQ']).sel(year=slice(2015,2022))
cesm2_tmq = xr.concat([cesm2_his,cesm2_ssp],dim='year')

final_mems = [x for x in cesm2_tmq.member_id.values if x in new_mems]

cesm2_tmq_sel = cesm2_tmq.sel(member_id=final_mems)
cesm2_tmq_sel

# %%
rtmq = linregress(cesm2_tmq_sel.sel(year=slice(1979,2018)),cesm2_tmq_sel.sel(year=slice(1979,2018)).year)
rtmq_ex = rtmq.where(rtmq.slope > rtmq.slope.mean('member_id')+1.0*rtmq.slope.std('member_id'),drop=True)
rtmq_ex

# %%
cesm2_tmq.plot.line(x='year');

cesm2_tmq.sel(member_id=rtmq_ex['member_id']).plot.line(x='year',c='k',linewidth=2);

# %%
green_fast = spread.where(slope_green > slope_green.mean('member_id')+1.5*slope_green.std('member_id'),drop=True)
green_fast

# %%
cesm2_jja = ds.sel(year=slice(1979,2018)).where(slope_green > slope_green.mean('member_id')+1.5*slope_green.std('member_id'),drop=True)
ds_x = cesm2_jja.year

spreadm = linregress(cesm2_jja.mean('member_id'),ds_x,dim='year')
spreadm

# %%
green_fast.member_id.values,new_mems[21],new_mems[27]

# %%
new_spread = spread
new_spread.coords['member_id'] = new_mems

print(rtmq_ex.member_id.values)
new_fast = new_spread.sel(member_id=rtmq_ex.member_id.values)
new_fast

# %%
ds_y = ds.sel(year=slice(1979,2018),member_id=rtmq_ex.member_id.values).mean('member_id')
print(ds_y)

new_fastm = linregress(ds_y,ds_y.year)
new_fastm

# %% [markdown]
# ## NEW

# %%
file = '/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/isotope-nudging2/atm/isotope-nudging2.cam.h0.Z3.197901-202212.plev.nc'

icesm_z3 = xr.open_dataset(file)
icesm_z3.coords['time'] = np.arange('1979-01-01','2023-01-01',dtype='datetime64[M]')
icesm_z3.coords['plev'] = icesm_z3['plev'] /100
icesm_z3

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

icesm_u = icesm_reader('U')
icesm_v = icesm_reader('V')
icesm_q = icesm_reader('Q')

icesm_vivt = (icesm_v['V']*icesm_q['Q']).integrate('plev')/(-9.81)
icesm_uivt = (icesm_u['U']*icesm_q['Q']).integrate('plev')/(-9.81)

# %%
ds_y = jja(icesm_vivt).sel(year=slice(1981,2022))
ds_x = ds_y.year

rv = linregress(ds_y,ds_x)

# %%
ds_y = jja(icesm_uivt).sel(year=slice(1981,2022))
ds_x = ds_y.year

ru = linregress(ds_y,ds_x)

# %%
abs((1000*ir.slope.where(ir.p < 0.05))).min().values

# %% [markdown]
# ## CESM2-LE Z200

# %%
green_fast.member_id.values

# %%
cesm2_z3hist = jja(xr.open_mfdataset('/raid/scratch/scratch-itbaxter/exp/CESM2-LE/Z200/b.e21.BHIST.f09_g17.cesm2-le.*f2.cam.h0.Z200.*.plev.nc',combine='nested',concat_dim='member_id')['Z3']).sel(year=slice(1979,2014))
cesm2_z3ssp = jja(xr.open_mfdataset('/raid/scratch/scratch-itbaxter/exp/CESM2-LE/Z200/b.e21.SSP370.f09_g17.cesm2-le.*f2.cam.h0.Z200.*.plev.nc',combine='nested',concat_dim='member_id')['Z3']).sel(year=slice(2015,2022))
cesm2_z3 = xr.concat([cesm2_z3hist,cesm2_z3ssp],dim='year')

final_mems = [x for x in cesm2_z3.member_id.values if x in new_mems]

cesm2_z3 = cesm2_z3.sel(member_id=final_mems).squeeze()
cesm2_z3

# %%
cesm2_z3.load()

# %%
ds_y = cesm2_z3.sel(year=slice(1979,2018)).squeeze()

cesm2_z3r = linregress(ds_y,ds_y.year)
cesm2_z3r

# %%
ds_y = cesm2_z3.sel(year=slice(1979,2018)).squeeze()

cesm2_z3rm = linregress(ds_y-ds_y.mean('member_id'),ds_y.year)
cesm2_z3rm

# %%
cesm2_z3r_fast = cesm2_z3r.sel(member_id=green_fast.member_id.values)
cesm2_z3r_fast

# %%
ds_y = cesm2_z3.sel(year=slice(1979,2018)).sel(member_id=green_fast.member_id.values).mean('member_id').squeeze()

cesm2_z3r_fastm = linregress(ds_y,ds_y.year)
cesm2_z3r_fastm

# %%
cesm2_z3r_new = cesm2_z3r.sel(member_id=rtmq_ex.member_id.values)
cesm2_z3r_new

# %% [markdown]
# ## IVT

# %%
def reader(member_id):
    member_idn = member_id[:-2]
    
    files = sorted(glob.glob(f'/raid/scratch/scratch-itbaxter/exp/CESM2-LE/ivt/cesm2-le.*u*t_jja.{member_idn}*.nc'))
    uivth = xr.open_dataset(files[0]).sel(year=slice(1981,2014)).drop_duplicates('year')
    uivts = xr.open_dataset(files[1]).sel(year=slice(2015,2022))

    uivt = xr.concat([uivth,uivts],dim='year')
    return uivt

uivt = [reader(member_id) for member_id in cesm2_z3r.member_id.values]
uivt = xr.concat(uivt,dim='member_id')
uivt.coords['member_id'] = cesm2_z3r.member_id.values
uivt 

# %%
def reader(member_id):
    member_idn = member_id[:-2]
    
    files = sorted(glob.glob(f'/raid/scratch/scratch-itbaxter/exp/CESM2-LE/ivt/cesm2-le.*.vivt_jja.{member_idn}*.nc'))
    uivth = xr.open_dataset(files[0]).sel(year=slice(1981,2014)).drop_duplicates('year')
    uivts = xr.open_dataset(files[1]).sel(year=slice(2015,2022))
    
    uivt = xr.concat([uivth,uivts],dim='year')
    return uivt

vivt = [reader(member_id) for member_id in cesm2_z3r.member_id.values]
vivt = xr.concat(vivt,dim='member_id')
vivt.coords['member_id'] = cesm2_z3r.member_id.values
vivt 

# %%
ruivt = linregress(uivt['uIVT'].sel(year=slice(1979,2018),member_id=green_fast.member_id.values).mean('member_id'),uivt.sel(year=slice(1979,2018)).year)

# %%
ruivt40 = linregress(uivt['uIVT'].sel(year=slice(1979,2018)).mean('member_id'),uivt.sel(year=slice(1979,2018)).year)

# %%
rvivt = linregress(vivt['vIVT'].sel(year=slice(1979,2018),member_id=green_fast.member_id.values).mean('member_id'),vivt.sel(year=slice(1979,2018)).year)

# %%
rvivt40 = linregress(vivt['vIVT'].sel(year=slice(1979,2018)).mean('member_id'),vivt.sel(year=slice(1979,2018)).year)

# %%

# %% [markdown]
# ## FINAL PLOT

# %%
# Create a custom RdYlBu colormap with white in the middle
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 7
mpl.rcParams['hatch.color'] = 'silver'

fig = plt.figure(figsize=(7.5,6))
## Panel 1 --------------------------------------------------
ax = fig.add_subplot(221)
ax.set_xlim([1978,2023])
#ax.set_ylim([100,1000])
ax.set_ylabel(r'JJA AR frequency [%]')
ax.set_xlabel('year')
for axis in ['bottom','left']:
    ax.spines[axis].set_linewidth(1.25)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.4))
    
ax.text(-0.08, 0.97, 'a', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

fills = ax.fill_between(np.arange(1979,2019,step=1),100*index_cesm2.min('member_id'),100*index_cesm2.max('member_id'),color='silver',alpha=0.4)

ln0 = ax.plot(np.arange(1979,2019,step=1),100*index_cesm2.sel(member_id=green_fast['member_id']).mean('member_id').sel(year=slice(1979,2018)).values,
              c='#6699CC',linestyle='--',
              linewidth=1.5,label=r'CESM2-LE >1.5*$\mathrm{\sigma}$')

ln1 = ax.plot(np.arange(1979,2019,step=1),100*index_cesm2.mean('member_id').sel(year=slice(1979,2018)).values,
              c='#6699CC',linewidth=1.5,label='CESM2-LE')
ln2 = ax.plot(np.arange(1979,2023,step=1),100*index_freq.sel(year=slice(1979,2022)).values,c='k',linewidth=1.5,label='iCESM1')
ln3 = ax.plot(np.arange(1979,2020,step=1),100*index_era5.sel(year=slice(1979,2019)).values,c='#AA4499',linewidth=1.5,label='ERA5')

plt.legend(frameon=False,loc='upper left')

# Panel 2--------------------------------------------------
ax = fig.add_subplot(222,projection=ccrs.NorthPolarStereo())
#ax = fig.add_axes([0,0.35,0.5,0.35],ccrs.NorthPolarStereo())
ax.set_extent([-180,180,40,90],ccrs.PlateCarree())
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax.set_title('iCESM1')

ax.text(-0.08, 0.97, 'b', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

colors1 = plt.cm.YlOrRd(np.linspace(0, 1, 128))
colors2 = plt.cm.Blues(np.linspace(0, 1, 128))[::-1]
new_colors = np.vstack((colors2[:], np.ones((20, 4)), colors1[:]))
new_cmap = LinearSegmentedColormap.from_list('RdYlBu_r_white', new_colors, N=16)

colors1 = plt.cm.GnBu(np.linspace(0, 1, 256))
colors2 = plt.cm.YlOrBr(np.linspace(0, 1, 256))[::-1]
new_colors = np.vstack((colors2[:], np.ones((40, 4)), colors1[:]))
new_cmap = LinearSegmentedColormap.from_list('RdYlBu_r_white', new_colors, N=16)

#new_cmap = colormaps.BlueWhiteOrangeRed
#bounds = np.arange(-2.8,2.81,0.4)
#bounds = [-2.4,-2.2,-2.0,-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,0,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4]
bounds = np.arange(-2.2,2.21,step=0.2)
irs,lons = add_cyclic_point(ir.slope,ir['lon'])
p = ax.contourf(lons,ir['lat'],irs*1000,
                cmap=new_cmap,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

s = ax.contourf(ir['lon'],ir['lat'],ir.slope.where(ir.p < 0.05)*1000,
                cmap=new_cmap,
                hatches=['xxxx'],
                #alpha=0,
                extend='both',
                levels=bounds,
                #levels=[-1.0,1.0],
                #linewidths=1.25,
                #colors='silver',
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

rvs = rv.slope #.where(rt.p < 0.05)
rus = ru.slope #.where(rt.p < 0.05)
Q = ax.quiver(rv['lon'], rv['lat'], rus*10, rvs*10, 
          color='red',
          edgecolor='k',
          transform=ccrs.PlateCarree(), 
          regrid_shape=20)

qk = ax.quiverkey(Q, 0.65, -0.06, 10, r'IVT: $10\ \mathrm{\frac{kg}{m\ s\ decade}}}$', 
                  labelpos='E',
                   #coordinates='figure'
                 )

# Colorbar
#cax = fig.add_axes([0.2,0.07,0.6,0.01])
cax = fig.add_axes([0.31,0.055,0.385,0.01])
cb = plt.colorbar(p, cax=cax,orientation='horizontal', 
                  drawedges=True,
                  #ticks=bounds[::2]
                 )
#cb.set_ticklabels(bounds[::2])
cb.set_label(r'JJA AR frequency [% $\mathrm{decade^{-1}}$] ')
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
gl.top_labels = False  # Turn off top labels
gl.left_labels = False  # Turn off left labels
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

# W Greenland
ax.add_patch(mpatches.Rectangle(xy=[-90, 60], width=40, height=25,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )


# northern Europe
ax.add_patch(mpatches.Rectangle(xy=[0, 50], width=20, height=20,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )

# central Europe
ax.add_patch(mpatches.Rectangle(xy=[60, 50], width=60, height=30,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )

# Siberia
ax.add_patch(mpatches.Rectangle(xy=[140, 60], width=30, height=10,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )

# Panel 3--------------------------------------------------
ax = fig.add_subplot(223,projection=ccrs.NorthPolarStereo())
#ax = fig.add_axes([0,0.35,0.5,0.35],ccrs.NorthPolarStereo())
ax.set_extent([-180,180,40,90],ccrs.PlateCarree())
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax.set_title('CESM2-LE mean')

ax.text(-0.08, 0.97, 'c', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

#cmap = colormaps.BlueWhiteOrangeRed
#bounds = np.arange(-2.4,2.41,0.4)
#bounds = [-2.6,-2.5,-2.4,-2.3,-2.2,-2.1,-2.0,0,2.0,2.1,2.2,2.3,2.4,2.5,2.6]
c2rs,lons = add_cyclic_point(c2r.slope.where(c2r.p < 0.05),c2r['lon'])
p = ax.contourf(lons,c2r['lat'],c2rs*1000,
                cmap=new_cmap,
                #hatches=['//'],
                #alpha=0,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

"""
s = ax.contour(c2r['lon'],c2r['lat'],c2r.slope.where(c2r.p > 0.05)*1000,
                cmap=new_cmap,
                hatches=['xxxx'],
                alpha=0,
                extend='both',
                transform=ccrs.PlateCarree(),
                levels=bounds,
                )
                

lats,_ = xr.broadcast(rz3['lat'],rz3.slope)
latline = ax.contour(rz3['lon'],rz3['lat'],lats,
                     levels=[70],
                     colors='tab:purple',
                     linewidths=1.0,
                     alpha=0.5,
                     #linestyle='--',
                     transform=ccrs.PlateCarree()
                    )
"""

# Geopotential Height
slope = cesm2_z3r.mean('member_id').slope*10
c = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
               levels=[-6,-4,6,10,14,16,18,20,22,24,26,30],
               linewidths=0.75,
               colors='k',
               transform=ccrs.PlateCarree()
              )

z = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
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

rvs = rvivt40.slope #.where(rt.p < 0.05)
rus = ruivt40.slope #.where(rt.p < 0.05)
Q = ax.quiver(rvs['lon'], rvs['lat'], rus*10, rvs*10, 
          color='red',
          edgecolor='k',
          transform=ccrs.PlateCarree(), 
          regrid_shape=20)

qk = ax.quiverkey(Q, 0.65, -0.06, 10, r'IVT: $10\ \mathrm{\frac{kg}{m\ s\ decade}}}$', 
                  labelpos='E',
                   #coordinates='figure'
                 )

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
gl.top_labels = False  # Turn off top labels
gl.left_labels = False  # Turn off left labels
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

# Panel 4--------------------------------------------------
ax = fig.add_subplot(224,projection=ccrs.NorthPolarStereo())
#ax = fig.add_axes([0,0.35,0.5,0.35],ccrs.NorthPolarStereo())
ax.set_extent([-180,180,40,90],ccrs.PlateCarree())
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax.set_title(r'CESM2-LE >1.5*$\mathrm{\sigma}$')

ax.text(-0.08, 0.97, 'd', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

cmap = colormaps.BlueWhiteOrangeRed
#bounds = np.arange(-2.4,2.41,0.4)
#bounds = [-2.6,-2.5,-2.4,-2.3,-2.2,-2.1,-2.0,0,2.0,2.1,2.2,2.3,2.4,2.5,2.6]
#c2rs,lons = add_cyclic_point(green_fast.mean('member_id').slope,c2r['lon'])
c2rs,lons = add_cyclic_point(spreadm.slope,c2r['lon'])
p = ax.contourf(lons,c2r['lat'],c2rs*1000,
                cmap=new_cmap,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

"""
c2rs,lons = add_cyclic_point(spreadm.slope.where(spreadm.p > 0.05),c2r['lon'])
s = ax.contourf(lons,spreadm['lat'],c2rs*1000,
                cmap=cmap,
                hatches=['xxxx'],
                alpha=0,
                extend='both',
                transform=ccrs.PlateCarree(),
                levels=bounds,
                )

lats,_ = xr.broadcast(rz3['lat'],rz3.slope)
latline = ax.contour(rz3['lon'],rz3['lat'],lats,
                     levels=[70],
                     colors='tab:purple',
                     linewidths=1.0,
                     alpha=0.5,
                     #linestyle='--',
                     transform=ccrs.PlateCarree()
                    )
"""

# Geopotential Height
slope = (cesm2_z3r_fastm.slope-cesm2_z3r.mean('member_id').slope)*10
c = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
               levels=[-8,-6,-4,4,6,8,10,14,18,22,26,30],
               linewidths=0.75,
               colors='k',
               transform=ccrs.PlateCarree()
              )

z = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
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


rvs = rvivt.slope #.where(rvivt.p < 0.05)
rus = ruivt.slope #.where(ruivt.p < 0.05)
Q = ax.quiver(rvs['lon'], rvs['lat'], rus*10, rvs*10, 
          color='red',
          edgecolor='k',
          transform=ccrs.PlateCarree(), 
          regrid_shape=20)

qk = ax.quiverkey(Q, 0.65, -0.06, 10, r'IVT: $10\ \mathrm{\frac{kg}{m\ s\ decade}}}$', 
                  labelpos='E',
                   #coordinates='figure'
                 )

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
gl.top_labels = False  # Turn off top labels
gl.left_labels = False  # Turn off left labels
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

x, y = [-120, -120, -60, -60, -120], [60, 85, 85, 60, 60]
#ln1 = ax.plot(x, y, c='tab:red', linewidth=1.5, linestyle='--', transform=ccrs.PlateCarree(),zorder=101)
#map_proj = ccrs.Orthographic(central_latitude=60.0, central_longitude=-90.0)
ax.add_patch(mpatches.Rectangle(xy=[-90, 60], width=40, height=25,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )


fig.subplots_adjust(top=0.96,bottom=0.12,left=0.07,right=0.93,hspace=0.3,wspace=0.1)
plt.savefig('./plots/moisture_tagging-figure_3-40mem_fast-ar.1979-2018_blue.png',dpi=600)

# %%
# Create a custom RdYlBu colormap with white in the middle
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 7
mpl.rcParams['hatch.color'] = 'silver'

fig = plt.figure(figsize=(7.5,6))
## Panel 1 --------------------------------------------------
ax = fig.add_subplot(221)
ax.set_xlim([1978,2023])
#ax.set_ylim([100,1000])
ax.set_ylabel(r'JJA AR frequency [%]')
ax.set_xlabel('year')
for axis in ['bottom','left']:
    ax.spines[axis].set_linewidth(1.25)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.4))
    
ax.text(-0.08, 0.97, 'a', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

fills = ax.fill_between(np.arange(1979,2019,step=1),100*index_cesm2.min('member_id'),100*index_cesm2.max('member_id'),color='silver',alpha=0.4)

ln0 = ax.plot(np.arange(1979,2019,step=1),100*index_cesm2.sel(member_id=green_fast['member_id']).mean('member_id').sel(year=slice(1979,2018)).values,
              c='#6699CC',linestyle='--',
              linewidth=1.5,label=r'CESM2-LE >1.5*$\mathrm{\sigma}$')

ln1 = ax.plot(np.arange(1979,2019,step=1),100*index_cesm2.mean('member_id').sel(year=slice(1979,2018)).values,
              c='#6699CC',linewidth=1.5,label='CESM2-LE')
ln2 = ax.plot(np.arange(1979,2023,step=1),100*index_freq.sel(year=slice(1979,2022)).values,c='k',linewidth=1.5,label='iCESM1')
ln3 = ax.plot(np.arange(1979,2020,step=1),100*index_era5.sel(year=slice(1979,2019)).values,c='#AA4499',linewidth=1.5,label='ERA5')

plt.legend(frameon=False,loc='upper left')

# Panel 2--------------------------------------------------
ax = fig.add_subplot(222,projection=ccrs.NorthPolarStereo())
#ax = fig.add_axes([0,0.35,0.5,0.35],ccrs.NorthPolarStereo())
ax.set_extent([-180,180,40,90],ccrs.PlateCarree())
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax.set_title('iCESM1')

ax.text(-0.08, 0.97, 'b', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

colors1 = plt.cm.YlOrRd(np.linspace(0, 1, 128))
colors2 = plt.cm.Blues(np.linspace(0, 1, 128))[::-1]
new_colors = np.vstack((colors2[:], np.ones((40, 4)), colors1[:]))
new_cmap = LinearSegmentedColormap.from_list('RdYlBu_r_white', new_colors, N=16)

colors1 = plt.cm.GnBu(np.linspace(0, 1, 256))
colors2 = plt.cm.YlOrBr(np.linspace(0, 1, 256))[::-1]
new_colors = np.vstack((colors2[:], np.ones((40, 4)), colors1[:]))
new_cmap = LinearSegmentedColormap.from_list('RdYlBu_r_white', new_colors, N=16)

bounds = np.arange(-2.2,2.21,step=0.2)
irs,lons = add_cyclic_point(ir.slope.where(abs(ir.slope) > 0.0006),ir['lon'])
p = ax.contourf(lons,ir['lat'],irs*1000,
                cmap=new_cmap,
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

rvs = rv.slope #.where(rt.p < 0.05)
rus = ru.slope #.where(rt.p < 0.05)
Q = ax.quiver(rv['lon'], rv['lat'], rus*10, rvs*10, 
          color='red',
          edgecolor='k',
          transform=ccrs.PlateCarree(), 
          regrid_shape=20)

qk = ax.quiverkey(Q, 0.65, -0.06, 10, r'IVT: $10\ \mathrm{\frac{kg}{m\ s\ decade}}}$', 
                  labelpos='E',
                   #coordinates='figure'
                 )

# Colorbar
cax = fig.add_axes([0.25,0.055,0.5,0.015])
cb = plt.colorbar(p, cax=cax,orientation='horizontal', 
                  drawedges=True,
                 )
cb.set_label(r'JJA AR frequency [% $\mathrm{decade^{-1}}$] ')
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
gl.top_labels = False  # Turn off top labels
gl.left_labels = False  # Turn off left labels
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

# W Greenland
ax.add_patch(mpatches.Rectangle(xy=[-90, 60], width=40, height=25,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )


# northern Europe
ax.add_patch(mpatches.Rectangle(xy=[0, 50], width=20, height=20,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )

# central Europe
ax.add_patch(mpatches.Rectangle(xy=[75, 50], width=50, height=30,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )

# Siberia
ax.add_patch(mpatches.Rectangle(xy=[140, 60], width=30, height=10,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )

# Panel 3--------------------------------------------------
ax = fig.add_subplot(223,projection=ccrs.NorthPolarStereo())
#ax = fig.add_axes([0,0.35,0.5,0.35],ccrs.NorthPolarStereo())
ax.set_extent([-180,180,40,90],ccrs.PlateCarree())
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax.set_title('CESM2-LE mean')

ax.text(-0.08, 0.97, 'c', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

c2rs,lons = add_cyclic_point(c2r.slope.where(c2r.p < 0.05),c2r['lon'])
p = ax.contourf(lons,c2r['lat'],c2rs*1000,
                cmap=new_cmap,
                #hatches=['//'],
                #alpha=0,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

# Geopotential Height
slope = cesm2_z3r.mean('member_id').slope*10
c = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
               levels=[-6,-4,6,10,14,16,18,20,22,24,26,30],
               linewidths=0.75,
               colors='k',
               transform=ccrs.PlateCarree()
              )

z = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
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

rvs = rvivt40.slope #.where(rt.p < 0.05)
rus = ruivt40.slope #.where(rt.p < 0.05)
Q = ax.quiver(rvs['lon'], rvs['lat'], rus*10, rvs*10, 
          color='red',
          edgecolor='k',
          transform=ccrs.PlateCarree(), 
          regrid_shape=20)

qk = ax.quiverkey(Q, 0.65, -0.06, 10, r'IVT: $10\ \mathrm{\frac{kg}{m\ s\ decade}}}$', 
                  labelpos='E',
                   #coordinates='figure'
                 )

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
gl.top_labels = False  # Turn off top labels
gl.left_labels = False  # Turn off left labels
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

# Panel 4--------------------------------------------------
ax = fig.add_subplot(224,projection=ccrs.NorthPolarStereo())
#ax = fig.add_axes([0,0.35,0.5,0.35],ccrs.NorthPolarStereo())
ax.set_extent([-180,180,40,90],ccrs.PlateCarree())
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax.set_title(r'CESM2-LE >1.5*$\mathrm{\sigma}$')

ax.text(-0.08, 0.97, 'd', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

cmap = colormaps.BlueWhiteOrangeRed
#bounds = np.arange(-2.4,2.41,0.4)
#bounds = [-2.6,-2.5,-2.4,-2.3,-2.2,-2.1,-2.0,0,2.0,2.1,2.2,2.3,2.4,2.5,2.6]
#c2rs,lons = add_cyclic_point(green_fast.mean('member_id').slope,c2r['lon'])
c2rs,lons = add_cyclic_point(spreadm.slope,c2r['lon'])
p = ax.contourf(lons,c2r['lat'],c2rs*1000,
                cmap=new_cmap,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

"""
c2rs,lons = add_cyclic_point(spreadm.slope.where(spreadm.p > 0.05),c2r['lon'])
s = ax.contourf(lons,spreadm['lat'],c2rs*1000,
                cmap=cmap,
                hatches=['xxxx'],
                alpha=0,
                extend='both',
                transform=ccrs.PlateCarree(),
                levels=bounds,
                )

lats,_ = xr.broadcast(rz3['lat'],rz3.slope)
latline = ax.contour(rz3['lon'],rz3['lat'],lats,
                     levels=[70],
                     colors='tab:purple',
                     linewidths=1.0,
                     alpha=0.5,
                     #linestyle='--',
                     transform=ccrs.PlateCarree()
                    )
"""

# Geopotential Height
slope = (cesm2_z3r_fastm.slope-cesm2_z3r.mean('member_id').slope)*10
c = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
               levels=[-8,-6,-4,4,6,8,10,14,18,22,26,30],
               linewidths=0.75,
               colors='k',
               transform=ccrs.PlateCarree()
              )

z = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
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


rvs = rvivt.slope #.where(rvivt.p < 0.05)
rus = ruivt.slope #.where(ruivt.p < 0.05)
Q = ax.quiver(rvs['lon'], rvs['lat'], rus*10, rvs*10, 
          color='red',
          edgecolor='k',
          transform=ccrs.PlateCarree(), 
          regrid_shape=20)

qk = ax.quiverkey(Q, 0.65, -0.06, 10, r'IVT: $10\ \mathrm{\frac{kg}{m\ s\ decade}}}$', 
                  labelpos='E',
                   #coordinates='figure'
                 )

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
gl.top_labels = False  # Turn off top labels
gl.left_labels = False  # Turn off left labels
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

x, y = [-120, -120, -60, -60, -120], [60, 85, 85, 60, 60]
#ln1 = ax.plot(x, y, c='tab:red', linewidth=1.5, linestyle='--', transform=ccrs.PlateCarree(),zorder=101)
#map_proj = ccrs.Orthographic(central_latitude=60.0, central_longitude=-90.0)
ax.add_patch(mpatches.Rectangle(xy=[-90, 60], width=40, height=25,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )


fig.subplots_adjust(top=0.96,bottom=0.12,left=0.07,right=0.93,hspace=0.3,wspace=0.1)
plt.savefig('./plots/moisture_tagging-figure_3-40mem_fast-ar.1979-2018_blue-clean.png',dpi=600)

# %% [markdown]
# ## 3 panels

# %%
fig = plt.figure(figsize=(7.5,6))

gs = fig.add_gridspec(2,2)
ax2 = fig.add_subplot(gs[0, 0],projection=ccrs.NorthPolarStereo())
ax3 = fig.add_subplot(gs[0, 1],projection=ccrs.NorthPolarStereo())
ax1 = fig.add_subplot(gs[1, :])

# Create a custom RdYlBu colormap with white in the middle
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 7
mpl.rcParams['hatch.color'] = 'silver'

# Panel 1-------------------------------------------------
ax2.set_extent([-180,180,40,90],ccrs.PlateCarree())
ax2.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax2.set_title('iCESM1')

ax2.text(-0.08, 0.97, 'a', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax2.transAxes)

colors1 = plt.cm.YlOrRd(np.linspace(0, 1, 128))
colors2 = plt.cm.Blues(np.linspace(0, 1, 128))[::-1]
new_colors = np.vstack((colors2[:], np.ones((20, 4)), colors1[:]))
new_cmap = LinearSegmentedColormap.from_list('RdYlBu_r_white', new_colors, N=16)

colors1 = plt.cm.GnBu(np.linspace(0, 1, 256))
colors2 = plt.cm.YlOrBr(np.linspace(0, 1, 256))[::-1]
new_colors = np.vstack((colors2[:], np.ones((40, 4)), colors1[:]))
new_cmap = LinearSegmentedColormap.from_list('RdYlBu_r_white', new_colors, N=16)

bounds = np.arange(-2.2,2.21,step=0.2)
irs,lons = add_cyclic_point(ir.slope,ir['lon'])
p = ax2.contourf(lons,ir['lat'],irs*1000,
                cmap=new_cmap,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

s = ax2.contourf(ir['lon'],ir['lat'],ir.slope.where(ir.p < 0.05)*1000,
                cmap=new_cmap,
                hatches=['xxxx'],
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

lats,_ = xr.broadcast(rz3['lat'],rz3.slope)
latline = ax2.contour(rz3['lon'],rz3['lat'],lats,
                     levels=[70],
                     colors='tab:purple',
                     linewidths=1.0,
                     alpha=0.5,
                     #linestyle='--',
                     transform=ccrs.PlateCarree()
                    )

# Geopotential Height
c = ax2.contour(rz3['lon'],rz3['lat'],rz3.slope*10,
               levels=[-6,-4,6,10,14,18],
               linewidths=0.75,
               colors='k',
               transform=ccrs.PlateCarree()
              )

z = ax2.contour(rz3['lon'],rz3['lat'],rz3.slope*10,
               levels=[0],
               linewidths=1.5,
               colors='k',
               transform=ccrs.PlateCarree()
              )


def fmt(x):
    return f'{x:2.0f}'

clab1 = ax2.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=7)
clab2 = ax2.clabel(z, z.levels, inline=True, fmt=fmt, fontsize=7)
for l in clab1+clab2:
    l.set_rotation(0)

rvs = rv.slope #.where(rt.p < 0.05)
rus = ru.slope #.where(rt.p < 0.05)
Q = ax2.quiver(rv['lon'], rv['lat'], rus*10, rvs*10, 
          color='red',
          edgecolor='k',
          transform=ccrs.PlateCarree(), 
          regrid_shape=20)

qk = ax2.quiverkey(Q, 0.65, -0.06, 10, r'IVT: $10\ \mathrm{\frac{kg}{m\ s\ decade}}}$', 
                  labelpos='E',
                   #coordinates='figure'
                 )

# Colorbar
cax = fig.add_axes([0.25,0.52,0.5,0.015])
cb = plt.colorbar(p, cax=cax,orientation='horizontal', 
                  drawedges=True,
                  #ticks=bounds[::2]
                 )
cb.set_label(r'JJA AR frequency [% $\mathrm{decade^{-1}}$] ')
cb.outline.set_color('k')
cb.outline.set_linewidth(0.8)
cb.dividers.set_color('k')
cb.dividers.set_linewidth(0.8)
cb.ax.tick_params(size=0)

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

ax2.set_boundary(circle, transform=ax2.transAxes)
ax2.spines['geo'].set_linewidth(1.25)

# Add gridlines
gl = ax2.gridlines(draw_labels=True, linewidth=0.35,  
                  x_inline=False,
                  y_inline=False,
                  rotate_labels=False, color='k', 
                  linestyle='--', alpha=0.4)
gl.top_labels = False  # Turn off top labels
gl.left_labels = False  # Turn off left labels
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

# W Greenland
ax2.add_patch(mpatches.Rectangle(xy=[-90, 60], width=40, height=25,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )


# northern Europe
ax2.add_patch(mpatches.Rectangle(xy=[0, 50], width=20, height=20,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )

# central Europe
ax2.add_patch(mpatches.Rectangle(xy=[60, 50], width=60, height=30,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )

# Siberia

s = ax2.contour(c2r['lon'],c2r['lat'],c2r.slope.where(c2r.p > 0.05)*1000,
                cmap=new_cmap,
                hatches=['xxxx'],
                alpha=0,
                extend='both',
                transform=ccrs.PlateCarree(),
                levels=bounds,
                )
                

lats,_ = xr.broadcast(rz3['lat'],rz3.slope)
latline = ax2.contour(rz3['lon'],rz3['lat'],lats,
                     levels=[70],
                     colors='tab:purple',
                     linewidths=1.0,
                     alpha=0.5,
                     #linestyle='--',
                     transform=ccrs.PlateCarree()
                    )

# Panel 3--------------------------------------------------
ax3.set_extent([-180,180,40,90],ccrs.PlateCarree())
ax3.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax3.set_title('CESM2-LE mean')

ax3.text(-0.08, 0.97, 'b', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax3.transAxes)

c2rs,lons = add_cyclic_point(c2r.slope.where(c2r.p < 0.05),c2r['lon'])
p = ax3.contourf(lons,c2r['lat'],c2rs*1000,
                cmap=new_cmap,
                #hatches=['//'],
                #alpha=0,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

# Geopotential Height
slope = cesm2_z3r.mean('member_id').slope*10
c = ax3.contour(cesm2_z3r['lon'],cesm2_z3r['lat'].sel(lat=slice(45,90)),slope.sel(lat=slice(45,90)),
               levels=[-6,-4,6,10,14,16,18,20,22,24,26,30],
               linewidths=0.75,
               colors='k',
               transform=ccrs.PlateCarree()
              )

z = ax3.contour(cesm2_z3r['lon'],cesm2_z3r['lat'].sel(lat=slice(45,90)),slope.sel(lat=slice(45,90)),
               levels=[0],
               linewidths=1.5,
               colors='k',
               transform=ccrs.PlateCarree()
              )

def fmt(x):
    return f'{x:2.0f}'

clab1 = ax3.clabel(c, c.levels, inline=True, fmt=fmt, fontsize=7)
clab2 = ax3.clabel(z, z.levels, inline=True, fmt=fmt, fontsize=7)
for l in clab1+clab2:
    l.set_rotation(0)

rvs = rvivt40.slope #.where(rt.p < 0.05)
rus = ruivt40.slope #.where(rt.p < 0.05)
Q = ax3.quiver(rvs['lon'], rvs['lat'], rus*10, rvs*10, 
          color='red',
          edgecolor='k',
          transform=ccrs.PlateCarree(), 
          regrid_shape=20)

qk = ax3.quiverkey(Q, 0.65, -0.06, 10, r'IVT: $10\ \mathrm{\frac{kg}{m\ s\ decade}}}$', 
                  labelpos='E',
                   #coordinates='figure'
                 )

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

ax3.set_boundary(circle, transform=ax3.transAxes)
ax3.spines['geo'].set_linewidth(1.25)

# Add gridlines
gl = ax3.gridlines(draw_labels=True, linewidth=0.35,  
                  x_inline=False,
                  y_inline=False,
                  rotate_labels=False, color='k', 
                  linestyle='--', alpha=0.4)
gl.top_labels = False  # Turn off top labels
gl.left_labels = False  # Turn off left labels
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

## Panel 3 --------------------------------------------------
#ax1 = fig.add_axes([0.2,0.07,0.65,0.37])
ax1.set_xlim([1978,2023])
#ax.set_ylim([100,1000])
ax1.set_ylabel(r'JJA AR frequency [%]')
ax1.set_xlabel('year')
for axis in ['bottom','left']:
    ax1.spines[axis].set_linewidth(1.25)
ax1.spines[['right', 'top']].set_visible(False)
ax1.xaxis.set_minor_locator(MultipleLocator(1))
ax1.yaxis.set_minor_locator(MultipleLocator(0.4))
    
ax1.text(-0.08, 0.97, 'c', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax1.transAxes)

fills = ax1.fill_between(np.arange(1979,2019,step=1),100*index_cesm2.min('member_id'),100*index_cesm2.max('member_id'),color='silver',alpha=0.4)

ln1 = ax1.plot(np.arange(1979,2019,step=1),100*index_cesm2.mean('member_id').sel(year=slice(1979,2018)).values,
              c='#6699CC',linewidth=1.5,label='CESM2-LE')
ln2 = ax1.plot(np.arange(1979,2023,step=1),100*index_freq.sel(year=slice(1979,2022)).values,c='k',linewidth=1.5,label='iCESM1')
ln3 = ax1.plot(np.arange(1979,2020,step=1),100*index_era5.sel(year=slice(1979,2019)).values,c='#AA4499',linewidth=1.5,label='ERA5')

ax1.legend(frameon=False,loc='upper left')

#fig.subplots_adjust(left=0.2,right=0.8)


fig.subplots_adjust(top=0.96,bottom=0.07,left=0.07,right=0.93,hspace=0.35,wspace=0.25)
plt.savefig('./plots/moisture_tagging-figure_3-ar.1979-2018_blue-3panel_main-flipped.png',dpi=600)

# %% [markdown]
# ## Return to previous

# %%
def reader(member_id):
    member_idn = member_id[:-2]
    
    files = sorted(glob.glob(f'/raid/scratch/scratch-itbaxter/exp/CESM2-LE/ivt/cesm2-le.*u*t_jja.{member_idn}*.nc'))
    print(files)
    uivth = xr.open_dataset(files[0]).sel(year=slice(1981,2014))
    uivts = xr.open_dataset(files[1]).sel(year=slice(2015,2022))

    uivt = xr.concat([uivth,uivts],dim='year')
    return uivt.drop_duplicates('year')

uivt2 = [reader(member_id) for member_id in cesm2_z3r_new.member_id.values]
uivt2 = xr.concat(uivt2,dim='member_id')
uivt2.coords['member_id'] = cesm2_z3r_new.member_id.values

def reader(member_id):
    member_idn = member_id[:-2]
    
    files = sorted(glob.glob(f'/raid/scratch/scratch-itbaxter/exp/CESM2-LE/ivt/cesm2-le.*.vivt_jja.{member_idn}*.nc'))
    print(files)
    uivth = xr.open_dataset(files[0]).sel(year=slice(1981,2014))
    uivts = xr.open_dataset(files[1]).sel(year=slice(2015,2022))

    uivt = xr.concat([uivth,uivts],dim='year')
    return uivt.drop_duplicates('year')

vivt2 = [reader(member_id) for member_id in cesm2_z3r_new.member_id.values]
vivt2 = xr.concat(vivt2,dim='member_id')
vivt2.coords['member_id'] = cesm2_z3r_new.member_id.values

ruivt2 = linregress(uivt2['uIVT'].sel(year=slice(1981,2022)),uivt.sel(year=slice(1981,2022)).year)
rvivt2 = linregress(vivt2['vIVT'].sel(year=slice(1981,2022)),vivt.sel(year=slice(1981,2022)).year)

# %%
fig = plt.figure(figsize=(7.5,6))

# Panel 1--------------------------------------------------
# Create a custom RdYlBu colormap with white in the middle
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 7
mpl.rcParams['hatch.color'] = 'silver'

## Panel 1 --------------------------------------------------
ax = fig.add_subplot(221)
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

cesm2_ex = cesm2_tmq.sel(member_id=rtmq_ex['member_id'])
for member_id in cesm2_ex.member_id:
    ln1 = ax.plot(np.arange(1979,2023,step=1),cesm2_ex.sel(member_id=member_id),
                  c='#6699CC',linewidth=1.5,label=member_id.values,zorder=10)

leg = plt.legend(frameon=False,fontsize=7,loc='lower left',
           title=r'Based on TMQ >1$\mathrm{\sigma}}$',
           bbox_to_anchor=[0.03, 0.65])
ax.add_artist(leg)

lns = []
cesm2_ex = cesm2_tmq.sel(member_id=cesm2_z3r_fast['member_id'])

ln2, = ax.plot(np.arange(1979,2023,step=1),cesm2_ex.isel(member_id=0),
              c='#D62728',linewidth=1.5,label=cesm2_ex['member_id'][0].values,zorder=10)
ln3, = ax.plot(np.arange(1979,2023,step=1),cesm2_ex.isel(member_id=1),
              c='#D62728',linewidth=1.5,label=cesm2_ex['member_id'][1].values,zorder=10)

ln1 = ax.plot(np.arange(1979,2023,step=1),cesm2_tmq.mean('member_id').values,
              c='k',linewidth=1.5,label='CESM2-LE',zorder=10)

plt.legend(handles=[ln2,ln3],frameon=False,fontsize=7,loc='lower left',
           title=r'Based on AR freq >1.5$\mathrm{\sigma}}$',
           bbox_to_anchor=[0.6, 0.01])

    
# Panel 4--------------------------------------------------
ax = fig.add_subplot(222,projection=ccrs.NorthPolarStereo())
ax.set_extent([-180,180,40,90],ccrs.PlateCarree())
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax.set_title(r'CESM2-LE: TMQ >1.0*$\mathrm{\sigma}$')

ax.text(-0.08, 0.97, 'b', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

cmap = colormaps.BlueWhiteOrangeRed
bounds = np.arange(-2.2,2.21,step=0.2)
c2rs,lons = add_cyclic_point(new_fastm.slope.where(new_fastm.p <0.05),c2r['lon'])
p = ax.contourf(lons,c2r['lat'],c2rs*1000,
                cmap=new_cmap,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

# Geopotential Height
#slope = cesm2_z3r_fast.mean('member_id').slope*10
slope = (cesm2_z3r_new.mean('member_id').slope-cesm2_z3r.mean('member_id').slope)*10
c = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
               #levels=[-6,-4,6,10,14,18,22,26,30],
               linewidths=0.75,
               colors='k',
               transform=ccrs.PlateCarree()
              )

z = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
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

rvs = rvivt2.slope.mean('member_id') #.where(rt.p < 0.05)
rus = ruivt2.slope.mean('member_id') #.where(rt.p < 0.05)
Q = ax.quiver(rvs['lon'], rvs['lat'], rus*10, rvs*10, 
          color='red',
          edgecolor='k',
          transform=ccrs.PlateCarree(), 
          regrid_shape=20)

qk = ax.quiverkey(Q, 0.66, 0.03, 10, r'IVT: $10\ \mathrm{\frac{kg}{m\ s\ decade}}}$', 
                  labelpos='E',
                   #coordinates='figure'
                 )

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
gl.top_labels = False  # Turn off top labels
gl.left_labels = False  # Turn off left labels
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

# Colorbar
#cax = fig.add_axes([0.2,0.07,0.6,0.01])
cax = fig.add_axes([0.25,0.055,0.5,0.015])
cb = plt.colorbar(p, cax=cax,orientation='horizontal', 
                  drawedges=True,
                  #ticks=[-2.4,-1.8,-1.2,-0.6,0.6,1.2,1.8,2.4],
                 )
#cb.set_ticklabels(bounds[::2])
cb.set_label(r'JJA AR frequency [% $\mathrm{decade^{-1}}$] ')
cb.outline.set_color('k')
cb.outline.set_linewidth(0.8)
cb.dividers.set_color('k')
cb.dividers.set_linewidth(0.8)
cb.ax.tick_params(size=0)

# W Greenland
ax.add_patch(mpatches.Rectangle(xy=[-90, 60], width=40, height=25,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )


# northern Europe
ax.add_patch(mpatches.Rectangle(xy=[0, 50], width=20, height=20,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )

# central Europe
ax.add_patch(mpatches.Rectangle(xy=[60, 50], width=60, height=30,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )

# Siberia
ax.add_patch(mpatches.Rectangle(xy=[140, 60], width=30, height=10,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )

# Panel 3--------------------------------------------------
ax = fig.add_subplot(223,projection=ccrs.NorthPolarStereo())
#ax = fig.add_axes([0,0.35,0.5,0.35],ccrs.NorthPolarStereo())
ax.set_extent([-180,180,40,90],ccrs.PlateCarree())
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax.set_title('CESM2-LE mean')

ax.text(-0.08, 0.97, 'c', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

c2rs,lons = add_cyclic_point(c2r.slope.where(c2r.p < 0.05),c2r['lon'])
p = ax.contourf(lons,c2r['lat'],c2rs*1000,
                cmap=new_cmap,
                #hatches=['//'],
                #alpha=0,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

# Geopotential Height
slope = cesm2_z3r.mean('member_id').slope*10
c = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
               levels=[-6,-4,6,10,14,16,18,20,22,24,26,30],
               linewidths=0.75,
               colors='k',
               transform=ccrs.PlateCarree()
              )

z = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
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

rvs = rvivt40.slope #.where(rt.p < 0.05)
rus = ruivt40.slope #.where(rt.p < 0.05)
Q = ax.quiver(rvs['lon'], rvs['lat'], rus*10, rvs*10, 
          color='red',
          edgecolor='k',
          transform=ccrs.PlateCarree(), 
          regrid_shape=20)

qk = ax.quiverkey(Q, 0.65, -0.06, 10, r'IVT: $10\ \mathrm{\frac{kg}{m\ s\ decade}}}$', 
                  labelpos='E',
                 )

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
gl.top_labels = False  # Turn off top labels
gl.left_labels = False  # Turn off left labels
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

# Panel 4--------------------------------------------------
ax = fig.add_subplot(224,projection=ccrs.NorthPolarStereo())
#ax = fig.add_axes([0,0.35,0.5,0.35],ccrs.NorthPolarStereo())
ax.set_extent([-180,180,40,90],ccrs.PlateCarree())
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax.set_title(r'CESM2-LE: $\mathrm{AR_{GL}}$ >1.5*$\mathrm{\sigma}$')

ax.text(-0.08, 0.97, 'd', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

cmap = colormaps.BlueWhiteOrangeRed
c2rs,lons = add_cyclic_point(spreadm.slope,c2r['lon'])
p = ax.contourf(lons,c2r['lat'],c2rs*1000,
                cmap=new_cmap,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

# Geopotential Height
slope = (cesm2_z3r_fastm.slope-cesm2_z3r.mean('member_id').slope)*10
c = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
               levels=[-8,-6,-4,4,6,8,10,14,18,22,26,30],
               linewidths=0.75,
               colors='k',
               transform=ccrs.PlateCarree()
              )

z = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
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


rvs = rvivt.slope 
rus = ruivt.slope
Q = ax.quiver(rvs['lon'], rvs['lat'], rus*10, rvs*10, 
          color='red',
          edgecolor='k',
          transform=ccrs.PlateCarree(), 
          regrid_shape=20)

qk = ax.quiverkey(Q, 0.65, -0.06, 10, r'IVT: $10\ \mathrm{\frac{kg}{m\ s\ decade}}}$', 
                  labelpos='E',
                 )

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
gl.top_labels = False  # Turn off top labels
gl.left_labels = False  # Turn off left labels
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

x, y = [-120, -120, -60, -60, -120], [60, 85, 85, 60, 60]
ax.add_patch(mpatches.Rectangle(xy=[-90, 60], width=40, height=25,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )


fig.subplots_adjust(top=0.94,bottom=0.12,left=0.07,right=0.93,hspace=0.3,wspace=0.1)
plt.savefig('./plots/moisture_tagging-figure_3-40mem_fast-supp.1979-2018_blue-new.png',dpi=600)

# %%
files = sorted(glob.glob('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/ERA5/q/*nc'))

era5_q = xr.open_mfdataset(files,combine='nested',concat_dim='time')['q']
era5_q.coords['latitude'] = icesm_q['lat'].values
era5_q.coords['longitude'] = icesm_q['lon'].values
era5_q = era5_q.rename({'latitude':'lat','longitude':'lon','level':'lev'})
era5_q

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
fig = plt.figure(figsize=(7.5,6))

# Panel 1--------------------------------------------------
# Create a custom RdYlBu colormap with white in the middle
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 7
mpl.rcParams['hatch.color'] = 'silver'

## Panel 1 --------------------------------------------------
ax = fig.add_subplot(221)
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

cesm2_ex = cesm2_tmq.sel(member_id=rtmq_ex['member_id'])
for member_id in cesm2_ex.member_id:
    ln1 = ax.plot(np.arange(1979,2023,step=1),cesm2_ex.sel(member_id=member_id),
                  c='#6699CC',linewidth=1.5,label=member_id.values,zorder=10)

leg = plt.legend(frameon=False,fontsize=7,loc='lower left',
           title=r'Based on TMQ >1$\mathrm{\sigma}}$',
           bbox_to_anchor=[0.03, 0.65])
ax.add_artist(leg)

lns = []
cesm2_ex = cesm2_tmq.sel(member_id=cesm2_z3r_fast['member_id'])

ln2, = ax.plot(np.arange(1979,2023,step=1),cesm2_ex.isel(member_id=0),
              c='#D62728',linewidth=1.5,label=cesm2_ex['member_id'][0].values,zorder=10)
ln3, = ax.plot(np.arange(1979,2023,step=1),cesm2_ex.isel(member_id=1),
              c='#D62728',linewidth=1.5,label=cesm2_ex['member_id'][1].values,zorder=10)

ln1 = ax.plot(np.arange(1979,2023,step=1),cesm2_tmq.mean('member_id').values,
              c='k',linewidth=1.5,label='CESM2-LE',zorder=10)

plt.legend(handles=[ln2,ln3],frameon=False,fontsize=7,loc='lower left',
           title=r'Based on AR freq >1.5$\mathrm{\sigma}}$',
           bbox_to_anchor=[0.6, 0.01])

    
# Panel 4--------------------------------------------------
ax = fig.add_subplot(223,projection=ccrs.NorthPolarStereo())
#ax = fig.add_axes([0,0.35,0.5,0.35],ccrs.NorthPolarStereo())
ax.set_extent([-180,180,40,90],ccrs.PlateCarree())
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax.set_title(r'CESM2-LE: TMQ >1.0*$\mathrm{\sigma}$')

ax.text(-0.08, 0.97, 'c', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

#cmap = colormaps.BlueWhiteOrangeRed
colors1 = plt.cm.GnBu(np.linspace(0, 1, 256))
colors2 = plt.cm.YlOrBr(np.linspace(0, 1, 256))[::-1]
new_colors = np.vstack((colors2[:], np.ones((40, 4)), colors1[:]))
new_cmap = LinearSegmentedColormap.from_list('RdYlBu_r_white', new_colors, N=16)
cmap = new_cmap
bounds = np.arange(-2.2,2.21,step=0.2)
c2rs,lons = add_cyclic_point(new_fastm.slope.where(new_fastm.p <0.05),c2r['lon'])
p = ax.contourf(lons,c2r['lat'],c2rs*1000,
                cmap=new_cmap,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

# Geopotential Height
slope = (cesm2_z3r_new.mean('member_id').slope-cesm2_z3r.mean('member_id').slope)*10
c = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
               #levels=[-6,-4,6,10,14,18,22,26,30],
               linewidths=0.75,
               colors='k',
               transform=ccrs.PlateCarree()
              )

z = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
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

rvs = rvivt2.slope.mean('member_id') #.where(rt.p < 0.05)
rus = ruivt2.slope.mean('member_id') #.where(rt.p < 0.05)
Q = ax.quiver(rvs['lon'], rvs['lat'], rus*10, rvs*10, 
          color='red',
          edgecolor='k',
          transform=ccrs.PlateCarree(), 
          regrid_shape=20)

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
gl.top_labels = False  # Turn off top labels
gl.left_labels = False  # Turn off left labels
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

# Colorbar
#cax = fig.add_axes([0.2,0.07,0.6,0.01])
cax = fig.add_axes([0.25,0.055,0.5,0.015])
cb = plt.colorbar(p, cax=cax,orientation='horizontal', 
                  drawedges=True,
                  #ticks=[-2.4,-1.8,-1.2,-0.6,0.6,1.2,1.8,2.4],
                 )
#cb.set_ticklabels(bounds[::2])
cb.set_label(r'JJA AR frequency [% $\mathrm{decade^{-1}}$] ')
cb.outline.set_color('k')
cb.outline.set_linewidth(0.8)
cb.dividers.set_color('k')
cb.dividers.set_linewidth(0.8)
cb.ax.tick_params(size=0)

# W Greenland
ax.add_patch(mpatches.Rectangle(xy=[-90, 60], width=40, height=25,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )


# northern Europe
ax.add_patch(mpatches.Rectangle(xy=[0, 50], width=20, height=20,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )

# central Europe
ax.add_patch(mpatches.Rectangle(xy=[60, 50], width=60, height=30,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )

# Panel 3--------------------------------------------------
ax = fig.add_subplot(222,projection=ccrs.NorthPolarStereo())
ax.set_extent([-180,180,40,90],ccrs.PlateCarree())
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax.set_title('ERA5')

ax.text(-0.08, 0.97, 'b', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

rs,lons = add_cyclic_point(r.slope,r['lon'])
p = ax.contourf(lons,r['lat'],rs*1000,
                cmap=new_cmap,
                #hatches=['//'],
                #alpha=0,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

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

#qk = ax.quiverkey(Q, 0.65, -0.06, 10, r'IVT: $10\ \mathrm{\frac{kg}{m\ s\ decade}}}$', 
#                  labelpos='E',
                   #coordinates='figure'
#                 )

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
gl.top_labels = False  # Turn off top labels
gl.left_labels = False  # Turn off left labels
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

# W Greenland
ax.add_patch(mpatches.Rectangle(xy=[-90, 60], width=40, height=25,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )


# northern Europe
ax.add_patch(mpatches.Rectangle(xy=[0, 50], width=20, height=20,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )

# central Europe
ax.add_patch(mpatches.Rectangle(xy=[60, 50], width=60, height=30,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )


# Panel 4--------------------------------------------------
ax = fig.add_subplot(224,projection=ccrs.NorthPolarStereo())
#ax = fig.add_axes([0,0.35,0.5,0.35],ccrs.NorthPolarStereo())
ax.set_extent([-180,180,40,90],ccrs.PlateCarree())
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax.set_title(r'CESM2-LE: $\mathrm{AR_{GL}}$ >1.5*$\mathrm{\sigma}$')

ax.text(-0.08, 0.97, 'd', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

cmap = colormaps.BlueWhiteOrangeRed
#bounds = np.arange(-2.4,2.41,0.4)
#bounds = [-2.6,-2.5,-2.4,-2.3,-2.2,-2.1,-2.0,0,2.0,2.1,2.2,2.3,2.4,2.5,2.6]
#c2rs,lons = add_cyclic_point(green_fast.mean('member_id').slope,c2r['lon'])
c2rs,lons = add_cyclic_point(spreadm.slope,c2r['lon'])
p = ax.contourf(lons,c2r['lat'],c2rs*1000,
                cmap=new_cmap,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

# Geopotential Height
slope = (cesm2_z3r_fastm.slope-cesm2_z3r.mean('member_id').slope)*10
c = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
               levels=[-8,-6,-4,4,6,8,10,14,18,22,26,30],
               linewidths=0.75,
               colors='k',
               transform=ccrs.PlateCarree()
              )

z = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
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


rvs = rvivt.slope #.where(rvivt.p < 0.05)
rus = ruivt.slope #.where(ruivt.p < 0.05)
Q = ax.quiver(rvs['lon'], rvs['lat'], rus*10, rvs*10, 
          color='red',
          edgecolor='k',
          transform=ccrs.PlateCarree(), 
          regrid_shape=20)

qk = ax.quiverkey(Q, 0.65, -0.06, 10, r'IVT: $10\ \mathrm{\frac{kg}{m\ s\ decade}}}$', 
                  labelpos='E',
                   #coordinates='figure'
                 )

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
gl.top_labels = False  # Turn off top labels
gl.left_labels = False  # Turn off left labels
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

x, y = [-120, -120, -60, -60, -120], [60, 85, 85, 60, 60]
ax.add_patch(mpatches.Rectangle(xy=[-90, 60], width=40, height=25,
                                    facecolor='none',
                                    edgecolor='magenta',
                                    linestyle='--',
                                    linewidth=1.5,
                                    zorder=103,
                                    alpha=0.85,
                                    transform=ccrs.PlateCarree())
                 )


fig.subplots_adjust(top=0.94,bottom=0.12,left=0.07,right=0.93,hspace=0.3,wspace=0.1)
plt.savefig('./plots/moisture_tagging-figure_3-40mem_fast-supp.1979-2018_blue-era5.png',dpi=600)

# %%
rtmq = linregress(cesm2_tmq.sel(year=slice(1981,2022)),cesm2_tmq.sel(year=slice(1981,2022)).year,dim='year')
rtmq

# %%
plt.plot(rtmq.slope)


# %%
for i in np.arange(0,40,step=1):
    if xr.concat(sorted(rtmq.slope),dim='member_id').member_id.values[i] == 'r20i1251p1f2' or xr.concat(sorted(rtmq.slope),dim='member_id').member_id.values[i] == 'r16i1281p1f2':
        print('*', xr.concat(sorted(rtmq.slope),dim='member_id').member_id.values[i],xr.concat(sorted(rtmq.slope),dim='member_id').values[i])
    else:
        print(xr.concat(sorted(rtmq.slope),dim='member_id').member_id.values[i],xr.concat(sorted(rtmq.slope),dim='member_id').values[i])

# %%
cesm2_tmq.sel(member_id=rtmq_ex['member_id']).member_id.values,cesm2_tmq.sel(member_id=cesm2_z3r_fast['member_id']).member_id.values

# %%
# Create a custom RdYlBu colormap with white in the middle
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 7
mpl.rcParams['hatch.color'] = 'silver'

fig = plt.figure(figsize=(7.5,6))
## Panel 1 --------------------------------------------------
ax = fig.add_subplot(221)
ax.set_xlim([1978,2023])
#ax.set_ylim([100,1000])
ax.set_ylabel(r'JJA AR frequency [%]')
ax.set_xlabel('year')
for axis in ['bottom','left']:
    ax.spines[axis].set_linewidth(1.25)
ax.spines[['right', 'top']].set_visible(False)
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.4))
    
ax.text(-0.08, 0.97, 'a', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

fills = ax.fill_between(np.arange(1979,2019,step=1),100*index_cesm2.min('member_id'),100*index_cesm2.max('member_id'),color='silver',alpha=0.4)

ln0 = ax.plot(np.arange(1979,2019,step=1),100*index_cesm2.sel(member_id=new_fast['member_id']).mean('member_id').sel(year=slice(1979,2018)).values,
              c='#6699CC',linestyle='--',
              linewidth=1.5,label=r'CESM2-LE >1*$\mathrm{\sigma}$')

ln1 = ax.plot(np.arange(1979,2019,step=1),100*index_cesm2.mean('member_id').sel(year=slice(1979,2018)).values,
              c='#6699CC',linewidth=1.5,label='CESM2-LE')
ln2 = ax.plot(np.arange(1979,2023,step=1),100*index_freq.sel(year=slice(1979,2022)).values,c='k',linewidth=1.5,label='iCESM1')
ln3 = ax.plot(np.arange(1979,2020,step=1),100*index_era5.sel(year=slice(1979,2019)).values,c='#AA4499',linewidth=1.5,label='ERA5')

plt.legend(frameon=False,loc='upper left')

# Panel 2--------------------------------------------------
ax = fig.add_subplot(222,projection=ccrs.NorthPolarStereo())
ax.set_extent([-180,180,50,90],ccrs.PlateCarree())
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax.set_title('iCESM1')

ax.text(-0.08, 0.97, 'b', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

colors1 = plt.cm.YlOrRd(np.linspace(0, 1, 128))
colors2 = plt.cm.Blues(np.linspace(0, 1, 128))[::-1]
new_colors = np.vstack((colors2[:], np.ones((20, 4)), colors1[:]))
new_cmap = LinearSegmentedColormap.from_list('RdYlBu_r_white', new_colors, N=16)

new_cmap = colormaps.BlueWhiteOrangeRed
bounds = [-2.4,-2.2,-2.0,-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,0,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4]
irs,lons = add_cyclic_point(ir.slope,ir['lon'])
p = ax.contourf(lons,ir['lat'],irs*1000,
                cmap=new_cmap,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

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

rvs = rv.slope #.where(rt.p < 0.05)
rus = ru.slope #.where(rt.p < 0.05)
Q = ax.quiver(rv['lon'], rv['lat'], rus*10, rvs*10, 
          color='purple',
          edgecolor='k',
          transform=ccrs.PlateCarree(), 
          regrid_shape=20)

qk = ax.quiverkey(Q, 0.65, -0.06, 10, r'IVT: $10\ \mathrm{\frac{kg}{m\ s\ decade}}}$', 
                  labelpos='E',
                   #coordinates='figure'
                 )

# Colorbar
#cax = fig.add_axes([0.2,0.07,0.6,0.01])
cax = fig.add_axes([0.31,0.055,0.385,0.01])
cb = plt.colorbar(p, cax=cax,orientation='horizontal', 
                  drawedges=True,
                  #ticks=bounds[::2]
                 )
#cb.set_ticklabels(bounds[::2])
cb.set_label(r'JJA AR frequency [% $\mathrm{decade^{-1}}$] ')
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
gl.top_labels = False  # Turn off top labels
gl.left_labels = False  # Turn off left labels
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

# Panel 3--------------------------------------------------
ax = fig.add_subplot(223,projection=ccrs.NorthPolarStereo())
ax.set_extent([-180,180,50,90],ccrs.PlateCarree())
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax.set_title('CESM2-LE mean')

ax.text(-0.08, 0.97, 'c', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

cmap = colormaps.BlueWhiteOrangeRed
c2rs,lons = add_cyclic_point(c2r.slope,c2r['lon'])
p = ax.contourf(lons,c2r['lat'],c2rs*1000,
                cmap=new_cmap,
                hatches=['//'],
                alpha=0,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

s = ax.contourf(lons,c2r['lat'],c2rs*1000,
                cmap=new_cmap,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

# Geopotential Height
slope = cesm2_z3r.mean('member_id').slope*10
c = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
               levels=[-6,-4,6,10,14,18,22,26,30],
               linewidths=0.75,
               colors='k',
               transform=ccrs.PlateCarree()
              )

z = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
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
gl.top_labels = False  # Turn off top labels
gl.left_labels = False  # Turn off left labels
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

# Panel 4--------------------------------------------------
ax = fig.add_subplot(224,projection=ccrs.NorthPolarStereo())
ax.set_extent([-180,180,50,90],ccrs.PlateCarree())
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax.set_title(r'CESM2-LE >1*$\mathrm{\sigma}$')

ax.text(-0.08, 0.97, 'd', weight='bold',
        fontsize=17,
        horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

cmap = colormaps.BlueWhiteOrangeRed
c2rs,lons = add_cyclic_point(new_fast.mean('member_id').slope,c2r['lon'])
p = ax.contourf(lons,c2r['lat'],c2rs*1000,
                cmap=new_cmap,
                extend='both',
                levels=bounds,
                transform=ccrs.PlateCarree())

# Geopotential Height
slope = cesm2_z3r_new.mean('member_id').slope*10
c = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
               levels=[-6,-4,6,10,14,18,22,26,30],
               linewidths=0.75,
               colors='k',
               transform=ccrs.PlateCarree()
              )

z = ax.contour(cesm2_z3r['lon'],cesm2_z3r['lat'],slope,
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
gl.top_labels = False  # Turn off top labels
gl.left_labels = False  # Turn off left labels
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([50,60,70,80])

fig.subplots_adjust(top=0.96,bottom=0.12,left=0.07,right=0.93,hspace=0.3,wspace=0.1)
plt.savefig('./plots/moisture_tagging-figure_3-40mem_fast-tmq.png',dpi=600)
