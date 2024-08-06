# %%
import xarray as xr
import numpy as np
import glob as glob
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from matplotlib.patches import Rectangle

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

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
from matplotlib.ticker import ScalarFormatter,AutoLocator,AutoMinorLocator,FixedLocator
from matplotlib.ticker import FuncFormatter

from imods.stats import *

# %% [markdown]
# ## START HERE

# %%
ds = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/amip-low-water-nudging3/moisture_tagging_paper/data/54region.jja_q_new.nc')['TA01V'].sel(year=slice(1981,2022))
ds

# %%
neg = (ds.sel(lat=slice(70,90)))
neg

# %%
neg_dtrend = detrend(neg,axis=1)
neg_dtrend

# %%
def calculate_rmse(actual, predicted):
    """
    Calculate Root Mean Squared Error (RMSE)

    Parameters:
    - actual (numpy array): Array of actual values
    - predicted (numpy array): Array of predicted values

    Returns:
    - rmse (float): Root Mean Squared Error
    """
    if len(actual) != len(predicted):
        raise ValueError("Lengths of actual and predicted arrays must be the same")

    # Calculate squared differences
    squared_diff = (actual - predicted) ** 2

    # Calculate mean squared error
    mean_squared_error = np.mean(squared_diff)

    # Calculate square root to get RMSE
    rmse = np.sqrt(mean_squared_error)

    return rmse

# Example usage:
actual_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
predicted_values = np.array([1.2, 1.8, 3.2, 4.2, 5.1])

rmse_result = calculate_rmse(actual_values, predicted_values)
print("Root Mean Squared Error:", rmse_result)

# %%
rmse_dtrend = xr.apply_ufunc(calculate_rmse, neg_dtrend.sum('region'), neg_dtrend.sum('region')-neg_dtrend,
                      input_core_dims=[['year'],['year']],
                      output_core_dims=[[]],
                      vectorize=True
                     )
rmse_dtrend

# %%
rmse = xr.apply_ufunc(calculate_rmse, neg.sum('region'), neg.sum('region')-neg,
                      input_core_dims=[['year'],['year']],
                      output_core_dims=[[]],
                      vectorize=True
                     )
rmse

# %%
rmse_reg = area_weighted_ave(rmse) 

rmse_reg_dtrend = area_weighted_ave(rmse_dtrend) 

# %%
rmse_rel = (rmse_reg/rmse_reg.sum('region')) #.plot()

# %%
areacella = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/CESM2-LE/CESM2.areacella.nc')['areacella'].squeeze().load()
#areacella.coords['lat'] = ti['lat']

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
                            dims=rmse_reg.dims,
                            coords=rmse_reg.coords)

#(reg_weights[i]/arc_weighted.values)
xreg_weights

# %%
# Number of boxes
num_boxes  = 54
xnum_boxes = 6
ynum_boxes = 9

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

# Create an xarray dataset with the grid and values
#grid = xr.Dataset(
#    {'grid_values': (['lat', 'lon'], grid_values)},
#    coords={'lat': (['lat'], latitudes[:-1]), 'lon': (['lon'], longitudes[:-1])},
#)

grid = xr.DataArray(rmse_reg.values.reshape(lat_grid.shape),
                    name='RMSE contribution to Arctic TMQ',
                    dims=('lat','lon'),
                    coords={'lat': latitudes[:], 'lon': longitudes[:]},
                   )

# Display the resulting xarray dataset
print(grid)

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

dgrid = xr.DataArray(rmse_reg_dtrend.values.reshape(lat_grid.shape),
                    name='RMSE contribution to Arctic TMQ',
                    dims=('lat','lon'),
                    coords={'lat': latitudes[:], 'lon': longitudes[:]},
                   )

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

# Create an xarray dataset with the grid and values
wgrid = xr.DataArray((rmse_reg*xreg_weights).values.reshape(lat_grid.shape),
                    name='RMSE contribution to Arctic TMQ',
                    dims=('lat','lon'),
                    coords={'lat': latitudes[:], 'lon': longitudes[:]},
                   )


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

# Create an xarray dataset with the grid and values
#grid = xr.Dataset(
#    {'grid_values': (['lat', 'lon'], grid_values)},
#    coords={'lat': (['lat'], latitudes[:-1]), 'lon': (['lon'], longitudes[:-1])},
#)

dwgrid = xr.DataArray((rmse_reg_dtrend*xreg_weights).values.reshape(lat_grid.shape),
                    name='RMSE contribution to Arctic TMQ',
                    dims=('lat','lon'),
                    coords={'lat': latitudes[:], 'lon': longitudes[:]},
                   )

# Display the resulting xarray dataset
print(dwgrid)

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

# Create an xarray dataset with the grid and values
rgrid = xr.DataArray(rmse_rel.values.reshape(lat_grid.shape),
                    name='RMSE contribution to Arctic TMQ',
                    dims=('lat','lon'),
                    coords={'lat': latitudes[:], 'lon': longitudes[:]},
                   )


# %%
def index(ds):
    return (ds*areacella).sel(lat=slice(70,90)).sum(dim=['lat','lon'])/areacella.sel(lat=slice(70,90)).sum(dim=['lat','lon'])

areacella = xr.open_dataset('/raid/scratch/scratch-itbaxter/exp/CESM2-LE/CESM2-FV2.areacella.nc')['areacella'].squeeze().load()


# %%
arc = areacella.sel(lat=slice(70,90)).sum(dim=['lat','lon'])

# %%
globe = areacella.sum(dim=['lat','lon'])

# %%
area_grid = np.zeros((9,6))
lats = np.arange(-90,90.1,step=20)
lons = np.arange(0,360.1,step=60)
for i in range(6):
    for j in range(9):
        area_grid[j,i] = areacella.sel(lat=slice(lats[j],lats[j+1]),lon=slice(lons[i],lons[i+1])).sum(dim=['lat','lon'])

area_grid,area_grid.shape

# %%
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

# Create an xarray dataset with the grid and values
#grid = xr.Dataset(
#    {'grid_values': (['lat', 'lon'], grid_values)},
#    coords={'lat': (['lat'], latitudes[:-1]), 'lon': (['lon'], longitudes[:-1])},
#)

area = xr.DataArray(rmse_rel.values.reshape(lat_grid.shape),
                    name='areacella',
                    dims=('lat','lon'),
                    coords={'lat': latitudes[:], 'lon': longitudes[:]},
                   )

# Display the resulting xarray dataset
print(area)

# %%
def area_weighting(ds,gw):
    # Make an area weighting matrix
    #weight=repmat(gw(:)',[length(lon) 1]);         
    spat, weight = xr.broadcast(ds,gw)
    weight=weight/np.nansum(weight)

    n2_q_globalmean= (ds*weight) #.sum(dim=['lat','lon']) #np.nansum(np.nansum(n2_q*weight,2),1)  
    return n2_q_globalmean

def sia_index(ds):
    """
    Calculates a Sea Ice Area (SIA) Index using all sic values >= 0.
    Spatial dimensions should be named 'lat' & 'lon' 
    """
    dx = ds.lon.diff('lon')
    dy = ds.lat.diff('lat')
    area = (dx[0]*3.1415926*6371./360.*2.)*(dy[0]*3.1415926*6371./360.*2.)

    if (ds.any() > 1.0):
        sic = ds.where(ds >= 0) / 100.
    else:
        sic = ds.where(ds >= 0)

    index = (sic * np.cos(sic.lat / 180. * 3.1415926) * area).sum(('lat','lon'),skipna=True) / 1000000
    return index 

def area_weighted_ave(ds):
    coslat = np.cos(np.deg2rad(ds.lat))
    ds,coslat = xr.broadcast(ds,coslat)
    ds = ds * coslat
    #return ds.mean(('lat','lon'),skipna=True)
    return ds.sum(('lat','lon'),skipna=True)/((ds/ds)*coslat).sum(('lat','lon'),skipna=True)

# %%
neg_rel = neg/neg.sum('region')

# %%
ds_y = (ds*areacella).sel(lat=slice(70,90)).sum(dim=['lat','lon'])/areacella.sel(lat=slice(70,90)).sum(dim=['lat','lon'])
tot = linregress(ds_y.sum('region'),neg.year)
tot

# %%
r = linregress(ds_y,ds_y.year)
r

# %%
wds_y = xreg_weights*(neg*areacella.sel(lat=slice(70,90))).sum(dim=['lat','lon'])/areacella.sel(lat=slice(70,90)).sum(dim=['lat','lon'])
wtot = linregress(wds_y.sum('region'),neg.year)
wtot

# %%
wr = linregress(wds_y,ds_y.year)
wr

# %%
r_diff = r.slope
r_diff_values = r_diff.values.reshape(9,6)


# %%
# Create a custom RdYlBu colormap with white in the middle
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 7
mpl.rcParams['hatch.color'] = 'silver'

fig = plt.figure(figsize=(7.5,7.5))
# Panel 1--------------------------------------------------
ax = fig.add_subplot(211,projection=ccrs.Robinson(central_longitude=180))
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax.set_title('Mean [1981-2022]')

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
        if abs(grid[i,j]) >= 0.07:
            ax.text(lon1+25, lat2-15, square['name'], color='white', fontsize=8, va='bottom', ha='left', transform=ccrs.PlateCarree())
        else:
            ax.text(lon1+25, lat2-15, square['name'], color='black', fontsize=8, va='bottom', ha='left', transform=ccrs.PlateCarree())
        cnt += 1

# Create a grid of values corresponding to the squares
x = np.linspace(0, 360, num_cols + 1)
y = np.linspace(-90, 90, num_rows + 1)
X, Y = np.meshgrid(x, y)
Z = np.random.rand(num_rows, num_cols)  

cmap = colormaps.BlueWhiteOrangeRed
boundaries = np.arange(0, 1.61, step=0.2)
colors = plt.cm.GnBu(np.linspace(0, 1, 256))
new_colors = np.vstack((np.ones((2, 4)), colors[:256]))
new_cmap = LinearSegmentedColormap.from_list('RdYlBu_r_white', new_colors, N=16)

norm = BoundaryNorm(boundaries, new_cmap.N, clip=True)

contour = ax.pcolormesh(X, Y, 1000*dwgrid.values, 
                        cmap=new_cmap, 
                        transform=ccrs.PlateCarree(), 
                        norm=norm
                       )

# Colorbar
cax = fig.add_axes([0.85,0.55,0.02,0.42])
cb = plt.colorbar(contour,cax=cax,orientation='vertical', 
                  drawedges=True,
                  ticks=boundaries[::1]
                 )

cb.set_label(r'Contribution to mean Arctic WV [g $\mathrm{m^{-2}}$]')
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

cb2 = plt.colorbar(contour,cax2,orientation='vertical',
                  drawedges=True,
                  ticks=boundaries[::1],
                 )
ticklabels = 100*(boundaries[::1]/np.sum(1000*dwgrid.values))
cb2.ax.set_yticklabels(["{:.02f}".format(i) for i in ticklabels])
cb2.outline.set_color('k')
cb2.outline.set_linewidth(0.8)
cb2.dividers.set_color('k')
cb2.dividers.set_linewidth(0.8)
cb2.ax.tick_params(size=0)
cb2.ax.minorticks_off()
cb2.set_label(r'Relative contribution to mean Arctic WV [%]')
cb2.ax.yaxis.set_label_position('right')

ax.spines['geo'].set_linewidth(1.25)

# Add gridlines
gl = ax.gridlines(draw_labels=True, linewidth=1.0,  
                  x_inline=False,
                  y_inline=False,
                  rotate_labels=False, color='k', 
                  linestyle='-', alpha=1.0)
gl.top_labels = False   
gl.left_labels = True 
gl.right_labels = False  
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([-90, -70, -50, -30, -10, 10, 30, 50, 70, 90])

# Panel 2--------------------------------------------------
ax = fig.add_subplot(212,projection=ccrs.Robinson(central_longitude=180))
ax.coastlines(linewidth=0.7,alpha=1.0,color='dimgray')
ax.set_title('Trends [1981-2022]')


ax.text(-0.03, 1.05, 'b', weight='bold',
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
        if abs(r_diff_values[i,j]) >= 1:
            ax.text(lon1+25, lat2-15, square['name'], color='white', fontsize=8, va='bottom', ha='left', transform=ccrs.PlateCarree())
        else:
            ax.text(lon1+25, lat2-15, square['name'], color='black', fontsize=8, va='bottom', ha='left', transform=ccrs.PlateCarree())
        cnt += 1

cmap = colormaps.BlueWhiteOrangeRed
levels = np.arange(-0.35,0.351,step=0.025)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

p = ax.pcolormesh(X, Y, 10000*wr.slope.values.reshape(9,6), 
                        cmap=cmap,
                        #shading='auto', 
                        alpha=1.0, 
                        norm=norm,
                        transform=ccrs.PlateCarree(),
                       )
# Colorbar

cax = fig.add_axes([0.85,0.08,0.02,0.42])
cb = plt.colorbar(p,cax=cax,orientation='vertical', 
                  drawedges=True,
                  ticks=levels[::2]
                 )

cb.set_label(r'Contribution to Arctic WV trend [g $\mathrm{m^{-2}\ decade^{-1}}$]')
cb.outline.set_color('k')
cb.outline.set_linewidth(0.8)
cb.dividers.set_color('k')
cb.dividers.set_linewidth(0.8)
cb.ax.tick_params(size=0)
cb.ax.minorticks_off()
cb.ax.yaxis.set_label_position('left')

cb.ax.set_aspect('auto')
pos = cb.ax.get_position()

# create a second axes instance and set the limits 
cax2 = cb.ax.twinx()
cax2.set_ylim([-1,1])

# resize the colorbar (otherwise it overlays the plot)
cax2.set_position(pos)

fmt = lambda x, pos: '{:.02f}'.format(x)

cb2 = plt.colorbar(p,cax2,orientation='vertical',
                  drawedges=True,
                  ticks=levels[::2],
                
                 )
ticklabels = 100*(levels[::2]/np.sum(10000*wr.slope.values))
cb2.ax.set_yticklabels(["{:.01f}".format(i) for i in ticklabels])
cb2.outline.set_color('k')
cb2.outline.set_linewidth(0.8)
cb2.dividers.set_color('k')
cb2.dividers.set_linewidth(0.8)
cb2.ax.tick_params(size=0)
cb2.ax.minorticks_off()
cb2.set_label(r'Relative Contribution to Arctic WV trend [%]')
cb2.ax.yaxis.set_label_position('right')

ax.spines['geo'].set_linewidth(1.25)

# Add gridlines
gl = ax.gridlines(draw_labels=True, linewidth=1.0,  
                  x_inline=False,
                  y_inline=False,
                  rotate_labels=False, color='k', 
                  linestyle='-', alpha=1.0)
gl.top_labels = False  
gl.left_labels = True  
gl.right_labels = False  
gl.ylabels = False
gl.xlabel_style={'color':'k'}

# Customize the gridline labels if needed
gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER
gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
gl.ylocator = mticker.FixedLocator([-90, -70, -50, -30, -10, 10, 30, 50, 70, 90])

fig.subplots_adjust(top=0.92,bottom=0.11,left=0.03,right=0.80,hspace=0.3,wspace=0.1)
#fig.savefig('./plots/moisture_tagging-figure_2-2panel-rob-weighted-dtrend.png',dpi=600)



# %%
