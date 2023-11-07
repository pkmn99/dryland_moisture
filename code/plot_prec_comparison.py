import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature

# temporal_res: 'y' 'mon' 'ymonmean'
# spatial_res: '025deg','050deg'
# Load ERA5 data with various options
def load_era5_data(var,temporal_res,spatial_res='05deg',start_year=1990,end_year=2022,cn_label=False):
    d = xr.open_dataset('../../data/ERA5/ERA5-%s-%d-%d-%s-%s.nc'%(var,start_year,end_year,
                                                                  'mon',spatial_res))
    var_text={'prec':'tp','ET':'e','temp':'t2m','solar':'msdwswrf'}
    var_unit_scaler={'prec':1000,'ET':-1000}

    # do unit conversion

    if temporal_res=='y':
        if ((var=='prec') or (var=='ET')):
            d=(d[var_text[var]]*d.time.dt.daysinmonth).resample(time="Y").sum(dim=['time'])*var_unit_scaler[var]
        else:
            d=d[var_text[var]].resample(time="Y").mean(dim=['time'])

    if temporal_res=='ymonmean':
        if ((var=='prec') or (var=='ET')):
            d=(d[var_text[var]]*d.time.dt.daysinmonth).groupby("time.month").mean(dim=['time'])*var_unit_scaler[var]
        else:
            d=d[var_text[var]].groupby("time.month").mean(dim=['time'])

    if temporal_res=='mon':
        if ((var=='prec') or (var=='ET')):
            d=(d[var_text[var]]*d.time.dt.daysinmonth)*var_unit_scaler[var]
        else:
            d=d[var_text[var]]

    # if China subset
    if cn_label:
#         d_ref=xr.open_dataset('../data/results/e_to_prec_mon_2008-2017.nc')
        d_ref=xr.open_dataset('../data/results/e_to_prec_mon_1990-2022.nc')

        return d.sel(lat=d_ref.lat, lon=d_ref.lon, method="nearest")
    else:
        return d
#     ERA5-prec-1990-2022-mon-025deg.nc


def load_e2p_data(var,temporal_res,start_year=1990,end_year=2022,et_data='ERA5'):
    if et_data=='GLEAM':
        d = xr.open_dataset('../data/results/%s_e_to_prec_mon_%d-%d.nc'%(et_data,start_year,
                                                                   end_year))
    else:
        d = xr.open_dataset('../data/results/e_to_prec_mon_%d-%d.nc'%(start_year,
                                                                 end_year))
    if temporal_res=='y':
        d=d[var].resample(time="Y").sum(dim=['time'])

    if temporal_res=='ymonmean':
        d=d[var].groupby("time.month").mean(dim=['time'])

    if temporal_res=='mon':
        d=d[var]

    return d


def plot_map(d, ax, levels, minmax=[],cmap='rainbow_r',extent=[73, 135, 28, 52],pr=ccrs.PlateCarree(),
             lw=0.5,mycolorbar=False):
    # Load geographical data
#     china_shp=shpreader.Reader('../data/shp/China_provinces_with_around_countries_simple.shp')
    china_shp=shpreader.Reader('/media/liyan/HDD/Project/data/China_gis/国界线/国界线.shp')

    cn_dryland_shp=shpreader.Reader('../data/shp/arid_areas/AI_1901-2017.shp')
  
    china_feature = ShapelyFeature(china_shp.geometries(), pr, facecolor='none')
    cn_dryland_feature = ShapelyFeature(cn_dryland_shp.geometries(), pr, facecolor='none')

   # im=d.plot.contourf(cmap=cmap, levels=levels, add_colorbar=False,ax=ax)
    im=d.plot(cmap=cmap, levels=levels, add_colorbar=False,ax=ax)
    if mycolorbar:
        cb = plt.colorbar(im, orientation="vertical", pad=0.15)
        cb.set_label(label='Precipitation (mm/yr)')

    ax.add_feature(china_feature,edgecolor='dimgrey', linewidth=lw)
    ax.add_feature(cn_dryland_feature,edgecolor='r', linewidth=lw)
    ax.set_extent(extent,ccrs.Geodetic())
    ax.coastlines()
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND)
    return im

# linear trend
def linear_trend(x,y):
    con = (~np.isnan(x)) & (~np.isnan(y))
    b = np.polyfit(x[con],y[con],deg=1)
    print('r=%f'%np.corrcoef(x[con],y[con])[0,1])
    return np.polyval(b,x)

def make_plot():

    # load data, precipitation by ERA5 land and all ET 
#    dep_ycn_land=load_e2p_data('e_to_prec_land','y',end_year=2022)
#    dep_ycn=load_e2p_data('e_to_prec','y',end_year=2022)
    # precipitation by gleam ET 
    dep_ycn_gleam=load_e2p_data('e_to_prec','y',end_year=2020,et_data='GLEAM')

    # ERA5 precipitation
    dp_ycn=load_era5_data('prec','y',cn_label=True)
    # Aridity index
    ai=xr.open_dataset('../data/AI_1901-2017_360x720.nc')

    ai_con=ai.Band1>0
    
    pr=ccrs.PlateCarree()
    
    fig = plt.figure(figsize=[10, 10])
    
    ax1 = fig.add_axes([0, 0.5, 0.4, 0.35], projection=pr)
    im=plot_map(dep_ycn_gleam.mean(dim='time').where(ai_con),ax=ax1, levels=np.arange(0,1000,10),
             cmap='Blues')
    ax1.set_title('Precipitation by upwind land ET')
    
    ax2 = fig.add_axes([0, 0.28, 0.4, 0.35], projection=pr)
    plot_map(dp_ycn.mean(dim='time').where(ai_con),ax=ax2, levels=np.arange(0,1000,10),
             cmap='Blues')
    ax2.set_title('Precipitation of ERA5')
    
    ax3 = fig.add_axes([0.5, 0.4, 0.4, 0.35])
    ax3.scatter(dep_ycn_gleam.mean(dim='time').where(ai.Band1>0).values.flatten(),
                dp_ycn.mean(dim='time').where(ai.Band1>0).values.flatten(),color='royalblue',s=5)
    
    plt.plot(dep_ycn_gleam.mean(dim='time').where(ai.Band1>0).values.flatten(),
             linear_trend(dep_ycn_gleam.mean(dim='time').where(ai.Band1>0).values.flatten(),
             dp_ycn.mean(dim='time').where(ai.Band1>0).values.flatten()),'r')
    ax3.plot([0,1200],[0,1200],'k--')
    ax3.text(1000,150,'r=0.83')
    
    ax3.set_xlabel('Precipitation by upwind land ET (mm/yr)')
    ax3.set_ylabel('Precipitation of ERA5 (mm/yr)')
    ax3.set_ylim([0,1200])
    ax3.set_xlim([0,1200])
    
    # # Add colorbar to big plot
    cbarbig1_pos = [ax2.get_position().x0, ax2.get_position().y0-0.03, ax2.get_position().width, 0.015]
    
    caxbig1 = fig.add_axes(cbarbig1_pos)
    
    cb = plt.colorbar(im, orientation="horizontal", pad=0.15,cax=caxbig1)
    cb.set_label(label='Precipitation (mm/yr)')

    ax1.text(-0.02, 1.05, 'a', fontsize=14, transform=ax1.transAxes, fontweight='bold')
    ax2.text(-0.02, 1.05, 'b', fontsize=14, transform=ax2.transAxes, fontweight='bold')
    ax3.text(-0.02, 1.05, 'c', fontsize=14, transform=ax3.transAxes, fontweight='bold')


    plt.savefig('../figure/fig_prec_comparision.png',dpi=300,bbox_inches='tight')
    print('Fig saved')

if __name__=="__main__":
    make_plot()
    
