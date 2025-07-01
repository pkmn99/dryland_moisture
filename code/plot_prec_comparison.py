import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature
from process_et2prec import load_era5_data,load_e2p_data

# Create area weights
# Usage df.weighted(weights).sum()
def my_weights(da):
    # create area weights following
    # https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html
    weights = np.cos(np.deg2rad(da.lat))
    weights.name = "weights"
    return weights

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
   # ax.add_feature(cfeature.LAND,facecolor='none')
    ax.add_feature(cfeature.LAND)
    return im

# linear trend and corr
def linear_trend_corr(x,y):
    con = (~np.isnan(x)) & (~np.isnan(y))
    b = np.polyfit(x[con],y[con],deg=1)
    print('r=%f'%np.corrcoef(x[con],y[con])[0,1])
    return np.polyval(b,x),np.corrcoef(x[con],y[con])[0,1]

def make_plot(et_data='GLEAM'):

    # load data, precipitation by ERA5 land and all ET 
#    dep_ycn_land=load_e2p_data('e_to_prec_land','y',end_year=2022)
#    dep_ycn=load_e2p_data('e_to_prec','y',end_year=2022)
    # precipitation by gleam ET 
    dep_ycn_gleam=load_e2p_data('e_to_prec','y',end_year=2022,et_data=et_data)

    # ERA5 precipitation
    dp_ycn=load_era5_data('prec','y',cn_label=True)
    print(dp_ycn.shape)
    print(dep_ycn_gleam.shape)
    # Aridity index
    ai=xr.open_dataset('../data/AI_1901-2017_360x720.nc')

    ai_con=ai.Band1>0
    
    pr=ccrs.PlateCarree()

    w=my_weights(dp_ycn)
    
    fig = plt.figure(figsize=[10, 10])
    
    ax1 = fig.add_axes([0, 0.5, 0.4, 0.35], projection=pr)
    im=plot_map(dep_ycn_gleam.mean(dim='time').where(ai_con),ax=ax1, levels=np.arange(0,1000,10),
             cmap='Blues')
    ax1.set_title('Precipitation by upwind ET (P$_\mathrm{E}$)')
    
    ax2 = fig.add_axes([0, 0.28, 0.4, 0.35], projection=pr)
    plot_map(dp_ycn.mean(dim='time').where(ai_con),ax=ax2, levels=np.arange(0,1000,10),
             cmap='Blues')
    ax2.set_title('Precipitation of ERA5 (P)')
    # add regional mean P
    ax1.text(0.5, 0.85, '%dmm'%dep_ycn_gleam.mean(dim='time').where(ai_con).weighted(w).mean().values,
             fontsize=12, transform=ax1.transAxes, ha='center')
    ax2.text(0.5, 0.85, '%dmm'%dp_ycn.mean(dim='time').where(ai_con).weighted(w).mean().values,
             fontsize=12, transform=ax2.transAxes, ha='center')
    print('precip in dryland area mean is %f'%dp_ycn.mean(dim='time').where(ai_con).weighted(w).mean().values)
    print('precip contribution in dryland area mean is %f'%dep_ycn_gleam.mean(dim='time').where(ai_con).weighted(w).mean().values)
    
    ax3 = fig.add_axes([0.5, 0.4, 0.4, 0.35])
    ax3.scatter(dep_ycn_gleam.mean(dim='time').where(ai.Band1>0).values.flatten(),
                dp_ycn.mean(dim='time').where(ai.Band1>0).values.flatten(),color='royalblue',s=5)

    [ft,r]=linear_trend_corr(dep_ycn_gleam.mean(dim='time').where(ai.Band1>0).values.flatten(),
                        dp_ycn.mean(dim='time').where(ai.Band1>0).values.flatten())
    plt.plot(dep_ycn_gleam.mean(dim='time').where(ai.Band1>0).values.flatten(),
             ft,'r')
    ax3.plot([0,1200],[0,1200],'k--')
    ax3.text(1000,150,'$r$=%.2f'%r)
    
    ax3.set_xlabel('P$_\mathrm{E}$ (mm/yr)')
    ax3.set_ylabel('P (mm/yr)')
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

    plt.savefig('../figure/fig_prec_comparision_%s0129.png'%et_data,dpi=300,bbox_inches='tight')
    print('Fig saved')

if __name__=="__main__":
    make_plot(et_data='ERA5')
    
