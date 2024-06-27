import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from plot_prec_comparison import plot_map
from plot_china_dryland_source import set_lat_lon

"""
A map figure to show moisture source of China drylands
"""

def make_plot():
    # Load data 
    s = xr.open_dataset('../data/results/china_dryland_prec_source_aridity.nc')
     
    levels=np.array([1,10,50,100,200,300,1000])/3

    # define map projection
    pr=ccrs.PlateCarree()

    fig = plt.figure(figsize=[18, 14])

    for i in range(12):
        ax = fig.add_subplot(4, 3, i+1, projection=pr)
        im1=plot_map(s.e_to_prec.sum(dim='aridity')[i], ax=ax, levels=levels, cmap='Blues',extent=[0, 150, 0, 70])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('Month %d'%(i+1))

        set_lat_lon(ax, range(0,151,30), range(10,71,20), label=True, pad=0.05, fontsize=10)

#    # Add colorbar to big plot
#    cbarbig1_pos = [ax1.get_position().x0, ax1.get_position().y0-0.03, ax1.get_position().width, 0.02]
#    caxbig1 = fig.add_axes(cbarbig1_pos)
#    cb = plt.colorbar(im1, orientation="horizontal", pad=0.15,cax=caxbig1,extend='neither',
#                     ticks=levels)
#    cb.set_label(label='Precipitation contribution (mm/yr)')
    
    plt.savefig('../figure/fig_china_dryland_prec_source_monthly1124.png',dpi=300,bbox_inches='tight')
    print('figure saved')

# report internal precipitation contribution from drylands itself
def print_dryland_ratio():
    s = xr.open_dataset('../data/results/china_dryland_prec_source_aridity.nc')
    ai=xr.open_dataset('../data/AI_1901-2017_360x720.nc')# aridity data
    # create area weights following 
    # https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html

    weights = np.cos(np.deg2rad(s.lat))
    weights.name = "weights"
    for i in range(4):
        pe_in = s.e_to_prec[i].sum(dim='month').where(ai.Band1>0).weighted(weights).sum().values # in drylands
        pe_out = s.e_to_prec[i].sum(dim='month').where(ai.Band1.isnull()).weighted(weights).sum().values# out drylands
        print('dryland internal contribution to precipitation for aridity %d is %f'%(i+1,pe_in/(pe_in+pe_out)))

#    for i in range(12):
#        pe_in = s.e_to_prec.sum(dim='aridity')[i].where(ai.Band1>0).weighted(weights).sum().values # in drylands
#        pe_out = s.e_to_prec.sum(dim='aridity')[i].where(ai.Band1.isnull()).weighted(weights).sum().values# out drylands
#        print('dryland internal contribution to precipitation for month %d is %f'%(i+1,pe_in/(pe_in+pe_out)))

if __name__=="__main__":
#   make_plot() 
   print_dryland_ratio()
