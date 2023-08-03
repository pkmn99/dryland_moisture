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

# Add lat lon to map figure
def set_lat_lon(ax, xtickrange, ytickrange, label=False,pad=0.05, fontsize=8,pr=ccrs.PlateCarree()):
    lon_formatter = LongitudeFormatter(zero_direction_label=True, degree_symbol='')
    lat_formatter = LatitudeFormatter(degree_symbol='')
    ax.set_yticks(ytickrange, crs=pr)
    ax.set_xticks(xtickrange, crs=pr)
    if label:
        ax.set_xticklabels(xtickrange,fontsize=fontsize)
        ax.set_yticklabels(ytickrange,fontsize=fontsize)
        ax.tick_params(axis='x', which='both', direction='out', bottom=False, top=True,labeltop=True,labelbottom=False, pad=pad)
        ax.tick_params(axis='y', which='both', direction='out', pad=pad)

    else:
        ax.tick_params(axis='x', which='both', direction='out', bottom=True, top=False, labeltop=False, labelleft=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', direction='out', left=True, labelleft=False)

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_ylabel('')
    ax.set_xlabel('')

def make_plot():
    # Load data 
    s = xr.open_dataset('../data/results/china_dryland_prec_source.nc')
     
    levels=[0.001,0.01,0.1,0.5,1,2]

    # define map projection
    pr=ccrs.PlateCarree()

    fig = plt.figure(figsize=[6, 4])

    ax1 = fig.add_axes([0, 0, 1, 1], projection=pr)

    im1=plot_map(s.e_to_prec.where(s.e_to_prec.sum(dim='month')>0.001).sum(dim='month'),
                 ax=ax1, levels=levels, cmap='Blues',extent=[0, 150, 10, 80])

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.set_title('Upwind source region of precipitation in China\'s drylands', fontsize=12)

    set_lat_lon(ax1, range(0,151,30), range(10,81,20), label=True, pad=0.05, fontsize=10)

    # Add colorbar to big plot
    cbarbig1_pos = [ax1.get_position().x0, ax1.get_position().y0-0.03, ax1.get_position().width, 0.02]
    caxbig1 = fig.add_axes(cbarbig1_pos)
    cb = plt.colorbar(im1, orientation="horizontal", pad=0.15,cax=caxbig1,extend='neither',
                     ticks=levels)
    cb.set_label(label='Precipitation contribution (mm/yr)')
    
    plt.savefig('../figure/fig_china_dryland_prec_source.png',dpi=300,bbox_inches='tight')
    print('figure saved')

if __name__=="__main__":
   make_plot() 
