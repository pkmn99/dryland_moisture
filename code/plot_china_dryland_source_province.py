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
def load_dryland_province_index():
    cn_grid=pd.read_csv('../data/china_province_grid_index.csv')
    subregion_index=pd.read_csv('../data/china_dryland_grid_index.csv')
    return subregion_index.merge(cn_grid,on=['lat','lon']).rename(columns={'id_x':'id_dry','id_y':'id_cn'})

def make_plot():
    # Load data 
    s = xr.open_dataset('../data/results/china_dryland_prec_source_province.nc')

    levels=np.array([1,10,50,100,200,300,1000])/4
    drycn_index=load_dryland_province_index()
    region_id=drycn_index['id_cn'].unique()

    # define map projection
    pr=ccrs.PlateCarree()

    fig = plt.figure(figsize=[18, 14])

    for i in range(region_id.shape[0]):
        ax = fig.add_subplot(5, 4, i+1, projection=pr)
        im1=plot_map(s.e_to_prec.sum(dim='month').sel(province=region_id[i]), ax=ax, levels=levels, cmap='Blues',extent=[0, 150, 0, 70])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('%s'%drycn_index.loc[drycn_index['id_cn']==region_id[i],'name'].iloc[0])
        set_lat_lon(ax, range(0,151,30), range(10,71,20), label=True, pad=0.05, fontsize=10)

#    # Add colorbar to big plot
    cbarbig1_pos = [ax.get_position().x1+0.015, ax.get_position().y0+0.01, 0.01, ax.get_position().height]
    caxbig1 = fig.add_axes(cbarbig1_pos)
    cb = plt.colorbar(im1, orientation="vertical", pad=0.15,cax=caxbig1,extend='neither',
                     ticks=levels)
    cb.set_label(label='Precipitation contribution (mm/yr)', fontsize=9)
    
    plt.savefig('../figure/fig_china_dryland_prec_source_province0808.png',dpi=300,bbox_inches='tight')
    print('figure saved')

if __name__=="__main__":
   make_plot() 
