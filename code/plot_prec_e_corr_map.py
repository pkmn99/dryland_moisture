import seaborn as sns
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cartopy.crs as ccrs
from plot_prec_comparison import plot_map
from plot_china_dryland_source import set_lat_lon
from process_et2prec import load_era5_data,load_gleam_data,load_e2p_data,make_cn_mask
from plot_pe_prec_corr_map import xr_detrend,xr_stats_corr

def make_plot(et_data='GLEAM'):
    #Load data
    if et_data=='GLEAM':
        de_ycn_gleam = load_gleam_data('y',cn_label=True) # GLEAM ET China
    else:
        de_ycn_gleam = load_era5_data('ET','y',cn_label=True) # ERA5 ET China

    dp_ycn=load_era5_data('prec','y',cn_label=True)

    cn_mask =make_cn_mask()
    
    # corr between prec con and downwind prec 
    [r_e_p,p_map]=xr_stats_corr(xr_detrend(dp_ycn.where(cn_mask>0).sel(time=slice('2001','2020')).fillna(0)),
                   xr_detrend(de_ycn_gleam.where(cn_mask>0).sel(time=slice('2001','2020')).fillna(0)), dim='time')

    pr=ccrs.PlateCarree()

    # Begin plotting
    fig = plt.figure(figsize=[8, 6])
    # panel a,b 
    ax1 = fig.add_axes([0, 0, 1, 1], projection=pr)
    im = plot_map(r_e_p,ax=ax1, levels=np.arange(-1,1.01,0.1), cmap='RdBu_r',
                  lw=2,extent=[73, 137, 15, 53])

    (p_map.where(p_map>0.05)).plot.contourf(hatches='////', colors='none', add_colorbar=False,ax=ax1)

    ax1.text(0.5,0.9,'$r$(P, E)',transform=ax1.transAxes, fontsize=12,ha='center')
    
    set_lat_lon(ax1, range(80,130,20), range(30,52,10), label=True, pad=0.05, fontsize=10)
    
    # add colorbar
    cbarbig1_pos = [ax1.get_position().x0, ax1.get_position().y0-0.03, ax1.get_position().width, 0.015]
    caxbig1 = fig.add_axes(cbarbig1_pos)
    cb = plt.colorbar(im, orientation="horizontal", pad=0.15,cax=caxbig1,extend='neither',
                     ticks=np.arange(-1,1.01,0.2))
    cb.set_label(label='Correlation')

    plt.savefig('../figure/fig_prec_e_corr_map_%s0129.png'%et_data,dpi=300,bbox_inches='tight')
    print('Fig saved')

if __name__=="__main__":
    make_plot(et_data='ERA5')
