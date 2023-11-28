import seaborn as sns
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from plot_prec_comparison import plot_map
from plot_china_dryland_source import set_lat_lon
from plot_pe_prec_corr_map import load_e2p_data,load_gleam_data
from process_et2prec import load_era5_data

def make_plot():
    # set time range 
    start_year=2001
    end_year=2020

    # Load data
    dep_ymonmeancn=load_e2p_data('e_to_prec','ymonmean',start_year=start_year,end_year=end_year)# prec con by ERA all ET
    dep_ymonmeancn_land=load_e2p_data('e_to_prec_land','ymonmean',start_year=start_year,end_year=end_year)# prec con by ERA land ET
    dep_ymonmeancn_cndry=load_e2p_data('e_to_prec_cndry','ymonmean',start_year=start_year,end_year=end_year)# prec con by ERA in China dryland
    dep_ymonmeancn_gleam=load_e2p_data('e_to_prec','ymonmean',start_year=start_year,end_year=end_year,et_data='GLEAM')# prec con by gleam ET
    dep_ymonmeancn_gleam_cndry=load_e2p_data('e_to_prec_cndry','ymonmean',start_year=start_year,end_year=end_year,et_data='GLEAM')# prec con by gleam ET
    dp_ymonmeancn=load_era5_data('prec','ymonmean',start_year=start_year,end_year=end_year,cn_label=True)# prec by ERA5
    
    ai=xr.open_dataset('../data/AI_1901-2017_360x720.nc')# aridity data
    ai_con=ai.Band1>0 # select all dryland 

    # calculate land-derived ratio of prec based on ERA5 data
    ratio = dep_ymonmeancn_land.sum(dim='month')/dep_ymonmeancn.sum(dim='month')
    # calculate china dryland-derived ratio of prec based on ERA5 data
    ratio2 = dep_ymonmeancn_cndry.sum(dim='month')/dep_ymonmeancn.sum(dim='month')

    levels=[0.001,0.01,0.1,0.5,1,2]
    
    pr=ccrs.PlateCarree()
    
    # Begin plot
    fig = plt.figure(figsize=[10, 10])
    # Panel A: map 
    ax1 = fig.add_axes([0, 0.5, 0.4, 0.35], projection=pr)
    im1=plot_map(ratio.where(ai_con),ax=ax1, levels=np.arange(0,1.01,0.1), cmap='OrRd',extent=[73, 128, 28, 50]) #'OrRd''YlOrBr'
    
    # Panel B: bar 
    ratio_mon = (dep_ymonmeancn_land.where(ai_con).mean(dim=['lat','lon'])/dep_ymonmeancn.where(ai_con).mean(dim=['lat','lon']))
    ratio2_mon = (dep_ymonmeancn_cndry.where(ai_con).mean(dim=['lat','lon'])/dep_ymonmeancn.where(ai_con).mean(dim=['lat','lon']))
    ax2 = fig.add_axes([0.5, ax1.get_position().y0, 0.3, ax1.get_position().height])
    #ratio_mon.to_dataframe('A').plot.bar(ax=ax2)
    #ratio_mon.to_dataframe('A').plot.bar(ax=ax2,legend=False,color=sns.color_palette()[3])
    ax2.bar(range(1,13,),ratio_mon,color=sns.color_palette()[3])
    ax2.bar(range(1,13,),ratio2_mon,color=sns.color_palette()[1])
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Precipitation source fraction')
    ax2.legend(['Land','Dryland'],frameon=False,handletextpad=0.5,ncols=2,loc=[0.2,0.85],fontsize=9)
    ax2.set_ylim([0,1.15])
    
    # Panel C: map
    ax3 = fig.add_axes([0, 0.24, 0.4, 0.35], projection=pr)
    im3=plot_map(dep_ymonmeancn_gleam.sum(dim='month').where(ai_con),ax=ax3, levels=np.arange(0,1000,100),
             cmap='Blues',extent=[73, 128, 28, 50])
    
    # Panel D: line chart
    ax4 = fig.add_axes([0.5, ax3.get_position().y0, 0.3, ax3.get_position().height])
    (dep_ymonmeancn_gleam.where(ai_con).mean(dim=['lat','lon'])).plot(ax=ax4)
    (dep_ymonmeancn_gleam_cndry.where(ai_con).mean(dim=['lat','lon'])).plot(ax=ax4)
    (dp_ymonmeancn.where(ai_con).mean(dim=['lat','lon'])).plot(ax=ax4)
    ax4.legend(['P$_\mathrm{E}$','P$_\mathrm{E}$(dryland)','P'],frameon=False,handlelength=1)
    ax4.set_ylabel('Precipitation (mm/yr)')
    ax4.set_xlabel('Month')

    # add lat lon tick
    set_lat_lon(ax1, range(80,130,20), range(30,52,10), label=True, pad=0.05, fontsize=10)
    set_lat_lon(ax3, range(80,130,20), range(30,52,10), label=True, pad=0.05, fontsize=10)

    # add colorbar
    cbarbig1_pos = [ax1.get_position().x0, ax1.get_position().y0-0.01, ax1.get_position().width, 0.01]
    caxbig1 = fig.add_axes(cbarbig1_pos)
    cb = plt.colorbar(im1, orientation="horizontal", pad=0.15,cax=caxbig1,extend='neither',
                     ticks=np.arange(0,1.01,0.2))
    cb.set_label(label='Land source fraction')

    cbarbig3_pos = [ax3.get_position().x0, ax3.get_position().y0-0.01, ax3.get_position().width, 0.01]
    caxbig3 = fig.add_axes(cbarbig3_pos)
    cb = plt.colorbar(im3, orientation="horizontal", pad=0.15,cax=caxbig3,extend='neither',
                     ticks=np.arange(0,1000,100))
    cb.set_label(label='P$_\mathrm{E}$ (mm/yr)')
    
    ax1.text(-0.05, 1.05, 'a', fontsize=14, transform=ax1.transAxes, fontweight='bold')
    ax2.text(-0.15, 1.05, 'b', fontsize=14, transform=ax2.transAxes, fontweight='bold')
    ax3.text(-0.05, 1.05, 'c', fontsize=14, transform=ax3.transAxes, fontweight='bold')
    ax4.text(-0.15, 1.05, 'd', fontsize=14, transform=ax4.transAxes, fontweight='bold')

    plt.savefig('../figure/fig_landratio_pe_map1127.png',dpi=300,bbox_inches='tight')
    print('Fig saved')

if __name__=="__main__":
    make_plot()
