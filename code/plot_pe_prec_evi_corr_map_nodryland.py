import seaborn as sns
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cartopy.crs as ccrs
from plot_prec_comparison import plot_map
from plot_china_dryland_source import set_lat_lon
from process_et2prec import load_era5_data,load_e2p_data,load_evi_data
from plot_pe_prec_corr_map import xr_detrend,xr_stats_corr
from plot_pe_evi_corr import ds_month_range

def make_plot(et_data='GLEAM'):
    start_year=2003
    end_year=2022
    source = 'aqua'#'aqua' 'terra'

    #Load data
    ai=xr.open_dataset('../data/AI_1901-2017_360x720.nc') #aridity
    ai_con=ai.Band1>0

    dp_ycn=load_era5_data('prec','y',cn_label=True)
    # precipitation contribution from all source grids
    dep_ycn_gleam=load_e2p_data('e_to_prec_land','y',start_year=start_year,end_year=end_year,et_data=et_data)
    # precipitation contribution only from dryland grids
    dep_ycn_cndry=load_e2p_data('e_to_prec_cndry','y',start_year=start_year,end_year=end_year)
    # precipitation contribution excluding dryland grids
    dep_ycn_nocndry=dep_ycn_gleam - dep_ycn_cndry 
    
    # growing season EVI
    evi_gs = ds_month_range(load_evi_data(temporal_res='mon',source=source))
    dp_ycn_gs=ds_month_range(load_era5_data('prec','mon',cn_label=True))
    # precipitation contribution from all source grids
    dep_ycn_gleam_gs=ds_month_range(load_e2p_data('e_to_prec_land','mon',end_year=end_year,et_data=et_data))

    # precipitation contribution only from dryland grids
    dep_ycn_cndry_gs=ds_month_range(load_e2p_data('e_to_prec_cndry','mon',start_year=start_year,end_year=end_year))
    # precipitation contribution excluding dryland grids
    dep_ycn_nocndry_gs=dep_ycn_gleam_gs - dep_ycn_cndry_gs
    
    # corr between prec con from China's drylands and downwind prec 
    [r_pe_p_cndry,p_map_cndry]=xr_stats_corr(xr_detrend(dp_ycn.where(ai_con).sel(time=slice(str(start_year),str(end_year))).fillna(0)),
                   xr_detrend(dep_ycn_cndry.where(ai_con).sel(time=slice(str(start_year),str(end_year))).fillna(0)), dim='time')

    # corr between prec con from upwind external dryland and downwind prec
    [r_pe_p_nocndry,p_map_nocndry]=xr_stats_corr(xr_detrend(dep_ycn_nocndry.where(ai_con).sel(time=slice(str(start_year),str(end_year))).fillna(0)), 
                   xr_detrend(dp_ycn.where(ai_con).sel(time=slice(str(start_year),str(end_year))).fillna(0)), dim='time')
    r_pe_p_diffcndry = r_pe_p_cndry-r_pe_p_nocndry
    print(r_pe_p_diffcndry.max())
    print(r_pe_p_diffcndry.min())
    r_dry=np.array([(r_pe_p_diffcndry>0).sum().values, (r_pe_p_diffcndry<0).sum().values])
    r_pe_p_diff_ratio=r_dry/r_dry.sum()


    # Grid corr between prec con from dryland and evi 
    [r_map_dry,p_temp]=xr_stats_corr(xr_detrend(evi_gs.where(ai_con).sel(time=slice(str(start_year),str(end_year))).fillna(0)),
                   xr_detrend(dep_ycn_cndry_gs.where(ai_con).sel(time=slice(str(start_year),str(end_year))).fillna(0)), dim='time')

    # Grid corr between prec con from external and evi 
    [r_map_nodry,p_temp]=xr_stats_corr(xr_detrend(evi_gs.where(ai_con).sel(time=slice(str(start_year),str(end_year))).fillna(0)),
                   xr_detrend(dep_ycn_nocndry_gs.where(ai_con).sel(time=slice(str(start_year),str(end_year))).fillna(0)), dim='time')
    r_map_diffdry=r_map_dry - r_map_nodry
    print(r_map_diffdry.max())
    print(r_map_diffdry.min())
    r_dry=np.array([(r_map_diffdry>0).sum().values, (r_map_diffdry<0).sum().values])
    r_map_diff_ratio=r_dry/r_dry.sum()
    
    pr=ccrs.PlateCarree()

    # Begin plotting
    fig = plt.figure(figsize=[10, 8])
    # panel a 
    ax1 = fig.add_axes([0, 0.7,0.5, 0.3], projection=pr)
    im1 = plot_map(r_pe_p_diffcndry.where(ai_con),ax=ax1, levels=np.arange(-1.25,1.26,0.25), cmap='RdBu_r',
                  extent=[73, 128, 28, 50])

    ax1.text(0.5,0.9,'$r$(P$_\mathrm{E}$(dryland), P) - $r$(P$_\mathrm{E}$(external), P)',transform=ax1.transAxes,fontsize=12,ha='center')
    ax1.text(0.4,0.75,'dryland:%d%%\n(>0)'%(np.round(r_pe_p_diff_ratio[0]*100)),
            transform=ax1.transAxes,fontsize=10,ha='center')
    ax1.text(0.6,0.75,'external:%d%%\n(<0)'%(np.round(r_pe_p_diff_ratio[1]*100)),
            transform=ax1.transAxes,fontsize=10,ha='center')
    
    # panel b
    ax2 = fig.add_axes([0, 0.35, 0.5, 0.3], projection=pr)
    plot_map(r_map_diffdry.where(ai_con),ax=ax2, levels=np.arange(-1.25,1.26,0.25), cmap='RdBu_r',
             extent=[73, 128, 28, 50])

    ax2.text(0.5,0.9,'$r$(P$_\mathrm{E}$(dryland), EVI) - $r$(P$_\mathrm{E}$(external), EVI)',transform=ax2.transAxes,fontsize=12,ha='center')
    ax2.text(0.4,0.75,'dryland:%d%%\n(>0)'%(np.round(r_map_diff_ratio[0]*100)),
            transform=ax2.transAxes,fontsize=10,ha='center')
    ax2.text(0.6,0.75,'external:%d%%\n(<0)'%(np.round(r_map_diff_ratio[1]*100)),
            transform=ax2.transAxes,fontsize=10,ha='center')
    
    set_lat_lon(ax1, range(80,130,20), range(30,52,10), label=True, pad=0.05, fontsize=10)
    set_lat_lon(ax2, range(80,130,20), range(30,52,10), label=True, pad=0.05, fontsize=10)
    
    # add colorbar
    cbarbig1_pos = [ax2.get_position().x0, ax2.get_position().y0-0.03, ax2.get_position().width, 0.015]
    caxbig1 = fig.add_axes(cbarbig1_pos)
    cb = plt.colorbar(im1, orientation="horizontal", pad=0.15,cax=caxbig1,extend='neither',
                     ticks=np.arange(-1.25,1.26,0.25))
    cb.set_label(label='Correlation differences')
    
    ax1.text(-0.05, 1.05, 'a', fontsize=14, transform=ax1.transAxes, fontweight='bold')
    ax2.text(-0.05, 1.05, 'b', fontsize=14, transform=ax2.transAxes, fontweight='bold')

    plt.savefig('../figure/fig_pe_prec_evi_corr_map_%s_nodryland0617.png'%et_data,dpi=300,bbox_inches='tight')
    print('Fig saved')

if __name__=="__main__":
    make_plot(et_data='ERA5')
