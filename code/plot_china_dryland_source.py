import xarray as xr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from plot_prec_comparison import plot_map,my_weights
from process_et2prec import load_era5_data,load_e2p_data
"""
A map figure to show moisture source of China drylands
"""

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

def make_plot(et_data='ERA5'):
    # set time range 
    start_year=2003
    end_year=2022

    # Load data 
    # Moisture source
    s = xr.open_dataset('../data/results/china_dryland_prec_source_aridity.nc')
    # precipitation contribution from default ERA5
    dep_ymonmeancn=load_e2p_data('e_to_prec','ymonmean',start_year=start_year,end_year=end_year)# prec con by ERA all ET
    dep_ymonmeancn_land=load_e2p_data('e_to_prec_land','ymonmean',start_year=start_year,end_year=end_year)# prec con by ERA land ET
    dep_ymonmeancn_cndry=load_e2p_data('e_to_prec_cndry','ymonmean',start_year=start_year,end_year=end_year)# prec con by ERA in China dryland
    if et_data=='GLEAM':
        dep_ymonmeancn_gleam=load_e2p_data('e_to_prec','ymonmean',start_year=start_year,end_year=end_year,et_data='GLEAM')# prec con by gleam ET
        dep_ymonmeancn_gleam_cndry=load_e2p_data('e_to_prec_cndry','ymonmean',start_year=start_year,end_year=end_year,et_data='GLEAM')# prec con by gleam ET
    else:
        dep_ymonmeancn_gleam=dep_ymonmeancn_land
        dep_ymonmeancn_gleam_cndry=dep_ymonmeancn_cndry

    dp_ymonmeancn=load_era5_data('prec','ymonmean',start_year=start_year,end_year=end_year,cn_label=True)# prec by ERA5

    w=my_weights(dp_ymonmeancn) # Grid area weights
    
    ai=xr.open_dataset('../data/AI_1901-2017_360x720.nc')# aridity data
    ai_con=ai.Band1>0 # select all dryland 

    # calculate land-derived ratio of prec for each grid based on ERA5 data
    ratio = dep_ymonmeancn_land.sum(dim='month')/dep_ymonmeancn.sum(dim='month')
    # calculate china dryland-derived ratio of prec based on ERA5 data
    # ratio2 = dep_ymonmeancn_cndry.sum(dim='month')/dep_ymonmeancn.sum(dim='month')

    r_land=dep_ymonmeancn_land.sum(dim='month').where(ai_con).weighted(w).mean()/dep_ymonmeancn.sum(dim='month').where(ai_con).weighted(w).mean()
    r_cndry=dep_ymonmeancn_cndry.sum(dim='month').where(ai_con).weighted(w).mean()/dep_ymonmeancn.sum(dim='month').where(ai_con).weighted(w).mean()
    print('land contribution to dryland precipitatin is %f'%r_land.values)
    print('dryland contribution to their own precipitatin is %f'%r_cndry.values)

#    print('monthly contribution from land moisture')
#    print(dep_ymonmeancn_land.where(ai_con).weighted(w).mean(['lat','lon']))
#    print('monthly contribution from dryland moisture')
#    print(dep_ymonmeancn_cndry.where(ai_con).weighted(w).mean(['lat','lon']))
#    print('monthly contribution from all moisture')
#    print(dep_ymonmeancn.where(ai_con).weighted(w).mean(['lat','lon']))

    # Temporary for review
#    plt.plot((dep_ymonmeancn_land-dep_ymonmeancn_cndry).where(ai_con).weighted(w).mean(['lat','lon']))
#    plt.plot(dep_ymonmeancn_cndry.where(ai_con).weighted(w).mean(['lat','lon']))
#    plt.plot(dep_ymonmeancn.where(ai_con).weighted(w).mean(['lat','lon']))
#    plt.xlabel('Month')
#    plt.ylabel('PE (mm)')
#    plt.legend([r'PE_external',r'PE_dryland','PE_all'])
#    plt.show()

     
    levels1=[1,10,50,100,200,300,1000] # for moistutre source map
#    levels2=[0.001,0.01,0.1,0.5,1,2]

    # define map projection
    pr=ccrs.PlateCarree()

    fig = plt.figure(figsize=[8, 12])

    ax1 = fig.add_axes([0, 0.5, 1, 0.5], projection=pr)
    
    dtemp=s.e_to_prec.sum(dim=['aridity','month'])
    im1=plot_map(dtemp, ax=ax1, levels=levels1, cmap='Blues',extent=[0, 150, 0, 70])

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.set_title('Moisture sources of China\'s drylands', fontsize=14)

    set_lat_lon(ax1, range(0,151,30), range(10,71,20), label=True, pad=0.05, fontsize=10)

    # Add colorbar to big plot
    cbarbig1_pos = [ax1.get_position().x0, ax1.get_position().y0-0.02, ax1.get_position().width, 0.01]
    caxbig1 = fig.add_axes(cbarbig1_pos)
    cb = plt.colorbar(im1, orientation="horizontal", pad=0.05,cax=caxbig1,extend='min',
                     ticks=levels1)
    cb.set_label(label='Precipitation contributed by moisture sources (mm/yr)',fontsize=12)


    #Add panel a inset to show aridity levels
     # Add customed colobar for Reds
    Reds = mpl.colormaps['Reds'].resampled(8)
    cmap_ai = ListedColormap(Reds(np.linspace(0.2, 1, 4))) # Start from 0.2 for a darker color

    ax1in = fig.add_axes([0, 0.485, 0.35, 0.25], projection=pr)
    im=ai.Band1.plot(cmap=cmap_ai,ax=ax1in,add_colorbar=False)
    ax1in.set_extent([73, 128, 29, 49],ccrs.Geodetic())

    cbarax1in_pos = [ax1in.get_position().x1+0.01, ax1in.get_position().y0, 0.01, ax1in.get_position().height]
    caxax1in = fig.add_axes(cbarax1in_pos)
    cb = plt.colorbar(im, orientation="vertical", pad=0.15, cax=caxax1in, extend='neither',
                     ticks=np.arange(1+3/4/2,4+3/4/2,3/4)) # Label 1 to 4 has 3  for length; divide 3/4 is tick interval, further divide by 2 for half shift
    cb.ax.set_yticklabels(['Dry subhumid','Semiarid','Arid','Hyperarid'], fontsize=10)
    ax1in.text(0.5, 0.8, 'Aridity', fontsize=12, transform=ax1in.transAxes, ha='center')


    ############## ax2 to ax5 
    ax2 = fig.add_axes([0, 0.235, 0.45, 0.35], projection=pr)
    im2=plot_map(ratio.where(ai_con),ax=ax2, levels=np.arange(0,1.01,0.1), cmap='OrRd',extent=[73, 128, 28, 50]) #'OrRd''YlOrBr'
    
    ### ax3 Panel B: bar 
    ratio_mon = (dep_ymonmeancn_land.where(ai_con).weighted(w).mean(dim=['lat','lon'])/dep_ymonmeancn.where(ai_con).weighted(w).mean(dim=['lat','lon']))
    print(ratio_mon)
    ratio2_mon = (dep_ymonmeancn_cndry.where(ai_con).weighted(w).mean(dim=['lat','lon'])/dep_ymonmeancn.where(ai_con).weighted(w).mean(dim=['lat','lon']))

    ax3 = fig.add_axes([0.55, ax2.get_position().y0, 0.45, ax2.get_position().height])
    ax3.bar(range(1,13,),ratio_mon,color=sns.color_palette()[3])
    ax3.bar(range(1,13,),ratio2_mon,color=sns.color_palette()[1])
    ax3.set_xlabel('Month',fontsize=12)
    ax3.set_ylabel('Terrestial source fraction')
    ax3.legend(['External','Dryland'],frameon=False,handletextpad=0.5,ncols=2,loc=[0.2,0.85],fontsize=9)
    ax3.set_ylim([0,1.15])
    
    # ax4 Panel C: map
    ax4 = fig.add_axes([0, 0.025, 0.45, 0.35], projection=pr)
    im3=plot_map(dep_ymonmeancn_gleam.sum(dim='month').where(ai_con),ax=ax4, levels=np.arange(0,1000,100),
             cmap='Blues',extent=[73, 128, 28, 50])
    
    # ax5 Panel D: line chart
    ax5 = fig.add_axes([0.55, ax4.get_position().y0, 0.45, ax4.get_position().height])
    (dep_ymonmeancn_gleam.where(ai_con).weighted(w).mean(dim=['lat','lon'])).plot(ax=ax5)
    (dep_ymonmeancn_gleam_cndry.where(ai_con).weighted(w).mean(dim=['lat','lon'])).plot(ax=ax5)
    (dp_ymonmeancn.where(ai_con).weighted(w).mean(dim=['lat','lon'])).plot(ax=ax5)
    ax5.legend(['P$_\mathrm{E}$','P$_\mathrm{E}$(dryland)','P'],frameon=False,handlelength=1)
    ax5.set_ylabel('Precipitation (mm/yr)')
    ax5.set_xlabel('Month',fontsize=12)

    # add lat lon tick
    set_lat_lon(ax2, range(80,130,20), range(30,52,10), label=True, pad=0.05, fontsize=10)
    set_lat_lon(ax4, range(80,130,20), range(30,52,10), label=True, pad=0.05, fontsize=10)

    # add colorbar
    cbarbig2_pos = [ax2.get_position().x0, ax2.get_position().y0-0.01, ax2.get_position().width, 0.01]
    caxbig2 = fig.add_axes(cbarbig2_pos)
    cb = plt.colorbar(im2, orientation="horizontal", pad=0.15,cax=caxbig2,extend='neither',
                     ticks=np.arange(0,1.01,0.2))
    cb.set_label(label='Terrestial source fraction', fontsize=12)

    cbarbig3_pos = [ax4.get_position().x0, ax4.get_position().y0-0.01, ax4.get_position().width, 0.01]
    caxbig3 = fig.add_axes(cbarbig3_pos)
    cb = plt.colorbar(im3, orientation="horizontal", pad=0.15,cax=caxbig3,extend='neither',
                     ticks=np.arange(0,1000,100))
    cb.set_label(label='P$_\mathrm{E}$ (mm/yr)',fontsize=12)
    
    ax1.text(-0.05, 1.05, 'a', fontsize=14, transform=ax1.transAxes, fontweight='bold')
    ax2.text(-0.1, 1.05, 'b', fontsize=14, transform=ax2.transAxes, fontweight='bold')
    ax2.text(-0.1, 1.05, 'c', fontsize=14, transform=ax3.transAxes, fontweight='bold')
    ax2.text(-0.1, 1.05, 'd', fontsize=14, transform=ax4.transAxes, fontweight='bold')
    ax2.text(-0.1, 1.05, 'e', fontsize=14, transform=ax5.transAxes, fontweight='bold')

    plt.savefig('../figure/fig_china_dryland_prec_source_0614.tif',dpi=300,bbox_inches='tight')
    print('figure saved')

# report internal precipitation contribution from drylands itself
def print_dryland_ratio():
    lsm=xr.open_dataset('../../data/ERA5/ERA5-landseamask-1990-2022-mon-05deg.nc')['lsm'].squeeze()
    land_mask=lsm>0.5
    s = xr.open_dataset('../data/results/china_dryland_prec_source_aridity.nc')
    ai=xr.open_dataset('../data/AI_1901-2017_360x720.nc')# aridity data
    # create area weights following 
    # https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html

    weights = np.cos(np.deg2rad(s.lat))
    weights.name = "weights"

#    # Annual ratio
    pe_in = s.e_to_prec.sum(dim=['aridity','month']).where((ai.Band1>0)&land_mask).weighted(weights).mean().values # in drylands
    pe_out = s.e_to_prec.sum(dim=['aridity','month']).where((ai.Band1.isnull()&land_mask)).weighted(weights).mean().values# out drylands
    pe_land = s.e_to_prec.sum(dim=['aridity','month']).where((ai.Band1>0)&land_mask).weighted(weights).mean().values # in drylands
    pe_all = s.e_to_prec.sum(dim=['aridity','month']).where(ai.Band1>0).weighted(weights).mean().values# all contributions
    print('terestrial contribution to precipitation is %f'%(pe_land/pe_all))
    print('dryland terestrial internal contribution to precipitation is %f'%(pe_in/(pe_in+pe_out)))
    print('dryland terestrial external contribution to precipitation is %f'%(pe_out/(pe_in+pe_out)))
    print('dryland terestrial contribution to precipitation is %f'%(pe_in/pe_all))

#    # Monthly ratio
#    pe_in = s.e_to_prec.sum(dim=['aridity']).where((ai.Band1>0)&(land_mask)).weighted(weights).mean(['lat','lon']).values # in drylands
#    pe_out = s.e_to_prec.sum(dim=['aridity']).where((ai.Band1.isnull())&land_mask).weighted(weights).mean(['lat','lon']).values# out drylands
#    pe_all = s.e_to_prec.sum(dim=['aridity']).where(ai.Band1>0).weighted(weights).mean().values# all contributions
#    print('dryland monthly internal contribution to precipitation is ')
#    print((pe_in/(pe_in+pe_out)))
#    print('dryland monthly external contribution to precipitation is ')
#    print((pe_out/pe_all))

if __name__=="__main__":
   make_plot() 
#   print_dryland_ratio()
