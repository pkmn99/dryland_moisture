import seaborn as sns
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cartopy.crs as ccrs
from plot_prec_comparison import plot_map
from plot_china_dryland_source import set_lat_lon
from process_et2prec import load_era5_data,load_gleam_data,load_e2p_data

# linear detrend for 3d xarray variable along time dim
def xr_detrend(df):
    return df - xr.polyval(coord=df.time,
                           coeffs=df.polyfit(dim="time", deg=1).polyfit_coefficients)

# Calculate correlation for 3d xarray var along time dim, and return R and P value
# Main ref from https://www.martinjung.eu/post/2018_xarrayregression/
# For two output vars: https://github.com/pydata/xarray/discussions/7845
def xr_stats_corr(x,y,dim='year',corr_type='pearson'):
    corr_method={"kendall":stats.kendalltau,
                "pearson":stats.pearsonr,
                "spearman":stats.spearmanr}
    return xr.apply_ufunc(
           corr_method[corr_type],x,y,
           input_core_dims=[[dim], [dim]],
           output_core_dims=[[], []],
           vectorize=True # !Important!
        )

# concate spatial corr values with aridity
# the column name ai_n could be 1 to 4 for aridity, or text 
def concat_r_ai(r,ai,ai_n):
    if isinstance(ai_n,int):
        R = r.where(ai.Band1==ai_n).stack(grid=['lat','lon']).dropna(dim='grid').to_pandas().rename('R').reset_index()
        R['AI']=ai_n
    if isinstance(ai_n,str):
        # ai_n is not 1 to 4, it could be text to represent var string
        R = r.where(ai.Band1>0).stack(grid=['lat','lon']).dropna(dim='grid').to_pandas().rename('R').reset_index()
        R['var']=ai_n
    return R

# convert correlation map to pandas with aridity levels for boxplot in seaborn
def convert_map_by_aridity(r_map,ai):
    temp = [concat_r_ai(r_map,ai,i) for i in range(1,5)]
    return pd.concat(temp).set_index(['lat','lon'])

def make_plot(et_data='GLEAM'):
    start_year=2003
    end_year=2022

    #Load data
    ai=xr.open_dataset('../data/AI_1901-2017_360x720.nc') #aridity
    de_ycn_gleam = load_gleam_data('y') # GLEAM ET China
    dep_ycn_gleam=load_e2p_data('e_to_prec_land','y',end_year=end_year,et_data=et_data)
    dp_ycn=load_era5_data('prec','y',cn_label=True)

    de_ycn_upwind_gleam=load_e2p_data('upwind_ET','y',end_year=end_year,et_data=et_data)
    dp_ycn_upwind_land=load_e2p_data('upwind_prec_land','y')

    ai_con=ai.Band1>0
    
    # corr between prec con and downwind prec 
    [r_pe_p,p_map1]=xr_stats_corr(xr_detrend(dp_ycn.where(ai_con).sel(time=slice(str(start_year),str(end_year))).fillna(0)),
                   xr_detrend(dep_ycn_gleam.where(ai_con).sel(time=slice(str(start_year),str(end_year))).fillna(0)), dim='time')
    # corr between upwind prec and downwind prec
    [r_pu_p,p_map2]=xr_stats_corr(xr_detrend(dp_ycn_upwind_land.where(ai_con).sel(time=slice(str(start_year),str(end_year))).fillna(0)), 
                   xr_detrend(dp_ycn.where(ai_con).sel(time=slice(str(start_year),str(end_year))).fillna(0)), dim='time')
    # corr between upwind ET and downwind prec
    [r_eu_p,p_map3]=xr_stats_corr(xr_detrend(de_ycn_upwind_gleam.where(ai_con).sel(time=slice(str(start_year),str(end_year))).fillna(0)), 
                   xr_detrend(dp_ycn.where(ai_con).sel(time=slice(str(start_year),str(end_year))).fillna(0)), dim='time')
    
    R_pe_p = convert_map_by_aridity(r_pe_p,ai)
    R_eu_p = convert_map_by_aridity(r_eu_p,ai)
    R_pu_p = convert_map_by_aridity(r_pu_p,ai)

    print('correlation of P and Pe by aridity')
    print(R_pe_p.groupby('AI').median())

    pr=ccrs.PlateCarree()

    # Begin plotting
    fig = plt.figure(figsize=[10, 8])
    # panel a,b 
    ax1 = fig.add_axes([0, 0.7, 0.5, 0.3], projection=pr)
    im = plot_map(r_pe_p.where(ai_con),ax=ax1, levels=np.arange(-1,1.01,0.1), cmap='RdBu_r',
                  extent=[73, 128, 28, 50])
    (p_map1.where(p_map1>0.05)).plot.contourf(hatches='////', colors='none', add_colorbar=False,ax=ax1)
    print('Prec and Pe corr, median over the region is %f'%r_pe_p.where(ai_con).median().values)
    print('Insignificant fraction is %f for map 1'%((p_map1>0.05).sum()/(p_map1>0).sum().values))

    ax1.text(0.5,0.9,'$r$(P$_\mathrm{E}$, P)',transform=ax1.transAxes,fontsize=12,ha='center')
    
    # switch green and yellow for sns colormap
    my_palette=[sns.color_palette()[0],sns.color_palette()[2],sns.color_palette()[1],sns.color_palette()[3]]
    
    # panel b 
    ax1in = fig.add_axes([0.6, ax1.get_position().y0, 0.35, ax1.get_position().height])
    sns.boxplot(x="AI", y="R", data=R_pe_p, showfliers=False,ax=ax1in,
               palette=my_palette)
    ax1in.set_ylim([-0.8,1.01])
    ax1in.set_yticks(np.arange(-0.8,1.01,0.4))
    ax1in.set_xticklabels(['Dry subhumid','Semiarid','Arid','HyperArid'])
    # ax1in.set_xticklabels(['0.65~0.5','0.5~0.2','0.2~0.05','<0.05'])
    ax1in.set_xlabel('')
    ax1in.set_ylabel('Correlation')
    ax1in.set_title('$r$(P$_\mathrm{E}$, P)')
    
    # panel c,d
    ax2 = fig.add_axes([0, 0.35, 0.5, 0.3], projection=pr)
    plot_map(r_pu_p.where(ai_con),ax=ax2, levels=np.arange(-1,1.01,0.1), cmap='RdBu_r',
             extent=[73, 128, 28, 50])
    (p_map2.where(p_map2>0.05)).plot.contourf(hatches='////', colors='none', add_colorbar=False,ax=ax2)
    ax2.text(0.5,0.9,'$r$(P$_\mathrm{up}$, P)',transform=ax2.transAxes,fontsize=12,ha='center')
    
    ax2in = fig.add_axes([0.6, ax2.get_position().y0, 0.35, ax2.get_position().height])
    sns.boxplot(x="AI", y="R", data=R_pu_p, showfliers=False,ax=ax2in,
               palette=my_palette)
    ax2in.set_ylim([-0.8,1.01])
    ax2in.set_yticks(np.arange(-0.8,1.01,0.4))
    ax2in.set_xticklabels(['Dry subhumid','Semiarid','Arid','Hyperarid'])
    ax2in.set_ylabel('Correlation')
    ax2in.set_xlabel('')
    ax2in.set_title('$r$(P$_\mathrm{up}$, P)')
    print('Insignificant fraction is %f for map 2'%((p_map2>0.05).sum()/(p_map2>0).sum().values))
    
    # panel e,f 
    ax3 = fig.add_axes([0, 0, 0.5, 0.3], projection=pr)
    im=plot_map(r_eu_p.where(ai_con),ax=ax3, levels=np.arange(-1,1.01,0.1), cmap='RdBu_r',
                extent=[73, 128, 28, 50])
    (p_map3.where(p_map3>0.05)).plot.contourf(hatches='////', colors='none', add_colorbar=False,ax=ax3)
    ax3.text(0.5,0.9,'$r$(E$_\mathrm{up}$, P)',transform=ax3.transAxes,fontsize=12,ha='center')
    
    ax3in = fig.add_axes([0.6, ax3.get_position().y0, 0.35, ax3.get_position().height])
    sns.boxplot(x="AI", y="R", data=R_eu_p, showfliers=False,ax=ax3in,
               palette=my_palette)
    ax3in.set_ylim([-0.8,1.01])
    ax3in.set_yticks(np.arange(-0.8,1.01,0.4))
    ax3in.set_xticklabels(['Dry subhumid','Semiarid','Arid','HyperArid'])
    ax3in.set_ylabel('Correlation')
    ax3in.set_xlabel('')
    ax3in.set_title('$r$(E$_\mathrm{up}$, P)')
    print('Insignificant fraction is %f for map 3'%((p_map3>0.05).sum()/(p_map3>0).sum().values))
    
    set_lat_lon(ax1, range(80,130,20), range(30,52,10), label=True, pad=0.05, fontsize=10)
    set_lat_lon(ax2, range(80,130,20), range(30,52,10), label=True, pad=0.05, fontsize=10)
    set_lat_lon(ax3, range(80,130,20), range(30,52,10), label=True, pad=0.05, fontsize=10)
    
    # add colorbar
    cbarbig1_pos = [ax3.get_position().x0, ax3.get_position().y0-0.03, ax3.get_position().width, 0.015]
    caxbig1 = fig.add_axes(cbarbig1_pos)
    cb = plt.colorbar(im, orientation="horizontal", pad=0.15,cax=caxbig1,extend='neither',
                     ticks=np.arange(-1,1.01,0.2))
    cb.set_label(label='Correlation')
    
    ax1.text(-0.05, 1.05, 'a', fontsize=14, transform=ax1.transAxes, fontweight='bold')
    ax2.text(-0.05, 1.05, 'c', fontsize=14, transform=ax2.transAxes, fontweight='bold')
    ax3.text(-0.05, 1.05, 'e', fontsize=14, transform=ax3.transAxes, fontweight='bold')
    
    ax1in.text(-0.075, 1.05, 'b', fontsize=14, transform=ax1in.transAxes, fontweight='bold')
    ax2in.text(-0.075, 1.05, 'd', fontsize=14, transform=ax2in.transAxes, fontweight='bold')
    ax3in.text(-0.075, 1.05, 'f', fontsize=14, transform=ax3in.transAxes, fontweight='bold')

#    plt.savefig('../figure/fig_pe_prec_corr_map_%s0129.png'%et_data,dpi=300,bbox_inches='tight')
    print('Fig saved')

if __name__=="__main__":
    make_plot(et_data='ERA5')
