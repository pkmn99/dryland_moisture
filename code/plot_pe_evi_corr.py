import seaborn as sns
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cartopy.crs as ccrs
from plot_prec_comparison import plot_map
from process_et2prec import load_era5_data,load_e2p_data,load_evi_data
from plot_pe_prec_corr_map import xr_detrend,xr_stats_corr
from plot_china_dryland_source import set_lat_lon

# Calculate correlation along time dim, and return R and P value
def stats_corr(x,y,corr_type='pearson'):#'pearson'
    corr_method={"kendall":stats.kendalltau,
                "pearson":stats.pearsonr,
                "spearman":stats.spearmanr}
    return corr_method[corr_type](x,y)

# detrend time series
def detrend(y,deg=1):
    b = np.polyfit(range(y.shape[0]),y,deg=deg)
    return (y-np.polyval(b,range(y.shape[0])))

# extract specific month range from monthly data array
def ds_month_range(ds,mon1=5,mon2=9):
    con=(ds.time.dt.month>=mon1)&(ds.time.dt.month<=mon2)
    return ds[con].resample(time='Y').mean()

# growing season defined as from May to Sept
def make_plot(et_data='GLEAM',p_data='pe',grow_season=True):
    start_year=2003
    end_year=2022
    source = 'aqua'#'aqua' 'terra'

    pr=ccrs.PlateCarree()

    # Load data
    if grow_season:
        evi = ds_month_range(load_evi_data(temporal_res='mon',source=source))
        dp_ycn=ds_month_range(load_era5_data('prec','mon',cn_label=True))
        dep_ycn_gleam=ds_month_range(load_e2p_data('e_to_prec','mon',end_year=end_year,et_data=et_data))

    else:
        evi = load_evi_data(source=source) # * 10 for terra
        dep_ycn_gleam=load_e2p_data('e_to_prec','y',end_year=end_year,et_data=et_data)
        dp_ycn=load_era5_data('prec','y',cn_label=True)
   #    evi = load_evi_data(temporal_res='y',source='aqua')

    ai=xr.open_dataset('../data/AI_1901-2017_360x720.nc') #aridity

#    if (p_data=="prec")&grow_season:
#        dep_ycn_gleam=ds_month_range(load_era5_data(p_data,'mon',end_year=end_year,cn_label=True))
#    elif p_data=="prec":
#        dep_ycn_gleam=load_era5_data('prec','y',end_year=end_year,cn_label=True)

    ai_con=ai.Band1>0

    # Grid corr between prec and evi 
    [r_map0,p_map0]=xr_stats_corr(xr_detrend(evi.where(ai_con).sel(time=slice(str(start_year),str(end_year))).fillna(0)),
                   xr_detrend(dp_ycn.where(ai_con).sel(time=slice(str(start_year),str(end_year))).fillna(0)), dim='time')

    # Grid corr between prec con and evi 
    [r_map,p_map]=xr_stats_corr(xr_detrend(evi.where(ai_con).sel(time=slice(str(start_year),str(end_year))).fillna(0)),
                   xr_detrend(dep_ycn_gleam.where(ai_con).sel(time=slice(str(start_year),str(end_year))).fillna(0)), dim='time')
    p_con = p_map<0.05 #<0.1 # significance threshlod to denote regions where evi is sensitive to precipitation

    # Calculate correlation
    # P_E and EVI
    rpe_temp = [stats_corr(detrend(dep_ycn_gleam.sel(time=slice(str(start_year),str(end_year))).where((ai.Band1==i)&(p_con)).mean(dim=['lat','lon'])).values,
                detrend(evi.sel(time=slice(str(start_year),str(end_year))).where((ai.Band1==i)&(p_con)).mean(dim=['lat','lon']).values*10000)) for i in range(1,5)]

    # P and KNDVI
    rp_temp = [stats_corr(detrend(dp_ycn.sel(time=slice(str(start_year),str(end_year))).where((ai.Band1==i)&(p_con)).mean(dim=['lat','lon'])).values,
               detrend(evi.sel(time=slice(str(start_year),str(end_year))).where((ai.Band1==i)&(p_con)).mean(dim=['lat','lon']).values*10000)) for i in range(1,5)]
    R_array = pd.DataFrame(np.concatenate([np.array(rpe_temp),np.array(rp_temp)]),columns=['R','p'])
    R_array.loc[0:4,'var']='PE'
    R_array.loc[4::,'var']='P'
    R_array.loc[:,'Aridity']=list(range(1,5))*2


    # Begin plotting 
    fig = plt.figure(figsize=[8,8])
    ax1 = fig.add_axes([0, 0.55, 0.425, 0.30])
    ax2 = fig.add_axes([0.6, 0.55, 0.425, 0.30])
    ax3 = fig.add_axes([0.05, 0, 0.9, 0.5], projection=pr)
    ax3in = fig.add_axes([0.375, 0.33, 0.3, 0.125], projection=pr)

    # Panel A 
    ax1.plot(range(start_year,end_year+1),detrend(dep_ycn_gleam.sel(time=slice(str(start_year),str(end_year))).where(ai_con).mean(dim=['lat','lon']).values),'b')
    
    ax1sec = ax1.twinx()
    ax1sec.plot(range(start_year,end_year+1),
                detrend(evi.sel(time=slice(str(start_year),str(end_year))).where(ai_con&p_con).mean(dim=['lat','lon']).values*10000),'g')
    
    [r,p]=stats_corr(detrend(dep_ycn_gleam.sel(time=slice(str(start_year),str(end_year))).where(ai_con&p_con).mean(dim=['lat','lon'])).values,
                     detrend(evi.sel(time=slice(str(start_year),str(end_year))).where(ai_con&p_con).mean(dim=['lat','lon']).values*10000))
    
    ax1.text(0.1,0.1,'r=%.2f\np=%.2f'%(r,p),transform=ax1.transAxes)
    ax1.set_ylabel('P$_\mathrm{E}$(mm/yr)',color='b',labelpad=0.5)
    ax1.set_xticks(range(2003,2022,3))
    ax1sec.set_ylabel('EVI*10000',color='g',labelpad=1)
    ax1.legend([ax1.lines[0],ax1sec.lines[0]],['P$_\mathrm{E}$','EVI'],
            frameon=False,loc='upper left')

    # Panel B 
    sns.barplot(x='Aridity',y='R',data=R_array,hue='var',ax=ax2)
    ax2.set_ylabel('Correlation',labelpad=1)
    ax2.set_xlabel('')
    ax2.set_ylim([0,0.95])
    ax2.set_xticklabels(['Dry subhumid','Semiarid','Arid','Hyperarid'])
    
    # Add label for p value, double ** for p <0.05 and single * for p< 0.1
    for i in range(8):
        if R_array['p'][i]<0.05:
            ax2.text(ax2.patches[i].get_x()+ax2.patches[0].get_width()/2 , R_array['R'][i]+0.025,'**',
                    ha='center',color='k')
        if (R_array['p'][i]<0.1)&(R_array['p'][i]>0.05):
            ax2.text(ax2.patches[i].get_x()+ax2.patches[0].get_width()/2 , R_array['R'][i]+0.025,'*',
                    ha='center',color='k')
    
    ax2.legend([ax2.patches[0],ax2.patches[4]],['$r$(P$_\mathrm{E}$, EVI)','$r$(P, EVI)'],frameon=False)


    # Panel C
    im = plot_map(r_map.where(p_map0<0.05),ax=ax3, levels=np.arange(-1,1.01,0.1), cmap='RdBu_r',
                  extent=[73, 128, 28, 50])
    set_lat_lon(ax3, range(80,130,20), range(30,52,10), label=True, pad=0.05, fontsize=10)

    ax3.text(0.15,0.9,'$r$(P$_\mathrm{E}$, EVI)',transform=ax3.transAxes,fontsize=12,ha='center')

    im2 = plot_map(r_map0.where(p_map0<0.1),ax=ax3in, levels=np.arange(-1,1.01,0.1), cmap='RdBu_r',
                  extent=[73, 128, 28, 50])
    ax3in.text(0.2,0.85,'$r$(P, EVI)',transform=ax3in.transAxes,fontsize=12,ha='center')
#    set_lat_lon(ax3, range(80,130,20), range(30,52,10), label=True, pad=0.05, fontsize=10)

    # add colorbar
    cbarbig1_pos = [ax3.get_position().x0, ax3.get_position().y0-0.03, ax3.get_position().width, 0.015]
    caxbig1 = fig.add_axes(cbarbig1_pos)
    cb = plt.colorbar(im, orientation="horizontal", pad=0.15,cax=caxbig1,extend='neither',
                     ticks=np.arange(-1,1.01,0.2))
    cb.set_label(label='Correlation')

    ax1.text(-0.12, 1.05, 'a', fontsize=14, transform=ax1.transAxes, fontweight='bold')
    ax2.text(-0.12, 1.05, 'b', fontsize=14, transform=ax2.transAxes, fontweight='bold')
    ax3.text(-0.05, 1.05, 'c', fontsize=14, transform=ax3.transAxes, fontweight='bold')

    plt.savefig('../figure/fig_pe_evi_corr_%s0129.png'%et_data,dpi=300,bbox_inches='tight')
    print('Fig saved')

if __name__=="__main__":
#    make_plot(et_data='GLEAM')
    make_plot(et_data='ERA5',grow_season=True)
