import seaborn as sns
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from plot_pe_prec_corr_map import load_era5_data,load_e2p_data
from plot_pe_kndvi_corr import stats_corr,detrend

# time series by aridity

def make_plot():
    c=0
    start_year=2001
    end_year=2020
    ai_label=['Dry subhumid','Semiarid','Arid','Hyperarid']

    # Load data
    ai=xr.open_dataset('../data/AI_1901-2017_360x720.nc') #
    dep_ycn_gleam=load_e2p_data('e_to_prec','y',end_year=2020,et_data='GLEAM')
    dp_ycn=load_era5_data('prec','y',cn_label=True)

    # Calculate correlation
    rpep_temp = [stats_corr(detrend(dep_ycn_gleam.sel(time=slice(str(start_year),str(end_year))).where(ai.Band1==i).mean(dim=['lat','lon'])).values,
                           detrend(dp_ycn.sel(time=slice(str(start_year),str(end_year))).where(ai.Band1==i).mean(dim=['lat','lon']).values)) for i in range(1,5)]
   # # P and KNDVI
   # rp_temp = [stats_corr(detrend(dp_ycn.sel(time=slice(str(start_year),str(end_year))).where(ai.Band1==i).mean(dim=['lat','lon'])).values,
   #             detrend(kndvi.set_index('Year').loc[slice(start_year,end_year)][str(i)].values*10000)) for i in range(1,5)]

    R_array = pd.DataFrame(np.array(rpep_temp),columns=['R','p'])
    R_array.loc[0:4,'var']='PEP'
    R_array.loc[:,'Aridity']=list(range(1,5))

    # Begin plotting 
    fig, axes=plt.subplots(2,2,figsize=(10,8))

    for i in range(4):
        # Panel A 
        axes.flatten()[i].plot(range(start_year,end_year+1),detrend(dep_ycn_gleam.sel(time=slice(str(start_year),str(end_year))).where(ai.Band1==i+1).mean(dim=['lat','lon']).values),'b')
        axes.flatten()[i].plot(range(start_year,end_year+1),detrend(dp_ycn.sel(time=slice(str(start_year),str(end_year))).where(ai.Band1==i+1).mean(dim=['lat','lon']).values),'orange')
        
        [r,p]=stats_corr(detrend(dep_ycn_gleam.sel(time=slice(str(start_year),str(end_year))).where(ai.Band1==i+1).mean(dim=['lat','lon'])).values,
                         detrend(dp_ycn.sel(time=slice(str(start_year),str(end_year))).where(ai.Band1==i+1).mean(dim=['lat','lon'])).values)

        axes.flatten()[i].text(0.5,0.85,'r=%.2f\np=%.2f'%(r,p),transform=axes.flatten()[i].transAxes)
        axes.flatten()[i].set_ylabel('Precipitation anomaly (mm/yr)',labelpad=0.5)
        axes.flatten()[i].set_xticks(range(2001,2021,3))
        axes.flatten()[i].legend(['P$_\mathrm{E}$','P'],frameon=False)
        axes.flatten()[i].set_title(ai_label[i])

#    ax1.text(-0.125, 1.05, 'a', fontsize=14, transform=ax1.transAxes, fontweight='bold')
#    ax2.text(-0.125, 1.05, 'b', fontsize=14, transform=ax2.transAxes, fontweight='bold')

#    plt.subplots_adjust(left=0.05,right=0.95,wspace=0.2)
    plt.savefig('../figure/fig_pe_prec_corr_aridity.png',dpi=300,bbox_inches='tight')
    print('Fig saved')

if __name__=="__main__":
    make_plot()
