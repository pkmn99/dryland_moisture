import seaborn as sns
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from plot_pe_prec_corr_map import load_era5_data,load_e2p_data 

# Calculate correlation along time dim, and return R and P value
def stats_corr(x,y,corr_type='pearson'):
    corr_method={"kendall":stats.kendalltau,
                "pearson":stats.pearsonr,
                "spearman":stats.spearmanr}
    return corr_method[corr_type](x,y)

# detrend time series
def detrend(y):
    b = np.polyfit(range(y.shape[0]),y,deg=1)
    return (y-np.polyval(b,range(y.shape[0])))

def make_plot():
    c=0
    start_year=2001
    end_year=2020

    # Load data
    kndvi= pd.read_csv('../data/kndvi_2001-2020_aridity.csv')
    ai=xr.open_dataset('../data/AI_1901-2017_360x720.nc') #
    dep_ycn_gleam=load_e2p_data('e_to_prec','y',end_year=2020,et_data='GLEAM')
    dp_ycn=load_era5_data('prec','y',cn_label=True)

    # Calculate correlation
    # P_E and KNDVI
    rpe_temp = [stats_corr(detrend(dep_ycn_gleam.sel(time=slice(str(start_year),str(end_year))).where(ai.Band1==i).mean(dim=['lat','lon'])).values,
                detrend(kndvi.set_index('Year').loc[slice(start_year,end_year)][str(i)].values*10000)) for i in range(1,5)]
    # P and KNDVI
    rp_temp = [stats_corr(detrend(dp_ycn.sel(time=slice(str(start_year),str(end_year))).where(ai.Band1==i).mean(dim=['lat','lon'])).values,
                detrend(kndvi.set_index('Year').loc[slice(start_year,end_year)][str(i)].values*10000)) for i in range(1,5)]
    R_array = pd.DataFrame(np.concatenate([np.array(rpe_temp),np.array(rp_temp)]),columns=['R','p'])
    R_array.loc[0:4,'var']='PE'
    R_array.loc[4::,'var']='P'
    R_array.loc[:,'Aridity']=list(range(1,5))*2


    # Begin plotting 
    fig, [ax1,ax2]=plt.subplots(1,2,figsize=(10,4))

    # Panel A 
    ax1.plot(range(start_year,end_year+1),detrend(dep_ycn_gleam.sel(time=slice(str(start_year),str(end_year))).where(ai.Band1>0).mean(dim=['lat','lon']).values),'b')
    
    ax1sec = ax1.twinx()
    ax1sec.plot(range(start_year,end_year+1),detrend(kndvi.set_index('Year').loc[slice(start_year,end_year)][str(c)].values*1000),'g')
    
    [r,p]=stats_corr(detrend(dep_ycn_gleam.sel(time=slice(str(start_year),str(end_year))).where(ai.Band1>0).mean(dim=['lat','lon'])).values,
                detrend(kndvi.set_index('Year').loc[slice(start_year,end_year)][str(c)].values))
    
    ax1.text(0.15,0.15,'r=%.2f\np=%.2f'%(r,p),transform=ax1.transAxes)
    ax1.set_ylabel('P$_\mathrm{E}$(mm/yr)',color='b',labelpad=0.5)
    ax1.set_xticks(range(2001,2021,3))
    ax1sec.set_ylabel('KNDVI*10000',color='g',labelpad=1)
    ax1.legend(['P$_\mathrm{E}$','P'],frameon=False)
    ax1.legend([ax1.lines[0],ax1sec.lines[0]],['P$_\mathrm{E}$','KNDVI'],frameon=False)


    # Panel B 
    sns.barplot(x='Aridity',y='R',data=R_array,hue='var',ax=ax2)
    ax2.set_ylabel('Corelation with KNDVI',labelpad=1)
    ax2.set_xlabel('')
    ax2.set_ylim([-0.4,0.7])
    ax2.set_xticklabels(['Dry subhumid','Semiarid','Arid','Hyperarid'])
    
    # Add label for p value
    for i in range(8):
        if R_array['p'][i]<0.05:
            ax2.text(ax2.patches[i].get_x()+ax2.patches[0].get_width()/2 , R_array['R'][i]+0.025,'**',
                    ha='center',color='k')
        if (R_array['p'][i]<0.1)&(R_array['p'][i]>0.05):
            ax2.text(ax2.patches[i].get_x()+ax2.patches[0].get_width()/2 , R_array['R'][i]+0.025,'*',
                    ha='center',color='k')
    
    ax2.legend([ax2.patches[0],ax2.patches[4]],['P$_\mathrm{E}$','P'],frameon=False)

    ax1.text(-0.075, 1.05, 'a', fontsize=14, transform=ax1.transAxes, fontweight='bold')
    ax2.text(-0.075, 1.05, 'b', fontsize=14, transform=ax2.transAxes, fontweight='bold')

    plt.subplots_adjust(left=0.05,right=0.95,wspace=0.3)
    plt.savefig('../figure/fig_pe_kndvi_corr.png',dpi=300,bbox_inches='tight')
    print('Fig saved')

if __name__=="__main__":
    make_plot()
