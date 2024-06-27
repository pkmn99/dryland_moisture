import pandas as pd
import numpy as np

# provide aridity to csv make table data for SEM
# 0: all area; 1 to 4 means aridity levels
def make_sem_data(ai_n,start_year=2003,end_year=2022,et_data='ERA5',save_ana=True):
    import xarray as xr
    from process_et2prec import load_era5_data,load_e2p_data,load_gleam_data,load_evi_data
    from plot_pe_evi_corr import detrend

    # Load necessary data
    de_ycn_upwind_gleam=load_e2p_data('upwind_ET_land','y',start_year=start_year, end_year=end_year)
    devi_ycn_upwind_gleam=load_e2p_data('upwind_evi','y',start_year=start_year, end_year=end_year)
    dp_ycn_upwind=load_e2p_data('upwind_prec_land','y',start_year=start_year, end_year=end_year)
    dep_ycn_gleam=load_e2p_data('e_to_prec_land','y',start_year=start_year, end_year=end_year)
#    de_ycn_gleam = load_gleam_data('y',start_year=start_year, end_year=end_year,cn_label=True)
    de_ycn_gleam = load_era5_data('ET','y',start_year=start_year, end_year=end_year,cn_label=True)
    dp_ycn=load_era5_data('prec','y',start_year=start_year, end_year=end_year,cn_label=True)

#    kndvi= pd.read_csv('../data/kndvi_2001-2020_aridity.csv')

    # Load AI data
    ai=xr.open_dataset('../data/AI_1901-2017_360x720.nc')

    # create EVI by aridity for SEM models
    evi0 = load_evi_data(source='aqua',start_year=start_year,end_year=end_year)
    temp=[evi0.where(ai.Band1==i).mean(dim=['lat','lon']).values for i in range(1,5)]
    evi = pd.DataFrame(temp,columns=range(start_year,end_year+1),index=['1','2','3','4']).transpose() # pd.date_range('20010101','20231231',freq='Y')
    evi.loc[:,'0'] = evi0.where(ai.Band1>0).mean(dim=['lat','lon']).values

    if ai_n==0:
        ai_con=ai.Band1>0
    else:
        ai_con=ai.Band1==ai_n
    # save anomaly or raw data
    if save_ana:
        de_up = detrend(de_ycn_upwind_gleam.where(ai_con).mean(dim=['lat','lon'])).to_pandas().rename('de_up')
        devi_up = detrend(devi_ycn_upwind_gleam.where(ai_con).mean(dim=['lat','lon'])).to_pandas().rename('devi_up')
        dp_up = detrend(dp_ycn_upwind.where(ai_con).mean(dim=['lat','lon'])).to_pandas().rename('dp_up')
        de = detrend(de_ycn_gleam.where(ai_con).mean(dim=['lat','lon'])).to_pandas().rename('de')
        dp = detrend(dp_ycn.where(ai_con).mean(dim=['lat','lon'])).to_pandas().rename('dp')
        dep = detrend(dep_ycn_gleam.where(ai_con).mean(dim=['lat','lon'])).to_pandas().rename('dep')
    else:
        de_up = de_ycn_upwind_gleam.where(ai_con).mean(dim=['lat','lon']).to_pandas().rename('de_up')
        devi_up = devi_ycn_upwind_gleam.where(ai_con).mean(dim=['lat','lon']).to_pandas().rename('devi_up')
        dp_up = dp_ycn_upwind.where(ai_con).mean(dim=['lat','lon']).to_pandas().rename('dp_up')
        de = de_ycn_gleam.where(ai_con).mean(dim=['lat','lon']).to_pandas().rename('de')
        dp = dp_ycn.where(ai_con).mean(dim=['lat','lon']).to_pandas().rename('dp')
        dep = dep_ycn_gleam.where(ai_con).mean(dim=['lat','lon']).to_pandas().rename('dep')

    # combine variables
    dall= pd.concat([dp_up.rename('dp_up'),de_up.rename('de_up'),devi_up.rename('devi_up'),dep.rename('dep'),
                     dp.rename('dp'),de.rename('de')],axis=1)

    if save_ana:
        dall['evi']=detrend(evi[str(ai_n)].values)
        dall.to_csv('../data/results/data_ana_for_sem_%d-%d_aridity%d.csv'%(start_year,end_year,ai_n))
    else:
        dall['evi']=evi[str(ai_n)].values
        dall.to_csv('../data/results/data_for_sem_%d-%d_aridity%d.csv'%(start_year,end_year,ai_n))
    print(dall.mean())
    print('csv file for aridity %d saved'%ai_n)


# Running this func requires sem environment
def do_sem(ai_n,fig_on=False,start_year=2003,end_year=2022):
    import semopy
    from scipy import stats
    from semopy import Model
    mydata= pd.read_csv('../data/results/data_ana_for_sem_%d-%d_aridity%d.csv'%(start_year,end_year,ai_n),index_col=0)
    print(mydata.mean())

## standadize data by max and min    
#    def min_max_scaling(series):
#        return (series - series.min()) / (series.max() - series.min())
#    
#    for col in mydata.columns:
#        mydata[col] = min_max_scaling(mydata[col])
    
#    mod = """ de_up ~ dp_up
#              eta_mr ~ de_up + dp_up
#              dp ~ dep
#              de ~ dp + evi
#              evi ~ dp
#              eta_mr =~ dep
#          """
    
    #mod = """ de_up ~ dp_up
    #          dep ~ de_up
    #          dp ~ dep
    #          de ~ dep
    #          evi ~ dep
    #      """

############The used sem model form
#    mod = """ de_up ~ dp_up
#              dep ~ de_up
#              dp ~ dep
#              de ~ dp + evi
#              evi ~ dp
##              evi ~~ de
#          """
    mod = """ de_up ~ dp_up + devi_up
              devi_up ~ dp_up
              dep ~ de_up
              dp ~ dep
              de ~ dp + evi
              evi ~ dp
          """
    
#    mod = """ de_up ~ dp_up
#              dep ~ de_up + dp_up
#              dp ~ dep + dp_up
#    #          dp ~ dep
#              de ~ dp + evi 
#              evi ~ dp
#    #          evi ~~ de
#    #          dp_up ~~ dp
#    #          de_up ~~ de
#          """
    
#    mod = """ de_up ~ dp_up
#              dep ~ de_up + dp_up
#   #           dep ~ de_up
#              dp ~ dep
#              de ~ dp + evi
#              evi ~ dp
#    #          evi ~~ de
#    #          dp_up ~~ dp
#    #          de_up ~~ de
#          """
    
    model = Model(mod)
    #model.fit(mydata,obj='GLS')
    model.fit(stats.zscore(mydata))
    print('report summary for aridity %d'%ai_n)
    print(model.inspect())
    print(semopy.calc_stats(model).T)
    # save model fig or not
    if fig_on:
        g = semopy.semplot(model, "pd_aridity%d0129.png"%ai_n)


if __name__=="__main__":
#   make sem data 
#    [make_sem_data(i,save_ana=False) for i in range(5)]
#    [make_sem_data(i) for i in range(5)]
#    make_sem_data(0,save_ana=False)
#
#    do_sem(1)
#    do_sem(2)
#    do_sem(3)
#    do_sem(4)
    do_sem(0,fig_on=False)
