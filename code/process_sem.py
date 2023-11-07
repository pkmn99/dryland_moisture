import pandas as pd
import numpy as np

# provide aridity to csv make table data for SEM
# 0: all area; 1 to 4 means aridity levels
def make_sem_data(ai_n):
    import xarray as xr
    from process_et2prec import load_era5_data,load_e2p_data,load_gleam_data
    from plot_pe_kndvi_corr import detrend

    # Load necessary data
    de_ycn_upwind_gleam=load_e2p_data('upwind_ET','y',start_year=2001, end_year=2020,et_data='GLEAM')
    dp_ycn_upwind=load_e2p_data('upwind_prec','y',start_year=2001, end_year=2020)
    dep_ycn_gleam=load_e2p_data('e_to_prec','y',start_year=2001, end_year=2020)
    de_ycn_gleam = load_gleam_data('y',start_year=2001, end_year=2020,cn_label=True)
    dp_ycn=load_era5_data('prec','y',start_year=2001, end_year=2020,cn_label=True)
    kndvi= pd.read_csv('../data/kndvi_2001-2020_aridity.csv')

    # Load AI data
    ai=xr.open_dataset('../data/AI_1901-2017_360x720.nc')

    if ai_n==0:
        ai_con=ai.Band1>0
    else:
        ai_con=ai.Band1==ai_n

    de_up = detrend(de_ycn_upwind_gleam.where(ai_con).mean(dim=['lat','lon'])).to_pandas().rename('de_up')
    dp_up = detrend(dp_ycn_upwind.where(ai_con).mean(dim=['lat','lon'])).to_pandas().rename('dp_up')
    de = detrend(de_ycn_gleam.where(ai_con).mean(dim=['lat','lon'])).to_pandas().rename('de')
    dp = detrend(dp_ycn.where(ai_con).mean(dim=['lat','lon'])).to_pandas().rename('dp')
    dep = detrend(dep_ycn_gleam.where(ai_con).mean(dim=['lat','lon'])).to_pandas().rename('dep')

    # combine variables
    dall= pd.concat([dp_up.rename('dp_up'),de_up.rename('de_up'),dep.rename('dep'),dp.rename('dp'),de.rename('de')],axis=1)

    dall['kndvi']=detrend(kndvi[str(ai_n)].values)

    dall.to_csv('../data/results/data_ana_for_sem_2001-2020_aridity%d.csv'%ai_n)
    print('csv file for aridity %d saved'%ai_n)


# Running this func requires sem environment
def do_sem(ai_n,fig_on=False):
    import semopy
    from scipy import stats
    from semopy import Model
    mydata= pd.read_csv('../data/results/data_ana_for_sem_2001-2020_aridity%d.csv'%ai_n,index_col=0)

## standadize data by max and min    
#    def min_max_scaling(series):
#        return (series - series.min()) / (series.max() - series.min())
#    
#    for col in mydata.columns:
#        mydata[col] = min_max_scaling(mydata[col])
    
#    mod = """ de_up ~ dp_up
#              eta_mr ~ de_up + dp_up
#              dp ~ dep
#              de ~ dp + kndvi
#              kndvi ~ dp
#              eta_mr =~ dep
#          """
    
    #mod = """ de_up ~ dp_up
    #          dep ~ de_up
    #          dp ~ dep
    #          de ~ dep
    #          kndvi ~ dep
    #      """

############The used sem model form
    mod = """ de_up ~ dp_up
              dep ~ de_up
              dp ~ dep
              de ~ dp + kndvi
              kndvi ~ dp
#              kndvi ~~ de
          """
    
#    mod = """ de_up ~ dp_up
#              dep ~ de_up + dp_up
#              dp ~ dep + dp_up
#    #          dp ~ dep
#              de ~ dp + kndvi 
#              kndvi ~ dp
#    #          kndvi ~~ de
#    #          dp_up ~~ dp
#    #          de_up ~~ de
#          """
    
#    mod = """ de_up ~ dp_up
#              dep ~ de_up + dp_up
#   #           dep ~ de_up
#              dp ~ dep
#              de ~ dp + kndvi
#              kndvi ~ dp
#    #          kndvi ~~ de
#    #          dp_up ~~ dp
#    #          de_up ~~ de
#          """
    
    model = Model(mod)
    #model.fit(mydata,obj='GLS')
    model.fit(stats.zscore(mydata))
    print(model.inspect())
    print(semopy.calc_stats(model).T)
    # save model fig or not
    if fig_on:
        g = semopy.semplot(model, "pd_aridity%d.png"%ai_n)


if __name__=="__main__":
#   make sem data 
#    [make_sem_data(i) for i in range(5)]

#    do_sem(1)
#    do_sem(2)
#    do_sem(3)
#    do_sem(4)
    do_sem(0,fig_on=True)
