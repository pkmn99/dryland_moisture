import xarray as xr
import numpy as np
import pandas as pd

# convert int8 to valid data values
def conversion_value(x):
    return np.exp(x*-0.1)

# get lat and lon for every grid within the mask
def convert_grid_index(df,n,var='Band1'):
    # tip from https://stackoverflow.com/questions/40592630/get-coordinates-of-non-nan-values-of-xarray-dataset#
    # ma3=d.moisture_flow[:,:,0,0].where(tb_mask==1) # use the 0,0 grid as target region to produce source mask
    ma_stacked = df[var].where(df[var]==n).stack(grid=['lat','lon'])
    myindex=ma_stacked[~np.isnan(ma_stacked)] # retain only valid grids
    temp=myindex.grid.to_pandas().reset_index().iloc[:,0:2]
    temp.loc[:,'id']=n
    return temp

# Save grid index of all provinces
def save_province_grid_index():
    d=xr.open_dataset('../data/China_province_360x720.nc')
    pro_list=pd.read_csv('../data/china_province_list.csv')
    # combine results of different provinces
    temp = pd.concat([convert_grid_index(d,n) for n in pro_list['id']])
    temp.merge(pro_list).to_csv('../data/china_province_grid_index.csv',index=None)
    print('file saved')

"""
Calculate ET contribution to total precipitation
"""
def save_prec_contribution(source_region='TP',et_data='GLEAM_v3.5a',var='E'):
    subregion_index=pd.read_csv('../data/china_province_grid_index.csv')
    # Use gleam ET
    dfe = xr.open_dataset('../data/E_2008-2017_%s_ymonmean_360x720_clean.nc'%et_data)['E']
    n_lat, n_lon = dfe.shape[1::]
    # regional id 
    region_id=subregion_index['id'].unique()
    
    # var to save precipitation contribution; dim: region, month, lat, lon
    pre= np.zeros([region_id.shape[0],12,n_lat,n_lon])

    for m in range(12):
        # load moisture data for month m
        dfm = xr.open_dataset('../data/utrack_climatology_0.5_%02d.nc'%(m+1))

        # Loop through provinces
        for i in range(region_id.shape[0]):
            region_ma = subregion_index['id']==region_id[i]
            # Select moisture flow from source region defined by grid index, process region by region to save memory
            # When selecting by pairs of lat,lon values, one have to add xr.DataArray， otherwise it retures n_lat*n_lon selection
            temp_dfm= conversion_value(dfm.moisture_flow.sel(sourcelat=xr.DataArray(subregion_index[region_ma].lat.values+0.25),
                                        sourcelon=xr.DataArray(subregion_index[region_ma].lon.values+0.25)))

            # Normalize to 1 by dividing sum of all fractions
            temp_dfm = temp_dfm / temp_dfm.sum(dim=['targetlat','targetlon'])

            # extract ET values for all grids at month i, dim = (13xx,)
            temp_et = dfe.sel(month=m+1,
                              lat=xr.DataArray(subregion_index.loc[region_ma,'lat'].values),
                              lon=xr.DataArray(subregion_index.loc[region_ma,'lon'].values))

            # Prec contribution = Multiple dfm (dim: grid, lat, lon) by 1d ET (dim: grid) 
            temp_p=temp_dfm * temp_et
            # Sum prec contribution of all grids in a subregion
            pre[i,m,:,:] = temp_p.sum(axis=0)
            print('region %d'%region_id[i])

        print('month %d completed'%(m+1))
    
    # create data array to save results
    dpre=xr.DataArray(pre, coords=[region_id, range(1,13), dfe.lat, dfe.lon],
                   dims=['region','month','lat','lon'],
                   name='pre_con',attrs={'unit':'mm'})

    dpre.to_netcdf('../data/results/prec_con_mon_sourceprovince_%s.nc'%et_data)
    print('prec contribution file saved')

"""
Calculate source region of precipitation
"""
def save_prec_source(source_region='TP',et_data='GLEAM_v3.5a',var='E'):
    subregion_index=pd.read_csv('../data/china_province_grid_index.csv')

    # Use ERA5 ET as it covers the ocean
    dfe = xr.open_dataset('../../data/ERA5/ERA5-ET-2008-2017-ymonmean-05deg-clean.nc')['E']

    # Use ERA5 prec to get actual precipitation at target grids
   # dp = xr.open_dataset('../data/prec_2008-2017_ERA5_ymonmean_360x720_clean.nc')['prec']
    dp = xr.open_dataset('../../data/ERA5/ERA5-prec-2008-2017-ymonmean-05deg-clean.nc')['prec']

    n_lat, n_lon = dfe.shape[1::]
    # regional id 
    region_id=subregion_index['id'].unique()
    
    # var to save precipitation source; dim: region, month, lat, lon
    pre= np.zeros([region_id.shape[0],12,n_lat,n_lon])

    for m in range(12):
        # load moisture data for month m
        dfm = xr.open_dataset('../data/utrack_climatology_0.5_%02d.nc'%(m+1))

        # Loop through provinces
        for i in range(region_id.shape[0]):
            region_ma = subregion_index['id']==region_id[i]
            # Select moisture flow from source region defined by grid index, process region by region to save memory
            # here select target grids by their lat,lon belongint to reigon id
            temp_dfm= conversion_value(dfm.moisture_flow.sel(targetlat=xr.DataArray(subregion_index[region_ma].lat.values+0.25),
                                targetlon=xr.DataArray(subregion_index[region_ma].lon.values+0.25)))

            # change axis to allow for broadcast; targets grids (1st dim) muptiple ET to get precipitation contribution from 
            # different grids (2nd, 3rd dims)
            temp_p=np.moveaxis(temp_dfm.values, -1, 0) * dfe[m].values
            
            # normalize to get the fraction of target precipitation from different source grids
#            # move axis to enable broadcast and convert back after calculation
#            temp_f= np.moveaxis(np.moveaxis(temp_p,0,-1)/temp_p.sum(axis=(1,2)),-1,0)   
            # convert to xarray to enable broadcast
            temp_f= xr.DataArray(temp_p)/xr.DataArray(temp_p.sum(axis=(1,2)))   
            
            # extract prec values for target grids 
            temp_prec = dp.sel(month=m+1,
                              lat=xr.DataArray(subregion_index.loc[region_ma,'lat'].values),
                              lon=xr.DataArray(subregion_index.loc[region_ma,'lon'].values))

            # Actual prec contribution = Multiple fraction (dim: grid, lat, lon) by 1d prec (dim: grid) 
            temp_p2=temp_f * temp_prec

            # Sum prec contribution of all target grids of a region
            pre[i,m,:,:] = temp_p2.sum(axis=0)
            print('region %d of %d completed'%(i,region_id.shape[0]))

        print('month %d completed'%(m+1))
    
    # create data array to save results
    dpre=xr.DataArray(pre, coords=[region_id, range(1,13), dfe.lat, dfe.lon],
                   dims=['region','month','lat','lon'],
                   name='pre_con',attrs={'unit':'mm'})

    dpre.to_netcdf('../data/results/prec_con_mon_targetprovince0601.nc')
    print('prec contribution file saved')


if __name__=="__main__":
#    save_province_grid_index()
#    et_data='GLEAM_v3.5a'
#    et_data='PML'
#    save_prec_contribution(et_data='ERA5')
    save_prec_source()
