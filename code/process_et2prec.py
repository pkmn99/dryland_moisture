import xarray as xr
import numpy as np
import pandas as pd
from process_utrack_data import conversion_value,convert_grid_index

# Grid index for China dryland, from 1 to 4 reprsenting more arid conditions
def save_China_dryland_grid_index():
    # Aridity
    ai=xr.open_dataset('../data/AI_1901-2017_360x720.nc')
    temp = pd.concat([convert_grid_index(ai,n) for n in range(1,5)])
    temp.to_csv('../data/china_dryland_grid_index.csv',index=None)
    print('Grid index saved')

# lat and lon grids for retangular area of China
def save_cn_label(rerun=False):
    if rerun:
       # d_ref=xr.open_dataset('../data/results/e_to_prec_ymonmean_2008-2017.nc')
        d_ref=xr.open_dataset('../data/results/ERA5_e_to_prec_mon_1990-20221127.nc')
        
       # d_ref.lat.to_pandas().reset_index()['lat'].to_csv('../data/results/cn_label_lat.csv',index=False)
       # d_ref.lon.to_pandas().reset_index()['lon'].to_csv('../data/results/cn_label_lon.csv',index=False)    
        d_ref.lat.to_netcdf('../data/results/cn_label_lat1127.nc')
        d_ref.lon.to_netcdf('../data/results/cn_label_lon1127.nc')    
        print('rerun; lat and lon label saved to file')
    else:
        lat =xr.open_dataarray('../data/results/cn_label_lat1127.nc')
        lon =xr.open_dataarray('../data/results/cn_label_lon1127.nc')
        return lat,lon

def apply_cn_label(d):
    [lat,lon]=save_cn_label()
    return d.sel(lat=lat, lon=lon, method="nearest")

# create cn_mask from provincial grids
def make_cn_mask(rerun=False):
    if rerun:
        cn_grid=pd.read_csv('../data/china_province_grid_index.csv')
        cn_mask=cn_grid.set_index(['lat','lon'])['id'].to_xarray()
        cn_mask.to_netcdf('../data/china_mask_05deg.nc')
        print('mask file saved')
    else:
        cn_mask=xr.open_dataarray('../data/china_mask_05deg.nc')
        return cn_mask

"""
# A warpper to load ERA5 data
# temporal_res: 'y' 'mon' 'ymonmean'
# spatial_res: '025deg','050deg'
# Load ERA5 data with various options
"""
def load_era5_data(var,temporal_res,spatial_res='05deg',start_year=1990,end_year=2022,cn_label=False):
    file_daterange=[1990,2022]
    d = xr.open_dataset('../../data/ERA5/ERA5-%s-%d-%d-%s-%s.nc'%(var,file_daterange[0],file_daterange[1],
                                                                  'mon',spatial_res)).sel(time=slice(str(start_year),str(end_year)))
    var_text={'prec':'tp','ET':'e','temp':'t2m','solar':'msdwswrf'}
    var_unit_scaler={'prec':1000,'ET':-1000}

    # do unit conversion
    if temporal_res=='y':
        if ((var=='prec') or (var=='ET')):
            d=(d[var_text[var]]*d.time.dt.daysinmonth).resample(time="Y").sum(dim=['time'])*var_unit_scaler[var]
        else:
            d=d[var_text[var]].resample(time="Y").mean(dim=['time'])

    if temporal_res=='ymonmean':
        if ((var=='prec') or (var=='ET')):
            d=(d[var_text[var]]*d.time.dt.daysinmonth).groupby("time.month").mean(dim=['time'])*var_unit_scaler[var]
        else:
            d=d[var_text[var]].groupby("time.month").mean(dim=['time'])

    if temporal_res=='mon':
        if ((var=='prec') or (var=='ET')):
            d=(d[var_text[var]]*d.time.dt.daysinmonth)*var_unit_scaler[var]
        else:
            d=d[var_text[var]]

    # if China subset
    if cn_label:
#        d_ref=xr.open_dataset('../data/results/e_to_prec_mon_2008-2017.nc')
#        return d.sel(lat=d_ref.lat, lon=d_ref.lon, method="nearest")
        return apply_cn_label(d)
    else:
        return d

def load_evi_data(var='EVI',temporal_res="y",start_year=2003,end_year=2022,cn_label=True,source='aqua'):
    if source=='aqua':
        d=xr.open_dataset('../data/EVI_mon_05deg_2003_2022_clean.nc')#.slice(time=slice(str(start_year),str(end_year)))
    if source=='terra':
        d = xr.open_dataset('../../data/MODIS_EVI/terra_EVI_2001_2023/MOD13A3_EVI_01_23_05deg_north.nc')

    if temporal_res=='ymonmean':
        d=d[var].groupby("time.month").mean(dim=['time'])

    if temporal_res=='y':
        d=d[var].resample(time="Y").mean(dim=['time'])

    if temporal_res=='ymax':
        d=d[var].resample(time="Y").max(dim=['time'])

    if temporal_res=='mon':
        d=d[var]

    # if China subset
    if cn_label:
        return apply_cn_label(d)
    else:
        return d

# Only yearly
def load_gpp_data(temporal_res="y",start_year=2003,end_year=2018,cn_label=True):
    d=xr.open_dataset('../../data/PML/PML_GPP_20032018_05deg.nc')['GPP_year'].sel(time=slice(str(start_year),str(end_year)))
    d['time'] = pd.date_range('20030101','20181231',freq='A') # change time index to end of year to match other data

    # if China subset
    if cn_label:
        return apply_cn_label(d)
    else:
        return d

# Load precipitation contribution from ET (P_E)
def load_e2p_data(var,temporal_res,start_year=1990,end_year=2022,et_data='ERA5'):
   # file_daterange={"GLEAM":[1990,2020],
    file_daterange={"GLEAM":[2002,2022],
                    "ERA5":[2003,2022]}
    d = xr.open_dataset('../data/results/%s_e_to_prec_mon_%d-%d.nc'%(et_data,file_daterange[et_data][0],
                                                                         file_daterange[et_data][1]))
    # aggregate to annual sum
    if temporal_res=='y':
        d=d[var].sel(time=slice(str(start_year),str(end_year))).resample(time="Y").sum(dim=['time'])
    # multi-year monthly mean
    if temporal_res=='ymonmean':
        d=d[var].sel(time=slice(str(start_year),str(end_year))).groupby("time.month").mean(dim=['time'])
    # monthly
    if temporal_res=='mon':
        d=d[var].sel(time=slice(str(start_year),str(end_year)))
    return d

# Load gleam et data
def load_gleam_data(temporal_res,spatial_res='05deg',start_year=2002,end_year=2020,cn_label=True):
    d = xr.open_dataset('../data/GLEAM_v4.1a-ET-2002-2023-mon-05deg.nc')['E'].sel(time=slice(str(start_year),str(end_year)))

    if temporal_res=='y':
        d=d.sel(time=slice(str(start_year),str(end_year))).resample(time="Y").sum(dim=['time'])

    if temporal_res=='ymonmean':
        d=d.sel(time=slice(str(start_year),str(end_year))).groupby("time.month").mean(dim=['time'])

    # if China subset
    if cn_label:
        return apply_cn_label(d)
    else:
        return d

# Remove linear trend based on xarray polyfit

"""
Calculate ET-derived precipitation of a target grid by using upwind ET source
"""
def save_e_to_prec(start_year=1990,end_year=2022,et_data='ERA5'):
    subregion_index=pd.read_csv('../data/china_province_grid_index.csv')

    # Use ERA5 ET as it covers the ocean
   # dfe = xr.open_dataset('../../data/ERA5/ERA5-ET-2008-2017-mon-05deg.nc')['e']
   # dfe = xr.open_dataset('../../data/ERA5/ERA5-ET-%d-%d-mon-05deg.nc'%(start_year,end_year))['e']
   # dfe=(dfe*dfe.time.dt.daysinmonth)*1000*-1

    # Be careful with ai variable with numpy array. It requires np.flipud to align with other data north as first row
    ai=xr.open_dataset('../data/AI_1901-2017_360x720.nc')

    dp = load_era5_data('prec','mon')
    devi = load_evi_data(var='EVI',temporal_res="mon",start_year=start_year,end_year=end_year,cn_label=False)

    if et_data=="ERA5":
        dfe = load_era5_data('ET','mon')
    if et_data=="GLEAM":
       # dfe = xr.open_dataset('../data/GLEAM_v3.5a-ET-1980-2020-mon-05deg.nc')['E'].sel(time=slice(str(start_year),str(end_year)))
        dfe = xr.open_dataset('../data/GLEAM_v4.1a-ET-2002-2023-mon-05deg.nc')['E'].sel(time=slice(str(start_year),str(end_year)))
       
    # land mask fraction
    lsm=xr.open_dataset('../../data/ERA5/ERA5-landseamask-1990-2022-mon-05deg.nc')['lsm'].squeeze()

    n_lat, n_lon = dfe.shape[1::]
    # regional id 
    region_id=subregion_index['id'].unique()
    date_list = pd.date_range('%d0101'%start_year,'%d1231'%end_year,freq='MS') #MS means start of Month

    # Use cross merge to add time 
    result = subregion_index.merge(pd.DataFrame(date_list.rename('time')),how='cross')

    for m in range(12):
#    for m in range(6,7):
        # load moisture data for month m
        dfm = xr.open_dataset('../data/utrack_climatology_0.5_%02d.nc'%(m+1))
        print('procssing month %d'%(m+1))

        mon_ma = pd.Series(date_list).dt.month==(m+1)

        # Loop through provinces
        for i in range(region_id.shape[0]):
#        for i in range(1):
            region_ma = (subregion_index['id']==region_id[i])
            # Select moisture flow from source region defined by grid index, process region by region to save memory
            # here select target grids by their lat,lon belonging to reigon id
            temp_dfm= conversion_value(dfm.moisture_flow.sel(targetlat=xr.DataArray(subregion_index[region_ma].lat.values+0.25),
                                    targetlon=xr.DataArray(subregion_index[region_ma].lon.values+0.25)))
            print('processing region %d'%i)

            for j,d in enumerate(date_list[mon_ma]):
                # change axis to allow for broadcast; targets grids (1st dim) muptiple ET to get precipitation contribution from 
                # different grids (2nd, 3rd dims)
                temp_p=np.moveaxis(temp_dfm.values, -1, 0) * dfe.sel(time=d,method='nearest').values
                temp_evi=np.where(np.moveaxis(temp_dfm.values,-1,0)>10**-10, temp_p, np.nan) * devi.sel(time=d,method='nearest').values # moisture * EVI in upwind area

                # Sum to get ET-derived precipitation for target grid
                result.loc[((result['id']==region_id[i])&(result['time']==d)),'e_to_prec'] = np.nansum(temp_p,axis=(1,2))
                ## contribution from dryland itself
                result.loc[((result['id']==region_id[i])&(result['time']==d)),'e_to_prec_cndry'] =np.nansum(np.where(np.flipud(ai.Band1)>0,temp_p,0),axis=(1,2))
                result.loc[((result['id']==region_id[i])&(result['time']==d)),'upwind_ET'] = \
                    np.nanmean(np.moveaxis(np.where(temp_dfm.values>10**-10, 1, np.nan),-1,0) * dfe.sel(time=d,method='nearest').values,axis=(1,2))

                # weighted EVI
               # result.loc[((result['id']==region_id[i])&(result['time']==d)),'upwind_evi_weighted']=np.nansum(temp_evi,axis=(1,2)) \
               #         /np.nansum(np.where(temp_evi>0, np.moveaxis(temp_dfm.values, -1, 0), np.nan),axis=(1,2))
                result.loc[((result['id']==region_id[i])&(result['time']==d)),'upwind_evi_weighted']=np.nansum(temp_evi,axis=(1,2)) \
                        /np.nansum(np.where(temp_evi>0, temp_p, np.nan),axis=(1,2))

                result.loc[((result['id']==region_id[i])&(result['time']==d)),'upwind_evi']= \
                    np.nanmean(np.moveaxis(np.where(temp_dfm.values>10**-10, 1, np.nan),-1,0) * devi.sel(time=d,method='nearest').values,axis=(1,2))

                if et_data=="ERA5":
                    result.loc[((result['id']==region_id[i])&(result['time']==d)),'e_to_prec_land'] =np.where(lsm>0.5,temp_p,0).sum(axis=(1,2))
                    result.loc[((result['id']==region_id[i])&(result['time']==d)),'e_to_prec_ocean'] = np.where(lsm<0.5,temp_p,0).sum(axis=(1,2))
                    result.loc[((result['id']==region_id[i])&(result['time']==d)),'upwind_ET_ocean'] = \
                        np.nanmean(np.moveaxis(np.where(temp_dfm.values>10**-10, 1, np.nan),-1,0) * dfe.sel(time=d,method='nearest').where(lsm<0.5).values,axis=(1,2))
                    result.loc[((result['id']==region_id[i])&(result['time']==d)),'upwind_ET_land'] = \
                        np.nanmean(np.moveaxis(np.where(temp_dfm.values>10**-10, 1, np.nan),-1,0) * dfe.sel(time=d,method='nearest').where(lsm>0.5).values,axis=(1,2))
                    result.loc[((result['id']==region_id[i])&(result['time']==d)),'upwind_prec_ocean'] = \
                        np.nanmean(np.moveaxis(np.where(temp_dfm.values>10**-10, 1, np.nan),-1,0) * dp.sel(time=d,method='nearest').where(lsm<0.5).values,axis=(1,2))

                    result.loc[((result['id']==region_id[i])&(result['time']==d)),'upwind_prec'] = \
                        np.nanmean(np.moveaxis(np.where(temp_dfm.values>10**-10, 1, np.nan),-1,0) * dp.sel(time=d,method='nearest').values,axis=(1,2))
                    result.loc[((result['id']==region_id[i])&(result['time']==d)),'upwind_prec_land'] = \
                        np.nanmean(np.moveaxis(np.where(temp_dfm.values>10**-10, 1, np.nan),-1,0) * dp.sel(time=d,method='nearest').where(lsm>0.5).values,axis=(1,2))


    if et_data=="ERA5":
        result_xr = result.set_index(['time','lat','lon'])[['e_to_prec','e_to_prec_cndry','e_to_prec_land','e_to_prec_ocean',
                                                        'upwind_prec','upwind_prec_land','upwind_prec_ocean',
                                                        'upwind_ET','upwind_ET_land','upwind_ET_ocean',
                                                        'upwind_evi','upwind_evi_weighted']].to_xarray()
        result_xr['e_to_prec'].attrs["units"] = "mm"
        result_xr['e_to_prec_land'].attrs["long_name"] = "Upwind land evaporation contributed precipitation"
        result_xr['e_to_prec_land'].attrs["units"] = "mm"
        result_xr['e_to_prec_ocean'].attrs["long_name"] = "Upwind ocean evaporation contributed precipitation"
        result_xr['e_to_prec_ocean'].attrs["units"] = "mm"
        result_xr['upwind_prec'].attrs["long_name"] = "Upwind mean precipitation"
        result_xr['upwind_evi'].attrs["long_name"] = "Upwind mean evi"
        result_xr['upwind_evi_weighted'].attrs["long_name"] = "Upwind evi weighted mean by moisture"
    else:
        result_xr = result.set_index(['time','lat','lon'])[['e_to_prec','e_to_prec_cndry','upwind_ET']].to_xarray()
    
    # Common variables 
    result_xr.lat.attrs["units"] = "degrees_north"
    result_xr.lon.attrs["units"] = "degrees_west"
    result_xr.lat.attrs["long_name"] = "latitude"
    result_xr.lon.attrs["long_name"] = "longitude"
    result_xr['e_to_prec_cndry'].attrs["units"] = "mm"
    result_xr['e_to_prec_cndry'].attrs["long_name"] = "China's dryland contributed precipitation"
    result_xr['e_to_prec'].attrs["long_name"] = "Upwind evaporation contributed precipitation"
    result_xr['upwind_ET'].attrs["long_name"] = "Upwind mean ET"

    result_xr.to_netcdf('../data/results/%s_e_to_prec_mon_%d-%d0805.nc'%(et_data,start_year,end_year))

    print('file saved')

def save_e_to_prec_ymonmean():
    subregion_index=pd.read_csv('../data/china_province_grid_index.csv')

    ai=xr.open_dataset('../data/AI_1901-2017_360x720.nc')

    # Use ERA5 ET as it covers the ocean
    dfe = load_era5_data('ET','ymonmean')
    
    n_lat, n_lon = dfe.shape[1::]

    # regional id 
    region_id=subregion_index['id'].unique()
    date_list = pd.date_range('20080101','20081231',freq='MS') #MS means start of Month
#    date_list = pd.date_range('20080101','20080131',freq='MS') #MS means start of Month

    # Use cross merge to add time 
    result = subregion_index.merge(pd.DataFrame(date_list.rename('time')),how='cross')

    for m,d in enumerate(date_list):
        # load moisture data for month m
        dfm = xr.open_dataset('../data/utrack_climatology_0.5_%02d.nc'%(m+1))
        print('procssing month %d'%(m+1))

        # Loop through provinces
#        for i in range(1):
        for i in range(region_id.shape[0]):
            region_ma = (subregion_index['id']==region_id[i])
            # Select moisture flow from source region defined by grid index, process region by region to save memory
            # here select target grids by their lat,lon belonging to reigon id
            temp_dfm= conversion_value(dfm.moisture_flow.sel(targetlat=xr.DataArray(subregion_index[region_ma].lat.values+0.25),
                                    targetlon=xr.DataArray(subregion_index[region_ma].lon.values+0.25)))
            print('processing region %d'%i)

            # change axis to allow for broadcast; targets grids (1st dim) muptiple ET to get precipitation contribution from 
            # different grids (2nd, 3rd dims)
            temp_p=np.moveaxis(temp_dfm.values, -1, 0) * dfe[m].values
#            print(temp_p.shape)
            # Sum to get ET-derived precipitation for target grid
           # result.loc[((result['id']==region_id[i])&(result['time']==d)),'e_to_prec'] = temp_p.sum(axis=(1,2))
            result.loc[((result['id']==region_id[i])&(result['time']==d)),'e_to_prec'] = np.nansum(temp_p,axis=(1,2))
            ## contribution from dryland itself
            result.loc[((result['id']==region_id[i])&(result['time']==d)),'e_to_prec_cndry'] =np.nansum(np.where(np.flipud(ai.Band1)>0,temp_p,0),axis=(1,2))

    result_xr = result.set_index(['time','lat','lon'])[['e_to_prec','e_to_prec_cndry']].to_xarray()
    result_xr.lat.attrs["units"] = "degrees_north"
    result_xr.lon.attrs["units"] = "degrees_west"
    result_xr.lat.attrs["long_name"] = "latitude"
    result_xr.lon.attrs["long_name"] = "longitude"
    result_xr.to_netcdf('../data/results/ERA5_e_to_prec_ymonmean_2008-20171127.nc')

    print('file saved')

# Calculate moisture region of dryland and save by aridity
def save_prec_source_aridity():
    subregion_index=pd.read_csv('../data/china_dryland_grid_index.csv')

    # Use ERA5 ET as it covers the ocean
   # dfe = xr.open_dataset('../data/ERA5-ET-2008-2017-ymonmean-05deg-clean.nc')['E']
    dfe = load_era5_data('ET','ymonmean')
    
    n_lat, n_lon = dfe.shape[1::]
    
    p = np.zeros([4,12, n_lat,n_lon]) # first dim is 4 aridity

    # regional id 
    region_id=subregion_index['id'].unique() # region id here is for four aridity levels 
    date_list = pd.date_range('20080101','20081231',freq='MS') #MS means start of Month
#    date_list = pd.date_range('20080101','20080228',freq='MS') #MS means start of Month

    for m,d in enumerate(date_list):
        # load moisture data for month m
        dfm = xr.open_dataset('../data/utrack_climatology_0.5_%02d.nc'%(m+1))
        print('procssing month %d'%(m+1))

        # Loop through provinces
#        for i in range(1):
        for i in range(region_id.shape[0]):
            region_ma = (subregion_index['id']==region_id[i])
            # Select moisture flow from source region defined by grid index, process region by region to save memory
            # here select target grids by their lat,lon belonging to reigon id
            temp_dfm= conversion_value(dfm.moisture_flow.sel(targetlat=xr.DataArray(subregion_index[region_ma].lat.values+0.25),
                                    targetlon=xr.DataArray(subregion_index[region_ma].lon.values+0.25)))
            print('processing region %d'%i)

            # change axis to allow for broadcast; targets grids (1st dim) muptiple ET to get precipitation contribution from 
            # different grids (2nd, 3rd dims)
            temp_p=np.moveaxis(temp_dfm.values, -1, 0) * dfe[m].values
            p[i,m,:,:]=temp_p.sum(axis=0)

            # Sum to get ET-derived precipitation for target grid

    dpre=xr.DataArray(p, coords=[region_id,range(1,13), dfe.lat, dfe.lon],dims=['aridity','month','lat','lon'],name='e_to_prec')

    dpre.to_netcdf('../data/results/china_dryland_prec_source_aridity.nc')
    print('file saved')

# Calculate precipitation source region for each province in China
def save_prec_source_province():
    subregion_index=pd.read_csv('../data/china_province_grid_index.csv')

    # Use ERA5 ET as it covers the ocean
    dfe = load_era5_data('ET','ymonmean')
    
    n_lat, n_lon = dfe.shape[1::]

    # regional id 
    region_id=subregion_index['id'].unique()
#    date_list = pd.date_range('20080101','20080228',freq='MS') #MS means start of Month
    date_list = pd.date_range('20080101','20081231',freq='MS') #MS means start of Month

    p = np.zeros([region_id.shape[0], 12, n_lat,n_lon])

    for m,d in enumerate(date_list):
        # load moisture data for month m
        dfm = xr.open_dataset('../data/utrack_climatology_0.5_%02d.nc'%(m+1))
        print('procssing month %d'%(m+1))

        # Loop through provinces
#        for i in range(1):
        for i in range(region_id.shape[0]):
            region_ma = (subregion_index['id']==region_id[i])
            # Select moisture flow from source region defined by grid index, process region by region to save memory
            # here select target grids by their lat,lon belonging to reigon id
            temp_dfm= conversion_value(dfm.moisture_flow.sel(targetlat=xr.DataArray(subregion_index[region_ma].lat.values+0.25),
                                    targetlon=xr.DataArray(subregion_index[region_ma].lon.values+0.25)))
            print('processing region %d'%i)

            # change axis to allow for broadcast; targets grids (1st dim) muptiple ET to get precipitation contribution from 
            # different grids (2nd, 3rd dims)
            temp_p=np.moveaxis(temp_dfm.values, -1, 0) * dfe[m].values
            p[i,m,:,:]=temp_p.sum(axis=0)

            # Sum to get ET-derived precipitation for target grid

    dpre=xr.DataArray(p, coords=[region_id,range(1,13), dfe.lat, dfe.lon],dims=['province','month','lat','lon'],name='e_to_prec')

    dpre.to_netcdf('../data/results/china_dryland_prec_source_province.nc')
    print('file saved')

if __name__=="__main__":
#    save_cn_label(rerun=True)
#    save_e_to_prec(start_year=1990, end_year=2020,et_data='GLEAM')
#    save_e_to_prec(start_year=2000, end_year=2020,et_data='GLEAM')
    save_e_to_prec(start_year=2002, end_year=2022,et_data='GLEAM')
#    save_e_to_prec(start_year=2000, end_year=2022,et_data='ERA5')
#    save_e_to_prec(start_year=2003, end_year=2022,et_data='ERA5')
#    save_e_to_prec(start_year=1990, end_year=2022,et_data='ERA5')
#    save_e_to_prec_ymonmean()
#    save_prec_source_aridity()
#    save_prec_source_province()
#    save_province_grid_index()
#    et_data='GLEAM_v3.5a'
