# %%
import xarray as xr
import numpy as np
# %%

e5 = [xr.open_dataset(f"C:/Users/matte/data/era5-land/europe-era5_land-tp_t2m-{y}.grib")['t2m'] for y in range(2000, 2023)]
e5 = xr.concat(e5, dim = 'time')
# %%
hr = e5.sel(longitude = slice(5, 17.7), 
latitude = slice(55, 42.15))
2# %%
lr_lon = np.linspace(hr.longitude.values[0], hr.longitude.values[-1], 32)
lr_lat = np.linspace(hr.latitude.values[-1], hr.latitude.values[0], 32)

lr = hr.interp(longitude = lr_lon, latitude = lr_lat)
# %%
hr.drop(['step', 'number', 'surface']).to_netcdf('hr_t2m.nc')
lr.drop(['step', 'number', 'surface']).to_netcdf('lr_t2m.nc')
# %%
