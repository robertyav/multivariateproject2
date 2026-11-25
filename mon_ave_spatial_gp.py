from pathlib import Path
import xarray as xr

#Load data
data_folder = Path("/Users/silingguo/Documents/sjsu/fall 2026/257/Math257 Project2")
dt = xr.open_dataset(data_folder/"europe_2022_May_Sep.nc", engine="netcdf4")

sst = dt["sea_surface_temperature"]
flags = dt["l2p_flags"]

#Remove land
land = (flags & 2)!= 0
non_land = ~land
sst = sst.where(non_land)

n_ocean = non_land.all(dim="time").sum().item()
print("Non-land locations:", n_ocean)

#Monthly average
sst_monthly = sst.groupby("time.month").mean(dim="time", skipna=True)

import pandas as pd

#Check missing values
results =[]

for month in [5, 6, 7, 8, 9]:
    sst_month = sst_monthly.sel(month=month)
    n_observed = sst_month.notnull().sum().item()
    n_na = n_ocean - n_observed
    coverage = n_observed/n_ocean*100

    results.append([month, n_observed, n_na, round(coverage, 2)])

summary = pd.DataFrame(results, columns=["Month", "Observed", "Missing", "Coverage (%)"])
print(summary)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#Plot
vmin =sst_monthly.min().item()
vmax =sst_monthly.max().item()

for month in [5, 6, 7, 8, 9]:
    sst_month = sst_monthly.sel(month=month)
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-12, 21, 34, 63])
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.gridlines(draw_labels=True)

    im = ax.pcolormesh(dt.lon, dt.lat, sst_month, cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="SST (K)")
    plt.title("Monthly Average SST - Month " + str(month))

plt.show()

#Convert to dataframe
sst_df = sst_monthly.to_dataframe(name="sst").reset_index()
observed = sst_df.dropna(subset=["sst"])

#Select July data for example(the peak of 2022 heatwave)
jul = observed[observed["month"] == 7]
jul = jul.sample(n=1000, random_state=1)

X = jul[["lon", "lat"]]
y = jul["sst"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

#GP models
kernels = {
    "Matern_32": ConstantKernel(1.0, (1e-3, 1e3))
    * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
    + WhiteKernel(1.0, (1e-5, 1e2)),

    "Matern_52": ConstantKernel(1.0, (1e-3, 1e3))
    * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
    + WhiteKernel(1.0, (1e-5, 1e2)),

    "RBF": ConstantKernel(1.0, (1e-3, 1e3))
    * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    + WhiteKernel(1.0, (1e-5, 1e2))
}
results =[]

for name, kernel in kernels.items():
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(X_train, y_train)
    y_pred = gp.predict(X_test)

    print(name)
    print(gp.kernel_)

    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    lml = gp.log_marginal_likelihood_value_

    results.append([name, rmse, mae, lml])

metrics = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "LML"])
print(metrics)