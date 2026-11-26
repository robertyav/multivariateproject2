from pathlib import Path
import xarray as xr

#Load data
data_dir = Path.home() / "Documents/sjsu/fall 2026/257/Math257 Project2"
dt = xr.open_dataset(data_dir/"europe_2022_May_Sep.nc", engine="netcdf4")

#Select Mediterranean area
#lon: 6°W to 19°E and lat: 36°N to 46°N
#include part of Atlantic. 【need to fix it later】
dt_med = dt.sel(lat=slice(36,46),lon=slice(-6,19))
print(dt_med.sizes)

#Use July data
variables = ["sea_surface_temperature",
             "wind_speed",
             "water_vapor"]
month_data = dt_med[variables].sel(time=slice("2022-07-01","2022-07-31"))

#original grid spacing
print(month_data.lat.values[1]-month_data.lat.values[0])
print(month_data.lon.values[1]-month_data.lon.values[0])
#Result:0.25°

#Spatial subsample from 0.25° to 0.75° for faster computation
month_data = month_data.isel(lat=slice(None,None,3), lon=slice(None,None,3))
print(month_data.sizes)
print(month_data.lat.values[1] - month_data.lat.values[0])
print(month_data.lon.values[1] - month_data.lon.values[0])
#Result: 0.75°


#Plot of daily mean of three variables
import matplotlib.pyplot as plt

#daily mean
sst = month_data["sea_surface_temperature"]-273.15 #kelvin to celsius
vapor = month_data["water_vapor"]
wind = month_data["wind_speed"]

sst_daily = sst.resample(time="1D").mean().mean(dim=["lat","lon"])
vapor_daily = vapor.resample(time="1D").mean().mean(dim=["lat","lon"])
wind_daily = wind.resample(time="1D").mean().mean(dim=["lat","lon"])

#Plot
fig, ax = plt.subplots(3,1, figsize=(10,8),sharex=True)
ax[0].plot(sst_daily.time,sst_daily,color="red")
ax[0].set_ylabel("SST (°C)")

ax[1].plot(vapor_daily.time, vapor_daily,color="blue")
ax[1].set_ylabel("Water vapor (kg/m²)")

ax[2].plot(wind_daily.time,wind_daily, color="green")
ax[2].set_ylabel("Wind speed (m/s)")
ax[2].set_xlabel("Date")
fig.suptitle("Daily Mean - July 2022")
plt.tight_layout()
#plt.show()


#Plot of spatial distribution
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#Monthly mean
sst_mean = sst.mean(dim="time")
vapor_mean = vapor.mean(dim="time")
wind_mean = wind.mean(dim="time")

data = [sst_mean,vapor_mean,wind_mean]
names = ["SST","Water vapor","Wind speed"]
units = ["°C", "kg/m²","m/s"]
cmaps = ["turbo","Blues","viridis"]

#Spatial distribution
fig, ax = plt.subplots(3,1,figsize=(12,12),
    subplot_kw={"projection":ccrs.PlateCarree()})

for i in range(3):
    data[i].plot(ax=ax[i], cmap=cmaps[i],
                 cbar_kwargs={"label":units[i]})
    ax[i].coastlines()
    ax[i].add_feature(cfeature.LAND,facecolor="lightgray")
    ax[i].set_extent([-6,19,36,46])
    ax[i].set_title(names[i])

    gl = ax[i].gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

fig.suptitle("Monthly Mean Spatial Distribution - July 2022")
plt.tight_layout()
plt.show()


#convvert to dataframe
points_month = month_data.to_dataframe().reset_index()
points_month = points_month.rename(columns={"sea_surface_temperature":"temperature"})
print(points_month.head())
print(points_month.shape)

#Coverage for each coordinate
model_variables = ["temperature","wind_speed","water_vapor"]
n_time_records = month_data.sizes["time"]
coverage = points_month.groupby(["lat","lon"])[model_variables].count()/n_time_records
coverage["minimum_coverage"] = coverage.min(axis=1)
print(coverage.head())
print(coverage.describe())

#Remove na values
observed = points_month.dropna(subset=model_variables).copy()
print(observed.head())
print(observed.shape)

#Count unique spatial locations
n_locations = observed[["lat","lon"]].drop_duplicates().shape[0]
print(n_locations)


from pyproj import Transformer

#Transform coordinates
transformer = Transformer.from_crs("EPSG:4326","EPSG:3034", always_xy=True)

observed["x"], observed["y"] = transformer.transform(
    observed["lon"].values, observed["lat"].values
)
print(observed.head())

#Convert time to days
observed["time_days"] = (observed["time"]-observed["time"].min()).dt.total_seconds()/86400

print(observed.head())
print(observed["time_days"].describe())
print(observed.shape)


import pandas as pd
import numpy as np

#train/test split
observed = observed.reset_index(drop=True)
coords = observed[["lat","lon"]].drop_duplicates()
n_test = round(len(coords)*0.2)

test_coords = coords.sample(n=n_test,random_state=0)
test_points = pd.MultiIndex.from_frame(test_coords)
all_points = pd.MultiIndex.from_frame(observed[["lat","lon"]])

observed["set"] = np.where(all_points.isin(test_points),"test","train")
train = observed[observed["set"]=="train"].copy()
test = observed[observed["set"]=="test"].copy()

print(len(train))
print(len(test))

train_locations = set(zip(train["lat"],train["lon"]))
test_locations = set(zip(test["lat"], test["lon"]))
print(len(train_locations))
print(len(test_locations))
print(len(train_locations & test_locations))

#Select features
features = ["x","y","time_days","water_vapor","wind_speed"]
X_train = train[features]
y_train = train["temperature"]
X_test = test[features]
y_test = test["temperature"]

print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import StandardScaler

#Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled[:5])


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, DotProduct, ConstantKernel, WhiteKernel

#Define kernels
kernels = {
    "Matern_32": ConstantKernel(1.0,(1e-3,1e3)) *
    Matern(1.0,(1e-2,1e2),nu=1.5)
    + WhiteKernel(1.0,(1e-5,1e2)),

    "Matern_52": ConstantKernel(1.0, (1e-3,1e3)) *
    Matern(1.0,(1e-2,1e2),nu=2.5)
    + WhiteKernel(1.0,(1e-5,1e2)),

    "RBF": ConstantKernel(1.0,(1e-3,1e3)) *
    RBF(1.0,(1e-2,1e2))
    + WhiteKernel(1.0,(1e-5,1e2)),

    "DotProduct": ConstantKernel(1.0,(1e-3,1e3)) *
    DotProduct(1.0,(1e-3,1e3))
    + WhiteKernel(1.0, (1e-4,1e2))
}

from sklearn.metrics import mean_squared_error, mean_absolute_error

models = {}
predictions = {}
results = []

#full GP models
for name, kernel in kernels.items():
    print(name)
    model = GaussianProcessRegressor(kernel=kernel,
        alpha=1e-4,normalize_y=True,
        n_restarts_optimizer=2, random_state=0)

    model.fit(X_train_scaled,y_train)
    y_pred,y_std = model.predict(X_test_scaled,return_std=True)

    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    lml = model.log_marginal_likelihood()

    models[name] = model
    predictions[name] = {"prediction":y_pred,"std":y_std}
    results.append({"kernel":name,"RMSE":rmse,
                    "MAE":mae,"LML":lml})

    print(model.kernel_)
    print(rmse,mae,lml)

results_df = pd.DataFrame(results)
print(results_df)


#0.5 result  took 30 mins to run
#       kernel      RMSE       MAE          LML
#0   Matern_32  0.550377  0.386600     3.121671
#1   Matern_52  0.561755  0.397963   -77.526148
#2         RBF  0.595939  0.427653  -282.268147
#3  DotProduct  1.261328  1.024720 -3837.550527

#0.75 result took 2 mins to run
#       kernel      RMSE       MAE          LML
#0   Matern_32  0.592794  0.428352  -607.842375
#1   Matern_52  0.603371  0.438322  -629.611737
#2         RBF  0.635218  0.466789  -690.522060
#3  DotProduct  1.405321  1.141568 -2105.769228