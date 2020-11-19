import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from functions import date_subtract, filter_date, stations_available, reduce_data_stations, \
    idw_2d, idw_1d, distance, closest_point

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 1000)

df = pd.read_csv("data/processed_data/df_2500m2.csv", index_col=0)
df_locations = gpd.read_file("data/processed_data/df_locations.geojson")
df_locations.columns = ["Gebiedscode", "river", "x_maas", "x_waal", "geometry"]

river_shape = gpd.read_file(
    "data/shapefiles/rivers/beleidslijn_grote_rivieren.shp")
rivers = gpd.read_file(
    "data/shapefiles/rivers_centerline/record.shp")

Maas = rivers.loc[0, 'geometry']
Waal = rivers.loc[1, 'geometry']

################
# SPATIAL DATA #
################

df.loc[:, 'date'] = df.date.astype(str)
df.loc[:, 'date_lag_1d'] = df.date.apply(lambda x: date_subtract(x, days=1) if x != 0 else "0")
df.loc[:, 'date_lag_2d'] = df.date.apply(lambda x: date_subtract(x, days=2) if x != "0" else "0")
df.loc[:, 'date_lag_7d'] = df.date.apply(lambda x: date_subtract(x, days=7) if x != "0" else "0")
df.loc[:, 'date_lag_14d'] = df.date.apply(lambda x: date_subtract(x, days=14) if x != "0" else "0")
df.loc[:, 'date_lag_1m'] = df.date.apply(lambda x: date_subtract(x, months=1) if x != "0" else "0")
df.loc[:, 'date_lag_6m'] = df.date.apply(lambda x: date_subtract(x, months=6) if x != "0" else "0")

# KNMI DATA
knmi_data = pd.read_csv("data/wind_and_precip_KNMI/KNMI_20191001.txt", skiprows=64)
knmi_data.columns = ["id", "date", "wind_direction", "wind_speed", "wind_speed_day", "wind_speed_max", "p_day", "p_max"]

knmi_data.wind_direction = \
    knmi_data.wind_direction.apply(lambda x: float(x) if len(x.strip()) > 0 else np.nan)

knmi_data.wind_speed = \
    knmi_data.wind_speed.apply(lambda x: float(x) if len(x.strip()) > 0 else np.nan) / 10

knmi_data.wind_speed_day = \
    knmi_data.wind_speed_day.apply(lambda x: float(x) if len(x.strip()) > 0 else np.nan) / 10

knmi_data.wind_speed_max = \
    knmi_data.wind_speed_max.apply(lambda x: float(x) if len(x.strip()) > 0 else np.nan) / 10

knmi_data.p_day = \
    knmi_data.p_day.apply(lambda x: float(x) if len(x.strip()) > 0 else np.nan) / 10
knmi_data.loc[knmi_data.p_day < 0, 'p_day'] = 0

knmi_data.p_max = \
    knmi_data.p_max.apply(lambda x: float(x) if len(x.strip()) > 0 else np.nan) / 10
knmi_data.loc[knmi_data.p_max < 0, 'p_max'] = 0

knmi_data.loc[:, 'date'] = knmi_data.date.astype(str)

# stations
knmi_stations = pd.read_csv("data/wind_and_precip_KNMI/stations.csv")
knmi_stations.columns = ["id", "lon", "lat", "altitude", "name"]
knmi_stations.loc[:, "id"] = knmi_stations.id.apply(lambda x: int(x[:-1]))
knmi_stations.loc[:, "geometry"] = knmi_stations.apply(lambda x: Point(x.lon, x.lat), axis=1)
knmi_stations = gpd.GeoDataFrame(knmi_stations)
knmi_stations.crs = {'init': 'epsg:4326'}


def extract_knmi_data(gebiedscode, start, end, col, fun, p=2):
    # KNMI data is well sorted on date. Only one entry per day.
    global knmi_data, knmi_stations, df_locations
    selected_data = filter_date(knmi_data, (start, end))
    available_stations = stations_available(knmi_stations, selected_data, col=col, keep_steps=False, mode='full')
    stations_data = reduce_data_stations(available_stations, selected_data, col=col, fun=fun)
    return idw_2d(stations_data, df_locations.loc[df_locations.Gebiedscode == gebiedscode, 'geometry'], col=col, p=p)


# TESTING RESULTS
# end = "20181023"
# start = date_subtract(end, days=7)
# gebiedscode = "W(001a)R-REFE"
# selected_data = filter_date(knmi_data, (start, end))
# available_stations = stations_available(knmi_stations, selected_data, col="p_day", keep_steps=True, mode='all')
# stations_data = reduce_data_stations(available_stations, selected_data, col="p_day", fun=pd.Series.sum)
# idw_2d(stations_data, df_locations.loc[df_locations.Gebiedscode == gebiedscode, 'geometry'], col="p_day", p=2)
#
# extract_knmi_data(gebiedscode, start, end, 'p_day', pd.Series.sum)


# SUM P_DAY
for date_lag in ['date_lag_1d', 'date_lag_2d', 'date_lag_7d', 'date_lag_14d', 'date_lag_1m', 'date_lag_6m']:
    new_col = "P_" + date_lag[date_lag.find("lag") + 4:] + "_sum"
    print(new_col)
    df.loc[:, new_col] = np.nan
    col = "p_day"
    fun = pd.Series.sum
    for index, row in df.iterrows():
        gebiedscode = row['Gebiedscode']
        if gebiedscode == 0 or gebiedscode == "Maak een keuze" or gebiedscode == "M(074b)R-REFE":
            continue
        start = row[date_lag]
        if start == "0":
            continue
        end = row['date']
        # print(gebiedscode, start, end, ospar_locations.loc[ospar_locations.Gebiedscode==gebiedscode,'geometry'])
        df.loc[index, new_col] = extract_knmi_data(gebiedscode, start, end, col, fun)

# MAX P_DAY
for date_lag in ['date_lag_1d', 'date_lag_2d', 'date_lag_7d', 'date_lag_14d', 'date_lag_1m', 'date_lag_6m']:
    new_col = "P_" + date_lag[date_lag.find("lag") + 4:] + "_max"
    print(new_col)
    df.loc[:, new_col] = np.nan
    col = "p_day"
    fun = pd.Series.max
    for index, row in df.iterrows():
        gebiedscode = row['Gebiedscode']
        if gebiedscode == 0 or gebiedscode == "Maak een keuze" or gebiedscode == "M(074b)R-REFE":
            continue
        start = row[date_lag]
        if start == "0":
            continue
        end = row['date']
        # print(gebiedscode, start, end, ospar_locations.loc[ospar_locations.Gebiedscode==gebiedscode,'geometry'])
        df.loc[index, new_col] = extract_knmi_data(gebiedscode, start, end, col, fun)

# MAX WIND SPEAD
for date_lag in ['date_lag_1d', 'date_lag_2d', 'date_lag_7d', 'date_lag_14d', 'date_lag_1m', 'date_lag_6m']:
    new_col = "U_" + date_lag[date_lag.find("lag") + 4:] + "_max"
    print(new_col)
    df.loc[:, new_col] = np.nan
    col = "wind_speed_day"
    fun = pd.Series.max
    for index, row in df.iterrows():
        gebiedscode = row['Gebiedscode']
        if gebiedscode == 0 or gebiedscode == "Maak een keuze" or gebiedscode == "M(074b)R-REFE":
            continue
        start = row[date_lag]
        if start == "0":
            continue
        end = row['date']
        # print(gebiedscode, start, end)
        df.loc[index, new_col] = extract_knmi_data(gebiedscode, start, end, col, fun)

# AVERAGE WIND SPEAD
for date_lag in ['date_lag_1d', 'date_lag_2d', 'date_lag_7d', 'date_lag_14d', 'date_lag_1m', 'date_lag_6m']:
    new_col = "U_" + date_lag[date_lag.find("lag") + 4:] + "_mean"
    print(new_col)
    df.loc[:, new_col] = np.nan
    col = "wind_speed_day"
    fun = pd.Series.mean
    for index, row in df.iterrows():
        gebiedscode = row['Gebiedscode']
        if gebiedscode == 0 or gebiedscode == "Maak een keuze" or gebiedscode == "M(074b)R-REFE":
            continue
        start = row[date_lag]
        if start == "0":
            continue
        end = row['date']
        # print(gebiedscode, start, end)
        df.loc[index, new_col] = extract_knmi_data(gebiedscode, start, end, col, fun)


s = "data/water_height_RWS/"
rws_data = []
for x in ["2017_1", "2017_2", "2018_1", "2018_2", "2019_1"]:
    rws_data.append(pd.read_csv(f"{s}{x}.csv", encoding="ISO-8859-1", sep=";", low_memory=False))

rws_data = pd.concat(rws_data)

rws_data = rws_data.loc[:, ["MEETPUNT_IDENTIFICATIE", 'WAARNEMINGDATUM', 'WAARNEMINGTIJD', 'NUMERIEKEWAARDE',
                            'KWALITEITSOORDEEL_CODE', 'STATUSWAARDE', 'X', 'Y']]
rws_data.columns = ["name", "date", "time", "value", "quality_code", "status", "x", "y"]
rws_data.loc[rws_data.value == 999999999, 'value'] = np.nan
rws_data.loc[:, 'date'] = rws_data.date.apply(lambda x: x[6:] + x[3:5] + x[:2])
rws_data = rws_data.loc[rws_data.name != "Eisden Mazenhove", :]
rws_data['value'] = rws_data['value'] / 100  # convert to [m]

rws_stations = rws_data.drop_duplicates(subset="name")
rws_stations.loc[:, 'x'] = rws_stations.x.apply(lambda x: float(x.replace(",", ".")))
rws_stations.loc[:, 'y'] = rws_stations.y.apply(lambda x: float(x.replace(",", ".")))
rws_stations.loc[:, 'geometry'] = rws_stations.apply(lambda x: Point(x.x, x.y), axis=1)
rws_stations = gpd.GeoDataFrame(rws_stations)
rws_stations.crs = {'init': 'epsg:25831'}
rws_stations = rws_stations.to_crs({'init': 'epsg:4326'})
rws_stations = rws_stations.loc[:, ['name', 'geometry']]
rws_stations.loc[:, 'maas'] = rws_stations.geometry.apply(lambda x: distance(x, closest_point(x, Maas)) < 3)
rws_stations.loc[:, 'waal'] = rws_stations.geometry.apply(lambda x: distance(x, closest_point(x, Waal)) < 3)
rws_stations.loc[rws_stations.name == 'Sint Andries Waal', 'maas'] = False
rws_stations.loc[:, 'x_maas'] = rws_stations.apply(lambda x: Maas.project(x.geometry) if x.maas else np.nan, axis=1)
rws_stations.loc[:, 'x_waal'] = rws_stations.apply(lambda x: Waal.project(x.geometry) if x.waal else np.nan, axis=1)
rws_stations = rws_stations.loc[
    np.logical_or(rws_stations.maas, rws_stations.waal), ['name', 'geometry', 'x_maas', 'x_waal']]
rws_stations.columns = ["id", "geometry", "x_maas", "x_waal"]
rws_stations.to_file("data/shapefiles/rws_locations/rws_locations.shp")

rws_data = rws_data.groupby(['name', 'date', 'time']).mean().reset_index()
rws_data = rws_data.groupby(['name', 'date']).mean().reset_index()
rws_data.columns = ["id", "date", "value"]


def extract_rws_data(gebiedscode, start, end, fun, coord_col, col='value', p=2):
    global rws_data, rws_stations, ospar_locations
    selected_data = filter_date(rws_data, (start, end))
    available_stations = stations_available(rws_stations, selected_data, col=col, keep_steps=False, mode='full')
    stations_data = reduce_data_stations(available_stations, selected_data, col=col, fun=fun)
    point = df_locations.loc[df_locations.Gebiedscode == gebiedscode, coord_col].values
    return idw_1d(stations_data, coordinate_col=coord_col, point=point, col=col, p=p)


# TEST RESULTS
# end = "20181023"
# start = "20181020"
# gebiedscode = "W(001a)R-REFE"
# fun = (lambda x: pd.Series.max(x) - x.iloc[0])
# col = 'value'
# coord_col = 'x_waal'
# selected_data = filter_date(rws_data, (start, end))
# available_stations = stations_available(rws_stations, selected_data, col=col, keep_steps=False, mode='full')
# stations_data = reduce_data_stations(available_stations, selected_data, col=col, fun=fun)
# point = df_locations.loc[df_locations.Gebiedscode == gebiedscode, coord_col].values
# idw_1d(stations_data, coordinate_col=coord_col, point=point, col=col, p=2)
#
# extract_rws_data(gebiedscode, start, end, fun, coord_col)


# Maximum water height above height at measurement time
for date_lag in ['date_lag_2d', 'date_lag_7d', 'date_lag_14d', 'date_lag_1m', 'date_lag_6m']:
    new_col = "h_" + date_lag[date_lag.find("lag") + 4:] + "_max_above_current"
    print(new_col)
    df.loc[:, new_col] = np.nan
    col = "value"
    fun = (lambda x: pd.Series.max(x) - x.iloc[0])
    for index, row in df.iterrows():
        gebiedscode = row['Gebiedscode']
        if gebiedscode == 0 or gebiedscode == "Maak een keuze" or gebiedscode == "M(074b)R-REFE":
            continue
        start = row[date_lag]
        if start == "0":
            continue
        end = row['date']
        # print(gebiedscode, start, end)
        if gebiedscode[0] == "M":
            coord_col = 'x_maas'
        if gebiedscode[0] == "W":
            coord_col = 'x_waal'
        df.loc[index, new_col] = extract_rws_data(gebiedscode=gebiedscode, start=start, end=end, col=col,
                                                  fun=fun, coord_col=coord_col)

df.to_csv("data/processed_data/df_hm_included.csv")
