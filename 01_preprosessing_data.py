import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import plot, parse_date, parse_point, river_distance, compare_codes, doy, dop

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

river_shape = gpd.read_file("data/shapefiles/rivers/beleidslijn_grote_rivieren.shp")
rivers = gpd.read_file("data/shapefiles/rivers_centerline/record.shp")
Maas = rivers.loc[0, 'geometry']
Waal = rivers.loc[1, 'geometry']

# SECTION 1
# Loading data

filename = "data/Database_26112019.xlsx"
df = pd.read_excel(filename, sheet_name="Database totaal", skiprows=9, na_values="Leeg", index_col=None)
df['plastic_kunststof_band_tiewraps'] = pd.to_numeric(df['plastic_kunststof_band_tiewraps'],
                                                      errors='coerce')  # (40) " "
df['plastic_sportvisspullen'] = pd.to_numeric(df['plastic_sportvisspullen'], errors='coerce')  # (5) '8*'
df['plastic_industrieel_verpakkingsmateriaal'] = pd.to_numeric(df['plastic_industrieel_verpakkingsmateriaal'],
                                                               errors='coerce')  # (40) ' '
df = df.loc[df.kon_de_meting_worden_uitgevoerd != 'Nee', :]
df = df.iloc[:, :-16]  # Drop summation columns
df.loc[:, 'date'] = df.datum_monitoring.apply(parse_date)
df.columns = [x.replace("-", "_") for x in list(df.columns)]
df.columns = ['Gebiedscode' if 'Gebiedscode nieuw' in x else x for x in list(df.columns)]

def clean(x):
    if isinstance(x, str):
        for token in ["meter", "m", "ca.", "ca", "Mr"]:
            x = x.replace(token, "")
    return x

df.gemeten_lengte_in_meters_parallel_aan_de_rivier = df.gemeten_lengte_in_meters_parallel_aan_de_rivier.apply(lambda x: clean(x))
df.loc[df.gemeten_lengte_in_meters_parallel_aan_de_rivier.str.contains("100/110", na=False), "gemeten_lengte_in_meters_parallel_aan_de_rivier"] = 100
df.loc[df.gemeten_lengte_in_meters_parallel_aan_de_rivier.str.contains("2 aal 50", na=False), "gemeten_lengte_in_meters_parallel_aan_de_rivier"] = 100
df.loc[df.gemeten_lengte_in_meters_parallel_aan_de_rivier == '50  aan beide zijden van de weg, dus totaal 100 ', "gemeten_lengte_in_meters_parallel_aan_de_rivier"] = 100
df.loc[df.gemeten_lengte_in_meters_parallel_aan_de_rivier == '70  (net als in februari en oktober 2018)', "gemeten_lengte_in_meters_parallel_aan_de_rivier"] = 70
df.loc[df.gemeten_lengte_in_meters_parallel_aan_de_rivier == '70  (net als in februari 2018)', "gemeten_lengte_in_meters_parallel_aan_de_rivier"] = 70
df.loc[df.gemeten_lengte_in_meters_parallel_aan_de_rivier == 'bij jullie bekend', "gemeten_lengte_in_meters_parallel_aan_de_rivier"] = 70
df.loc[df.gemeten_lengte_in_meters_parallel_aan_de_rivier == '+/_ 95', "gemeten_lengte_in_meters_parallel_aan_de_rivier"] = 70
df.gemeten_lengte_in_meters_parallel_aan_de_rivier = df.gemeten_lengte_in_meters_parallel_aan_de_rivier.astype(int)

# Filling in the wholes for new and discontinued items
FIRST_ITEM = "plastic_6_packringen"
LAST_ITEM = "granulaat_korrels"
ind_item = [list(df.columns).index(x) for x in [FIRST_ITEM, LAST_ITEM]]
s_items = slice(ind_item[0], ind_item[1] + 1, None)

df_copy = df.copy()

colnames = []
colnames_original = list(df.columns)
for column in list(df.columns)[ind_item[0]:ind_item[1] + 1]:
    if "NEW" in column or "OLD" in column:
        for m in range(4):
            s = df.loc[df.meting == m, column]
            # If a single value is present in the measurement round, the rest are assumed to be zeros,
            # while if none are present, it is assumed that the item was not on the list for the given round.
            if s.isna().sum() != s.size:
                df.loc[np.logical_and(df.meting == m, df[column].isna()), column] = 0
        colnames.append(column[4:])
    else:
        df.loc[df[column].isna(), column] = 0
        colnames.append(column)
colnames_original[ind_item[0]:ind_item[1] + 1] = colnames
df.columns = colnames_original

# Visual comparison after filling the holes
f, ax = plt.subplots(ncols=2)
ax[0].imshow(df.iloc[:, ind_item[0]:ind_item[1] + 1].isna())
ax[1].imshow(df_copy.iloc[:, ind_item[0]:ind_item[1] + 1].isna())
plt.show()

ITEMS = [item for item in list(df.columns)[ind_item[0]:ind_item[1] + 1]]

# Calculate 'Day of project' and 'Day of year'
df.loc[:, 'date'] = df.date.astype(str)
df.loc[:, 'doy'] = df.loc[:, 'date'].apply(lambda x: doy(x) if x != "0" else np.nan)
df.loc[:, 'dop'] = df.loc[:, 'date'].apply(lambda x: dop(x) if x != "0" else np.nan)

# SECIONT 2
# Loading SDN locations

df_locations = pd.read_excel(filename,
                             sheet_name="Tracelijst 2019", skiprows=27, na_values=["Leeg", "Blank"]).iloc[:, :4]
df_locations.loc[df_locations.coordinaten == "51.4251.6.1561", 'coordinaten'] = "51.4251,6.1561"
df_locations.loc[df_locations.coordinaten == "51.7851.5.3641", 'coordinaten'] = "51.7851,5.3641"
df_locations.loc[df_locations.coordinaten == "51.7143.4,783131", 'coordinaten'] = "51.7143,4.783131"
df_locations.loc[:, "geometry"] = df_locations.coordinaten.apply(lambda x: parse_point(x))
df_locations = gpd.GeoDataFrame(df_locations)
df_locations.crs = {'init': 'epsg:4326'}
df_locations = df_locations.loc[~df_locations.geometry.isna(), :]
df_locations.loc[:, 'river'] = df_locations['Nieuwe code'].apply(lambda x: x[0])
df_locations = df_locations.iloc[:, [1, 4, 5]]
df_locations.columns = ['Gebiedscode', 'geometry', 'river']

ax = plot()
df_locations.plot(ax=ax, transform=ccrs.PlateCarree())
plt.title("Locations of river OSPAR observations")
plt.show()

# SECTION 3
# QUALITY CONTROL

dubious_codes = []
# Sample IDs that seem to be duplicates
for i in df.Gebiedscode.unique():
    if i == 0 or i == '0':
        continue
    x = df.loc[df.Gebiedscode == i, 'meting']
    if df[df.duplicated(keep=False)].shape[0] > 0:
        dubious_codes.append(i)

df_locations.loc[:, 'distance_waal'] = df_locations.geometry.apply(lambda x: river_distance(Waal, x))
df_locations.loc[:, 'distance_maas'] = df_locations.geometry.apply(lambda x: river_distance(Maas, x))
df_locations.loc[:, 'river'] = df_locations.apply(lambda x: "W" if x.distance_waal < x.distance_maas else "M", axis=1)
df_locations.loc[:, 'compare'] = df_locations.apply(lambda x: compare_codes(x.river, x.Gebiedscode), axis=1)

df_locations = df_locations.iloc[:, :2]

# Calculate distances on the river
df_locations.columns = ['Gebiedscode', 'geometry']
df_locations.loc[:, 'river'] = df_locations.Gebiedscode.apply(lambda x: x[0])
df_locations.loc[:, 'x_maas'] = np.nan
df_locations.loc[:, 'x_waal'] = np.nan
df_locations.loc[df_locations.river == "M", 'x_maas'] = \
    df_locations.loc[df_locations.river == "M", 'geometry'].apply(lambda x: Maas.project(x))
df_locations.loc[df_locations.river == "W", 'x_waal'] = \
    df_locations.loc[df_locations.river == "W", 'geometry'].apply(lambda x: Waal.project(x))

print("The following codes seem to be inconsistent")
print(*dubious_codes)

# Saving the data
df.loc[:, ['Gebiedscode', 'meting', 'date', 'doy', 'dop'] + ITEMS].to_csv("data/processed_data/df_raw.csv")
df_raw = df.copy()

# Converting all observations to a standard plot of 2500 m2
for item in ITEMS:
    df[item] = np.floor(df[item] / df.m2 * 2500)
df.loc[:, ['Gebiedscode', 'meting', 'date', 'doy', 'dop'] + ITEMS].to_csv("data/processed_data/df_2500m2.csv")

df = df_raw.copy()

# Converting all observations to m2
for item in ITEMS:
    df[item] = df[item] / df.m2
df.loc[:, ['Gebiedscode', 'meting', 'date', 'doy', 'dop'] + ITEMS].to_csv("data/processed_data/df_m2.csv")

df = df_raw.copy()

# Converting all observations to km
for item in ITEMS:
    df[item] = np.floor(df[item] / df.gemeten_lengte_in_meters_parallel_aan_de_rivier * 1000)
df.loc[:, ['Gebiedscode', 'meting', 'date', 'doy', 'dop'] + ITEMS].to_csv("data/processed_data/df_per_km.csv")

df_locations.to_file("data/processed_data/df_locations.geojson", driver="GeoJSON")
