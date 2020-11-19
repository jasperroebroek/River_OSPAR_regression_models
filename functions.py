import geomappy as mp
from geomappy.basemap import ProjectCustomExtent
import matplotlib.pyplot as plt
from shapely.geometry import Point
import cartopy.feature as cf
import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np
import pandas as pd


rivers_and_lakes = \
    gpd.read_file("data/shapefiles/ne_10m_rivers_lake_centerlines/ne_10m_rivers_lake_centerlines.shp")
rivers_and_lakes_extension = \
    gpd.read_file("data/shapefiles/ne_10m_rivers_europe/ne_10m_rivers_europe.shp")
river_shape = gpd.read_file("data/shapefiles/rivers/beleidslijn_grote_rivieren.shp")
centerlines = gpd.read_file("data/shapefiles/rivers_centerline/record.shp")


# PLOTTING FUNCTIONS
def plot(figsize=(10, 10), fontsize=10, borderstyle="-", background_river=True, rivers=True):
    ax = mp.basemap(x0=3.17, x1=7.5, y0=50.4, y1=53.9,
                    projection=ProjectCustomExtent(epsg=28992, extent=[-1000000, 500000, -100000, 800000]),
                    resolution='10m', xticks=1, yticks=1, fontsize=fontsize, figsize=figsize,
                    grid_alpha=0.75)
    ax.add_feature(
        cf.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', '10m', facecolor='none', edgecolor='k'),
        label='Stereo', lw=1, linestyle=borderstyle)
    ax.add_feature(cf.NaturalEarthFeature('physical', 'lakes', '10m', facecolor='none', edgecolor='black'))
    if background_river:
        river_shape.plot(ax=ax, transform=ax.projection, edgecolor='none', facecolor='lightgrey', alpha=0.9,
                         linewidth=0.1)
    if rivers:
        centerlines.plot(ax=ax, transform=ccrs.PlateCarree(), edgecolor='cornflowerblue', alpha=0.95)

    return ax


def draw_labels(ax, label, xy, va='top', ha='right', hpad=0, vpad=0, arrow=False, **kwargs):
    xy = xy[0]
    xytext = (xy[0] + hpad, xy[1] + vpad)
    if arrow:
        points = np.linspace(xy, xytext, num=8)
        xs = points[2:-1, 0]
        ys = points[2:-1, 1]
        ax.plot(xs, ys, transform=kwargs.get('transform',ccrs.PlateCarree()), color='Grey')
    ax.text(xytext[0], xytext[1], label, va=va, ha=ha, **kwargs)


def draw_annotations(t, x, y, ax=None, overlapping_pixels=0, fontsize=10, **kwargs):
    if isinstance(ax, type(None)):
        ax = plt.gca()

    mask = np.zeros(ax.figure.canvas.get_width_height(), bool)
    plt.tight_layout()
    ax.figure.canvas.draw()

    va_positions = {'b': 'bottom', 't': 'top', 'c': 'center'}
    ha_positions = {'l': 'left', 'r': 'right', 'c': 'center'}

    indices = np.arange(len(t))

    for i in indices:
        for position in ['bl', 'tl', 'tr', 'br', 'cl', 'cr', 'tc', 'bc']:
            va = va_positions[position[0]]
            ha = ha_positions[position[1]]

            a = ax.text(x=x[i], y=y[i], s=t[i], ha=ha, va=va, fontsize=fontsize, **kwargs)

            bbox = a.get_window_extent()
            x0 = int(bbox.x0)+overlapping_pixels
            x1 = int(np.ceil(bbox.x1))-overlapping_pixels
            y0 = int(bbox.y0)+overlapping_pixels
            y1 = int(np.ceil(bbox.y1))-overlapping_pixels

            s = np.s_[x0:x1 + 1, y0:y1 + 1]
            if np.any(mask[s]):
                a.set_visible(False)
            else:
                mask[s] = True
                break


# PARSING THE INPUT
def river_distance(river, p):
    if pd.isna(p):
        return np.nan
    projected_point = closest_point(p, river)
    return distance(projected_point, p)


def compare_codes(r, code):
    if pd.isna(code):
        return False
    return code[0] == r


def parse_date(d):
    if pd.isna(d):
        return "0"
    else:
        d = str(d)
    return d[:4] + d[5:7] + d[8:10]


def parse_point(p):
    if isinstance(p, (int, float)):
        return p
    else:
        split_on = p.find(",")
        return Point(float(p[split_on + 1:]), float(p[:split_on]))


# GEOMETRY CALCULATIONS
def closest_point(p, l):
    """Calculate closest from a point on a line"""
    return l.interpolate(l.project(p))


def distance_cart(p1, p2):
    """distance between two points"""
    # Cartesian distance
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def distance_haversine(p1, p2):
    # Haversine distance
    import math
    lat1, lon1 = p1.x, p1.y
    lat2, lon2 = p2.x, p2.y
    radius = 6371  # [km]

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d


def distance(p1, p2):
    return distance_haversine(p1, p2)


# Date functions
def is_leap_year(year):
    """ if year is a leap year return True
        else return False """
    if year % 100 == 0:
        return year % 400 == 0
    return year % 4 == 0


def doy(date):
    """ given year, month, day return day of year
        Astronomical Algorithms, Jean Meeus, 2d ed, 1998, chap 7
        format: yyyymmdd"""
    y = int(date[:4])
    m = int(date[4:6])
    d = int(date[6:])
    if is_leap_year(y):
        k = 1
    else:
        k = 2
    return int(int((275 * m) / 9.0) - k * int((m + 9) / 12.0) + d - 30)


def dop(date, start="20170101"):
    """Returns the number of days bewteen the 'start' date and the 'date'
       format: yyyymmdd"""
    date_year = int(date[:4])
    start_year = int(start[:4])

    count_days = 0
    if date < start:
        return -1

    if date_year > start_year:
        for y in range(start_year, date_year):
            count_days += 365 if not is_leap_year(y) else 366
    count_days -= doy(start)
    count_days += doy(date)
    return count_days


def date_subtract(d, years=0, months=0, days=0):
    """"
    Routine to subtract a certain amount of years, months and days of a given date string

    d : str
        datestring in format yyyymmdd
    """
    if len(d) != 8:
        return "0"
    year = int(d[:4]) - years
    month = int(d[4:6])
    day = int(d[6:])
    l = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for i in range(months):
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    for i in range(days):
        day -= 1
        if day == 0:
            month -= 1
            if month == 0:
                month = 12
            day = l[month - 1]
    return str(year) + str(month).rjust(2, '0') + str(day).rjust(2, '0')


# Interpolation
def idw_2d(df, point, col, p=1):
    """
    IDW interpolation between a geodataframe and a point

    Parameters
    ----------
    df : geodataframe
        observation locations containing `col`
    point : shapely.Point
        location for which the interpolated value will be calculated
    col : str
        column name in `df` which will be used to calculate the interpolated value
    p : numeric
        power of the IDW formula

    Returns
    -------
    float
    """
    distance_to_point = df.geometry.apply(lambda x: distance(x, point))
    weight = 1 / (distance_to_point.pow(p))
    return (weight * df[col]).sum() / weight.sum()


def idw_1d(df, coordinate_col, point, col, p=1):
    """
    IDW interpolation between a geodataframe and a point

    Parameters
    ----------
    df : geodataframe
        observation locations containing `col`
    coordinate_col : str
        column that contains the coordinate (the geometry will not be used, see idw_2d)
    point : numeric
        location for which the interpolated value will be calculated.
    col : str
        column name in `df` which will be used to calculate the interpolated value
    p : numeric
        power of the IDW formula

    Returns
    -------
    float
    """
    distance_to_point = (df[coordinate_col] - point).abs()
    weight = 1 / (distance_to_point.pow(p))
    return (weight * df[col]).sum() / weight.sum()


def interpolate_over_geometries(df, coord_col, col, river, n, p=2):
    df = df.loc[~pd.isna(df[coord_col]), :]
    geometries = [river.interpolate(x, normalized=True) for x in np.linspace(0, 1, n)]
    coords = [river.project(x) for x in geometries]
    df_plot_locations = gpd.GeoDataFrame(geometry=geometries)
    df_plot_locations.loc[:, coord_col] = coords
    df_plot_locations.loc[:, 'values'] = df_plot_locations.apply(
        lambda x: idw_1d(df, coord_col, x[coord_col], col, p=p), axis=1)
    return df_plot_locations


# Data extraction from KNMI/RWS databases
def filter_date(data, period, date="date"):
    """
    filters a dataframe on a specified timeframe

    Parameters
    ----------
    data : pd.DataFrame
        pandas dataframe
    period : list
        list containing start and end filter
    date : str, optional
        name of the column containing the date information. Default is 'date'

    Returns
    -------
    Same dataframe as the input, filtered on a specified timeframe
    """
    return data.loc[np.logical_and(data[date] > period[0], data[date] <= period[1]), :]


def stations_available(stations, data, col, id_col='id', fraction_accepted=0.8, mode="full", keep_steps=False):
    """
    filters a dataframe with stations based on a dataframe with data at these locations (containing NaNs)

    Parameters
    ----------
    stations : pd.DataFrame
        dataframe with stations, featuring an 'id' column that is matched against a similar column in
        the `data` dataframe
    data : pd.DataFrame
        dataframe with observations at the same locations as the stations dataframe. Featuring an 'id'
        column as `stations`
    col : str
        name of the column from which the data will be selected
    id_col : str
        name of column that contains the unique station indication. Needs to be equal in both dataframes
    fraction_accepted : float, optional
        fraction of entries that must be not NaN
    mode : str, optional
        full
            return only those stations that are under the threshold of nans and are operational all the time
        reduced
            return only those stations that are under the threshold of nans
        all
            return all stations
    keep_steps : bool, optional
        keep the columns containing the selection procedure. The default is False

    Returns
    -------
    Filtered dataframe
    """
    col_names = list(stations.columns) + ['nan_count', 'size', 'available', 'full']

    data = data.loc[:, [id_col, col]].groupby(id_col)
    nan_count = data.agg(lambda x: pd.Series.sum(pd.isnull(x))).squeeze()
    size = data.count().squeeze()
    available = (nan_count / size) < (1 - fraction_accepted)
    full = (size == size.max())
    stations = stations.merge(pd.DataFrame(nan_count), how='left', on=id_col)
    stations = stations.merge(pd.DataFrame(size), how='left', on=id_col)
    stations = stations.merge(pd.DataFrame(available), how='left', on=id_col)
    stations = stations.merge(pd.DataFrame(full), how='left', on=id_col)

    stations.columns = col_names
    stations.loc[stations.available.isna(), 'available'] = False
    stations.loc[stations.full.isna(), 'full'] = False

    if keep_steps:
        s = slice(None, None)
    else:
        s = slice(None, -4)

    if mode == 'all':
        return stations.iloc[:, s]
    elif mode == 'reduced':
        return stations.loc[stations.available, :].iloc[:, s]
    elif mode == 'full':
        return stations.loc[np.logical_and(stations.full, stations.available), :].iloc[:, s]


def reduce_data_stations(stations, data, col, fun, id_field='id'):
    """reduce data in `data` column `col` with function `fun` and add it to stations"""
    data = data.loc[data[id_field].isin(stations[id_field])]
    reduced_data = data.loc[:, [id_field, col]].groupby(id_field).apply(lambda x: fun(x))[[col]]
    return stations.merge(reduced_data, how='left', on=id_field)


if __name__ == "__main__":
    plot()
    plt.show()
