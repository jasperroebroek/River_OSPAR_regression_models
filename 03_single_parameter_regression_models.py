import os
import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from geomappy.utils import progress_bar
from sklearn.metrics import r2_score


def shuffle_data(data):
    for col in list(data.columns)[:-1]:
        np.random.shuffle(data.loc[:, col].values)
    return data


def calc_r2(y, y_predict):
    mask = ~np.logical_or(np.isnan(y), np.isnan(y_predict))
    return r2_score(y[mask], y_predict[mask])


permutations = 1000

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

df = pd.read_csv("data/processed_data/df_hm_included.csv", index_col=0)
df_locations = gpd.read_file("data/processed_data/df_locations.geojson")

FIRST_ITEM = "plastic_6_packringen"
LAST_ITEM = "granulaat_korrels"
ind_item = [list(df.columns).index(x) for x in [FIRST_ITEM, LAST_ITEM]]
s_items = slice(ind_item[0], ind_item[1], None)

ITEMS = [item for item in list(df.columns)[s_items]]

df.loc[:, 'sampling'] = df.Gebiedscode.apply(lambda x: 'Reference' if "REFE" in x else 'Volunteer')
df.loc[:, 'river'] = df.Gebiedscode.apply(lambda x: x[0])
df = df.loc[df.river.isin(["W", "M"])]
df.loc[:, 'all'] = df.loc[:, ITEMS].sum(axis=1)
df.loc[:, 'season'] = df.meting.apply(lambda x: "Autumn" if x in [0, 2] else "Spring")
df = df.loc[df.date != 0, :]

df.date = pd.to_datetime(df.date, format="%Y%m%d")
df.loc[:, 'time_after_last'] = np.nan

for i, row in df.iterrows():
    date_differences = (row.date - df.loc[df.Gebiedscode == row.Gebiedscode, 'date']).dt.days
    previous_dates = date_differences[date_differences > 0]
    if previous_dates.size == 0:
        df.time_after_last.loc[i] = np.nan
    else:
        df.time_after_last.loc[i] = previous_dates.min()

for parameter in ["time_after_last", "dop", "season", "river", "sampling",
                  "P_2d_max", "P_7d_max", "P_14d_max", "P_1m_max", "P_6m_max",
                  "U_2d_max", "U_7d_max", "U_14d_max", "U_1m_max", "U_6m_max",
                  "h_2d_max_above_current", "h_7d_max_above_current", "h_14d_max_above_current",
                  "h_1m_max_above_current", "h_6m_max_above_current"]:
    print(parameter)
    print("_____________________________________________________________________________")
    for item in ITEMS:
        print(item)

        # check if the file already exists
        if os.path.exists(f"data/item_variance/{parameter}_{item}.csv"):
            continue

        if parameter in ['river', 'sampling', 'season']:
            formula = f'{item} ~ C({parameter})'
        else:
            formula = f'{item} ~ {parameter}'

        perm_data = df.loc[:, [parameter, item]]
        try:
            model = smf.glm(formula=formula, data=perm_data, family=sm.families.NegativeBinomial()).fit(full_output=True)
        except (ValueError, statsmodels.tools.sm_exceptions.PerfectSeparationError):
            continue

        true_r2 = calc_r2(perm_data.iloc[:, 1], model.predict(perm_data))

        df_r2 = pd.DataFrame(np.full((permutations, 1), np.nan))

        for i in range(permutations):
            progress_bar((i + 1) / permutations)
            while True:
                # Sometimes a model comes up that does not solve. Simply try again.
                try:
                    perm_data = shuffle_data(perm_data)
                    random_model = smf.glm(formula=formula, data=perm_data, family=sm.families.NegativeBinomial()).fit(full_output=True)
                except (ValueError, statsmodels.tools.sm_exceptions.PerfectSeparationError):
                    pass
                else:
                    break

            df_r2.loc[i, :] = calc_r2(perm_data.iloc[:, 1], random_model.predict(perm_data))
        print(f"   {((df_r2 < true_r2).sum()/permutations)[0]}")

        df_r2.to_csv(f"data/item_variance/{parameter}_{item}.csv")
