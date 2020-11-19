import os
import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import geomappy as mp
from geomappy.utils import progress_bar
from scipy.stats import pearsonr

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
df.loc[:, 'season'] = df.meting.apply(lambda x: "Autumn" if x in [0,2] else "Spring")


def shuffle_data(data):
    for col in list(data.columns)[:-1]:
        np.random.shuffle(data.loc[:, col].values)
    return data


def calc_r2(x, y):
    mask = ~np.logical_or(np.isnan(x), np.isnan(y))
    return pearsonr(x[mask], y[mask])[0] ** 2


for parameter in ["P_7d_max", "U_7d_max", "h_7d_max_above_current", "dop", "season", "river", "sampling"]:
    print(parameter)
    print("_____________________________________________________________________________")
    for item in ITEMS:
        # check if the file already exists
        if os.path.exists(f"data/item_variance/{parameter}_{item}.csv"):
            continue

        if parameter in ['river', 'sampling', 'season']:
            formula = f'{item} ~ C({parameter})'
        else:
            formula = f'{item} ~ {parameter}'

        perm_data = df.loc[:, [parameter, item]]
        model = smf.glm(formula=formula, data=perm_data, family=sm.families.NegativeBinomial()).fit(full_output=True)
        true_r2 = calc_r2(perm_data.iloc[:, 1], model.predict(perm_data.iloc[:, 0]))

        df_r2 = pd.DataFrame(np.full((permutations, 1), np.nan))
        print(item)
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
