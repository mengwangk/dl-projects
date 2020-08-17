import pandas as pd
import numpy as np


def format_tabular(original_file):
    df = pd.read_csv(original_file, header=0, sep=',', quotechar='"', parse_dates=[
                     'DrawDate'], dtype={'PrizeType': str})
    df_tabular = df.melt(
        id_vars=["DrawNo", "DrawDate"], var_name="PrizeType", value_name="LuckyNo")
    df_tabular = df_tabular.sort_values(
        ["DrawNo", "DrawDate", "PrizeType"], ascending=True)
    df_tabular['DrawDate'] = pd.to_datetime(
        df_tabular['DrawDate'], format='%Y%m%d')
    df_tabular.reset_index(inplace=True, drop=True)
    return df_tabular

