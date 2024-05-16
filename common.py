import polars as pl
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def get_starting_df():
    return (
        pl.read_excel('Data for Student Hiring Project - Citizen Science .xlsx', read_options={"has_header": False})
            .rename({'column_1': 'user_id', 'column_2': 'timestamp'})
            .with_columns(pl.col('timestamp').str.to_datetime())
    )


def find_sessions(df):
    """Identifing the sessions. giving an id for each session. Also giving a numeric id to a user.
    """
    return (
        df.sort(['user_id', 'timestamp'])
        .with_columns(user_diff=(pl.col('user_id').ne(pl.col('user_id').shift())).fill_null(True))
        .with_columns(user=pl.col("user_diff").cum_sum())
        .with_columns(ts_diff=(pl.col('timestamp').diff().dt.total_seconds().fill_null(0)))
        .with_columns(C_new_session_mark=((pl.col('ts_diff') > 5 * 60) | pl.col("user_diff")))
        .with_columns(cont_session=pl.col('C_new_session_mark').cum_sum())
        .with_columns(new_session_mark=((pl.col('ts_diff') > 30 * 60) | pl.col("user_diff")))
        .with_columns(agg_session=pl.col('new_session_mark').cum_sum())

        .with_columns(U_seq=pl.col("timestamp").cum_count().over('user'))
        .with_columns(A_seq=pl.col("timestamp").cum_count().over('agg_session'))
        .with_columns(C_seq=pl.col("timestamp").cum_count().over('cont_session'))

        .with_columns(A_in_U=pl.col("agg_session").cum_count().over('user'))
        .with_columns(C_in_A=pl.col("cont_session").cum_count().over('agg_session'))
    )


# def add_regression_target(df):
#     return (
#         df
#         .with_columns(
#             ((pl.col("timestamp").max().over("agg_session") - pl.col('timestamp'))
#              .dt.total_seconds()).alias('target')
#         )
#     )


# def calc_y(df):
#     return (df['target'] < 5 * 60).alias('disengage')


def add_target(df):
    return (
        df
        .with_columns(
            target=(pl.col("timestamp").max().over("agg_session") - pl.col('timestamp'))
            .dt.total_seconds() < 5 * 60)
    )


def add_fold(df):
    groups = df['user']
    sgkf = StratifiedGroupKFold(n_splits=4)
    df_with_fold = (
        df
        .with_columns(pl.repeat(-1, df.shape[0]).alias('fold'))
    )
    for i, (_, test_index) in enumerate(sgkf.split(df_with_fold, df['target'], groups)):
        df_with_fold[test_index, 'fold'] = i
    return df_with_fold
