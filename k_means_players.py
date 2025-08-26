import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import streamlit as st

# from IPython.display import display, Markdown
import google.generativeai as genai
import json
import plotly.express as px

def kmeans(df, n):
    le = LabelEncoder()

    player = df.Player
    nation = df.Nation
    squad = df.Squad
    comp = df.Comp

    df = df.drop(columns = ['Player', 'Nation', 'Squad', 'Comp'])

    df['Pos'] = le.fit_transform(df['Pos'])

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    k_means = KMeans(n_clusters = n, random_state=42).fit(df_scaled)

    df['cluster'] = k_means.labels_

    df['Pos'] = le.inverse_transform(df['Pos'])

    df['Player'] = player
    df['Nation'] = nation
    df['Squad'] = squad
    df['Comp'] = comp

    df['Loss'] = k_means.transform(df_scaled).min(axis=1)

    return df

def cr_df(df, clstr):
  return df.loc[df.cluster == clstr].copy()

New = pd.read_csv('Files/gmm_players.csv')

t0 = cr_df(New, 0)
t0 = kmeans(t0, 3)
t0_0 = cr_df(t0, 0)
t0_0['cluster'] = 0
t0_1 = cr_df(t0, 1)
t0_1['cluster'] = 1
t0_2 = cr_df(t0, 2)
t0_2['cluster'] = 2

t1 = cr_df(New, 1)
t1 = kmeans(t1, 6)
t1_0 = cr_df(t1, 0)
t1_0['cluster'] = 3
t1_1 = cr_df(t1, 1)
t1_1['cluster'] = 4
t1_2 = cr_df(t1, 2)
t1_2['cluster'] = 5
t1_3 = cr_df(t1, 3)
t1_3['cluster'] = 6
t1_4 = cr_df(t1, 4)
t1_4['cluster'] = 7
t1_5 = cr_df(t1, 5)
t1_5['cluster'] = 8

t2 = cr_df(New, 2)
t2 = kmeans(t2, 3)
t2_0 = cr_df(t2, 0)
t2_0['cluster'] = 9
t2_1 = cr_df(t2, 1)
t2_1['cluster'] = 10
t2_2 = cr_df(t2, 2)
t2_2['cluster'] = 11

t3 = cr_df(New, 3)
t3 = kmeans(t3, 1)
t3_0 = cr_df(t3, 0)
t3_0['cluster'] = 11

t4 = cr_df(New, 4)
t4 = kmeans(t4, 3)
t4_0 = cr_df(t4, 0)
t4_0['cluster'] = 12
t4_1 = cr_df(t4, 1)
t4_1['cluster'] = 13
t4_2 = cr_df(t4, 2)
t4_2['cluster'] = 14

t5 = cr_df(New, 5)
t5 = kmeans(t5, 1)
t5_0 = cr_df(t5, 0)
t5_0['cluster'] = 15

t6 = cr_df(New, 6)
t6 = kmeans(t6, 4)
t6_0 = cr_df(t6, 0)
t6_0['cluster'] = 16
t6_1 = cr_df(t6, 1)
t6_1['cluster'] = 17
t6_2 = cr_df(t6, 2)
t6_2['cluster'] = 18
t6_3 = cr_df(t6, 3)
t6_3['cluster'] = 19

t7 = cr_df(New, 7)
t7 = kmeans(t7, 6)
t7_0 = cr_df(t7, 0)
t7_0['cluster'] = 20
t7_1 = cr_df(t7, 1)
t7_1['cluster'] = 21
t7_2 = cr_df(t7, 2)
t7_2['cluster'] = 22
t7_3 = cr_df(t7, 3)
t7_3['cluster'] = 23
t7_4 = cr_df(t7, 4)
t7_4['cluster'] = 24
t7_5 = cr_df(t7, 5)
t7_5['cluster'] = 25

t8 = cr_df(New, 8)
t8 = kmeans(t8, 4)
t8_0 = cr_df(t8, 0)
t8_0['cluster'] = 26
t8_1 = cr_df(t8, 1)
t8_1['cluster'] = 27
t8_2 = cr_df(t8, 2)
t8_2['cluster'] = 28
t8_3 = cr_df(t8, 3)
t8_3['cluster'] = 29

t9 = cr_df(New, 9)
t9 = kmeans(t9, 2)
t9_0 = cr_df(t9, 0)
t9_0['cluster'] = 30
t9_1 = cr_df(t9, 1)
t9_1['cluster'] = 31

t10 = cr_df(New, 10)
t10 = kmeans(t10, 6)
t10_0 = cr_df(t10, 0)
t10_0['cluster'] = 32
t10_1 = cr_df(t10, 1)
t10_1['cluster'] = 33
t10_2 = cr_df(t10, 2)
t10_2['cluster'] = 34
t10_3 = cr_df(t10, 3)
t10_3['cluster'] = 35
t10_4 = cr_df(t10, 4)
t10_4['cluster'] = 36
t10_5 = cr_df(t10, 5)
t10_5['cluster'] = 37

t11 = cr_df(New, 11)
t11 = kmeans(t11, 6)
t11_0 = cr_df(t11, 0)
t11_0['cluster'] = 38
t11_1 = cr_df(t11, 1)
t11_1['cluster'] = 39
t11_2 = cr_df(t11, 2)
t11_2['cluster'] = 40
t11_3 = cr_df(t11, 3)
t11_3['cluster'] = 41
t11_4 = cr_df(t11, 4)
t11_4['cluster'] = 42
t11_5 = cr_df(t11, 5)
t11_5['cluster'] = 43

t12 = cr_df(New, 12)
t12 = kmeans(t12, 4)
t12_0 = cr_df(t12, 0)
t12_0['cluster'] = 44
t12_1 = cr_df(t12, 1)
t12_1['cluster'] = 45
t12_2 = cr_df(t12, 2)
t12_2['cluster'] = 46
t12_3 = cr_df(t12, 3)
t12_3['cluster'] = 47

t13 = cr_df(New, 13)
t13 = kmeans(t13, 3)
t13_0 = cr_df(t13, 0)
t13_0['cluster'] = 48
t13_1 = cr_df(t13, 1)
t13_1['cluster'] = 49
t13_2 = cr_df(t13, 2)
t13_2['cluster'] = 50

t14 = cr_df(New, 14)
t14 = kmeans(t14, 7)
t14_0 = cr_df(t14, 0)
t14_0['cluster'] = 51
t14_1 = cr_df(t14, 1)
t14_1['cluster'] = 52
t14_2 = cr_df(t14, 2)
t14_2['cluster'] = 53
t14_3 = cr_df(t14, 3)
t14_3['cluster'] = 54
t14_4 = cr_df(t14, 4)
t14_4['cluster'] = 55
t14_5 = cr_df(t14, 5)
t14_5['cluster'] = 56
t14_6 = cr_df(t14, 6)
t14_6['cluster'] = 57

t15 = cr_df(New, 15)
t15 = kmeans(t15, 1)
t15_0 = cr_df(t15, 0)
t15_0['cluster'] = 58

t16 = cr_df(New, 16)
t16 = kmeans(t16, 2)
t16_0 = cr_df(t16, 0)
t16_0['cluster'] = 59
t16_1 = cr_df(t16, 1)
t16_1['cluster'] = 60

t17 = cr_df(New, 17)
t17 = kmeans(t17, 2)
t17_0 = cr_df(t17, 0)
t17_0['cluster'] = 61
t17_1 = cr_df(t17, 1)
t17_1['cluster'] = 62

t18 = cr_df(New, 18)
t18 = kmeans(t18, 4)
t18_0 = cr_df(t18, 0)
t18_0['cluster'] = 63
t18_1 = cr_df(t18, 1)
t18_1['cluster'] = 64
t18_2 = cr_df(t18, 2)
t18_2['cluster'] = 65
t18_3 = cr_df(t18, 3)
t18_3['cluster'] = 66

t19 = cr_df(New, 19)
t19 = kmeans(t19, 4)
t19_0 = cr_df(t19, 0)
t19_0['cluster'] = 67
t19_1 = cr_df(t19, 1)
t19_1['cluster'] = 68
t19_2 = cr_df(t19, 2)
t19_2['cluster'] = 69
t19_3 = cr_df(t19, 3)
t19_3['cluster'] = 70

t20 = cr_df(New, 20)
t20 = kmeans(t20, 5)
t20_0 = cr_df(t20, 0)
t20_0['cluster'] = 71
t20_1 = cr_df(t20, 1)
t20_1['cluster'] = 72
t20_2 = cr_df(t20, 2)
t20_2['cluster'] = 73
t20_3 = cr_df(t20, 3)
t20_3['cluster'] = 74
t20_4 = cr_df(t20, 4)
t20_4['cluster'] = 75

t21 = cr_df(New, 21)
t21 = kmeans(t21, 1)
t21_0 = cr_df(t21, 0)
t21_0['cluster'] = 76

t22 = cr_df(New, 22)
t22 = kmeans(t22, 1)
t22_0 = cr_df(t22, 0)
t22_0['cluster'] = 77

t23 = cr_df(New, 23)
t23 = kmeans(t23, 3)
t23_0 = cr_df(t23, 0)
t23_0['cluster'] = 78
t23_1 = cr_df(t23, 1)
t23_1['cluster'] = 79
t23_2 = cr_df(t23, 2)
t23_2['cluster'] = 80

t24 = cr_df(New, 24)
t24 = kmeans(t24, 2)
t24_0 = cr_df(t24, 0)
t24_0['cluster'] = 81
t24_1 = cr_df(t24, 1)
t24_1['cluster'] = 82
t24_2 = cr_df(t24, 2)
t24_2['cluster'] = 83

t25 = cr_df(New, 25)
t25 = kmeans(t25, 4)
t25_0 = cr_df(t25, 0)
t25_0['cluster'] = 84
t25_1 = cr_df(t25, 1)
t25_1['cluster'] = 85
t25_2 = cr_df(t25, 2)
t25_2['cluster'] = 86
t25_3 = cr_df(t25, 3)
t25_3['cluster'] = 87

t26 = cr_df(New, 26)
t26 = kmeans(t26, 4)
t26_0 = cr_df(t26, 0)
t26_0['cluster'] = 88
t26_1 = cr_df(t26, 1)
t26_1['cluster'] = 89
t26_2 = cr_df(t26, 2)
t26_2['cluster'] = 90
t26_3 = cr_df(t26, 3)
t26_3['cluster'] = 91

t27 = cr_df(New, 27)
t27 = kmeans(t27, 2)
t27_0 = cr_df(t27, 0)
t27_0['cluster'] = 92
t27_1 = cr_df(t27, 1)
t27_1['cluster'] = 93

t28 = cr_df(New, 28)
t28 = kmeans(t28, 2)
t28_0 = cr_df(t28, 0)
t28_0['cluster'] = 94
t28_1 = cr_df(t28, 1)
t28_1['cluster'] = 95

t29 = cr_df(New, 29)
t29 = kmeans(t29, 1)
t29_0 = cr_df(t29, 0)
t29_0['cluster'] = 96

all_players = pd.concat([t0_0, t0_1, t0_2, t1_0, t1_1, t1_2, t1_3, t1_4, t1_5,
                 t2_0, t2_1, t2_2, t3_0, t4_0, t4_1, t4_2, t5_0, t6_0, t6_1, t6_2, t6_3, t7_0, t7_1, t7_2, t7_3, t7_4, t7_5,
                 t8_0, t8_1, t8_2, t8_3, t9_0, t9_1, t10_0, t10_1, t10_2, t10_3, t10_4, t10_5,
                 t11_0, t11_1, t11_2, t11_3, t11_4, t11_5, t12_0, t12_1, t12_2, t12_3,
                 t13_0, t13_1, t13_2,
                 t14_0, t14_1, t14_2, t14_3, t14_4, t14_5, t14_6,
                 t15_0, t16_0, t16_1,
                 t17_0, t17_1, t18_0, t18_1, t18_2, t18_3,
                 t19_0, t19_1, t19_2, t19_3,
                 t20_0, t20_1, t20_2, t20_3, t20_4,
                 t21_0, t22_0,
                 t23_0, t23_1, t23_2,
                 t24_0, t24_1, t24_2,
                 t25_0, t25_1, t25_2, t25_3,
                 t26_0, t26_1, t26_2, t26_3,
                 t27_0, t27_1,
                 t28_0, t28_1,
                 t29_0])

# Resetting the index
# all.reset_index(drop=True, inplace=True)

# Making the first row of 'all' empty
empty_list = [''] * len(all_players.columns)
empty = pd.DataFrame([empty_list], columns=all_players.columns)
all_players = pd.concat([empty, all_players])

all_players.to_csv('Files/k_means_players.csv', index = False)