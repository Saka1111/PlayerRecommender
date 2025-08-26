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

st.set_page_config(layout = "wide")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv('Files/PlayerData.csv')


df.drop(columns = ['Rk', 'Born', 'Rk_stats_shooting', 'Nation_stats_shooting', 'Pos_stats_shooting',
                  'Comp_stats_shooting', 'Age_stats_shooting', 'Born_stats_shooting', '90s_stats_shooting', 'Gls_stats_shooting',
                  'PK_stats_shooting', 'PKatt_stats_shooting', 'xG_stats_shooting', 'npxG_stats_shooting',
                  'Rk_stats_passing', 'Nation_stats_passing', 'Pos_stats_passing', 'Comp_stats_passing', 'Age_stats_passing',
                  'Born_stats_passing', '90s_stats_passing', 'Ast_stats_passing', 'xAG_stats_passing', 'PrgP_stats_passing',
                  'Rk_stats_passing_types',	'Nation_stats_passing_types', 'Pos_stats_passing_types', 'Comp_stats_passing_types',
                  'Age_stats_passing_types', 'Born_stats_passing_types', '90s_stats_passing_types', 'Att_stats_passing_types',
                  'Cmp_stats_passing_types', 'Rk_stats_gca', 'Nation_stats_gca', 'Pos_stats_gca', 'Comp_stats_gca', 'Age_stats_gca',
                  'Born_stats_gca', '90s_stats_gca', 'Rk_stats_defense', 'Nation_stats_defense', 'Pos_stats_defense',
                  'Comp_stats_defense', 'Age_stats_defense', 'Born_stats_defense', '90s_stats_defense', 'Rk_stats_possession',
                  'Nation_stats_possession', 'Pos_stats_possession', 'Comp_stats_possession', 'Age_stats_possession',
                  'Born_stats_possession', '90s_stats_possession', 'Rk_stats_playing_time', 'Nation_stats_playing_time',
                  'Pos_stats_playing_time', 'Comp_stats_playing_time', 'Age_stats_playing_time', 'Born_stats_playing_time',
                  'MP_stats_playing_time', 'Min_stats_playing_time', '90s_stats_playing_time', 'Starts_stats_playing_time',
                  'Rk_stats_misc', 'Nation_stats_misc', 'Pos_stats_misc', 'Comp_stats_misc', 'Age_stats_misc', 'Born_stats_misc',
                  '90s_stats_misc', 'Rk_stats_keeper', 'Nation_stats_keeper', 'Comp_stats_keeper', 'Age_stats_keeper',
                  'Born_stats_keeper', 'MP_stats_keeper', 'Starts_stats_keeper', 'Min_stats_keeper', '90s_stats_keeper',
                  'Rk_stats_keeper_adv', 'Nation_stats_keeper_adv', 'Pos_stats_keeper_adv', 'Comp_stats_keeper_adv',
                  'Age_stats_keeper_adv', 'Born_stats_keeper_adv', '90s_stats_keeper_adv', 'Pos_stats_keeper', 'CrdY_stats_misc',
                  'CrdR_stats_misc', 'GA_stats_keeper_adv', 'PKA_stats_keeper_adv'], inplace = True)

PlayerData = df.loc[df.Min > 1000]

# Dropping all columns where there are nulls (apart from GK columns)

PlayerData = PlayerData.drop(columns = ['SoT%', 'G/Sh', 'G/SoT', 'Dist', 'npxG/Sh', 'Tkl%', 'Succ%', 'Tkld%', 'Mn/Sub', 'On-Off', 'Won%'])


# Making the Nation into 'tr Tur'
PlayerData.loc[PlayerData.Nation.isnull(), 'Nation'] = 'tr Tur'


PlayerData = PlayerData.fillna(0)
# PlayerData.isnull().sum()


# Removing one player due to sensitivity
PlayerData = PlayerData.loc[PlayerData.Player != 'Diogo Jota']

New = PlayerData.copy()

player = New.Player
nation = New.Nation
squad = New.Squad
comp = New.Comp
age = New.Age

New = New.drop(columns = ['Player', 'Nation', 'Squad', 'Comp', 'Age'])

le = LabelEncoder()
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()

New['Pos'] = le.fit_transform(New['Pos'])

scaler = MinMaxScaler()
N_scaled = scaler.fit_transform(New)

gmm = GaussianMixture(n_components=30, random_state=42)
gmm.fit(N_scaled)

New['cluster'] = gmm.predict(N_scaled)

New['Pos'] = le.inverse_transform(New['Pos'])

New['Player'] = player
New['Nation'] = nation
New['Squad'] = squad
New['Comp'] = comp
New['Age'] = age

New.to_csv('Files/gmm_players.csv', index = False)