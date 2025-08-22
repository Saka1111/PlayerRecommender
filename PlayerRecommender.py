import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import streamlit as st

from IPython.display import display, Markdown
import google.generativeai as genai
import json
import plotly.express as px

st.set_page_config(layout = "wide")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv('PlayerData.csv')


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

all = pd.concat([t0_0, t0_1, t0_2, t1_0, t1_1, t1_2, t1_3, t1_4, t1_5,
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
empty_list = [''] * len(all.columns)
empty = pd.DataFrame([empty_list], columns=all.columns)
all = pd.concat([empty, all])

players = list(all.Player.sort_values().unique())

# Canvas
st.set_page_config(page_title='Player Comparison Tool', page_icon='⚽️', layout='wide')

# Creating 2 tabs
tab1, tab2 = st.tabs(['The Player Recommender', 'Stats Dictionary'])

# Tab 1: Player Comparison Tool
with tab1:
  st.title('The Player Recommender ⚽️', width = 'stretch')
      
  system_instructions = ("You are a soccer data analyst and scout."
                    "You will always analyse based on the data provided to you and you never complain about the data that is given to you."
                    "You always write in bullet points and your reasonings are clear and concise.")
  
  system_instructions1 = ("You are a soccer data analyst and scout, and a skilled Python programmer"
                    "You will always analyse based on the data provided to you and you never complain about the data that is given to you."
                    "You always write in bullet points and your reasonings are clear and concise."
                    "The code you generate should be valid Python code for a Streamlit app.")

  # player = input('Input a player? ')
  try:
    player = st.selectbox('Input a player:', list(players), placeholder='Select a player')
  except:
    st.write('Player not in database')

  # age = int(input('What is the max age of the recommended players? '))
  age = st.number_input('What is the max age of the recommended players?', min_value=16, max_value=40, value=25)

  # search = str(input('Do you want a specific or broad search? '))
  search = st.selectbox('Do you want a specific or broad search?', ['specific', 'broad'])

  if search == 'specific':
    if player in all.Player.unique():
      x = all.loc[all.Player == player][['cluster']]
      c = x.iloc[0,0]
      df1 = all.loc[all.cluster == c].sort_values(by = 'Loss', ascending = True).copy()
      try:
        df1 = df1.loc[df1.Age <= age]
      except:
        st.write('')
      
      data_json = df1.to_json(orient='records')

      message = f'Given their statistics in the 24/25 soccer season, can you give me the best suited players to replace {player} out of these players in the JSON: {data_json}.'

      prompt = f'{system_instructions} {message} Just note that the "cluster" and "Loss" columns are not important for the analysis, they are just for the clustering and similarity purposes.'

      if st.button('Show Recommended Players'):
          with st.spinner('Generating recommendations...'):
              st.session_state['recommended_players'] = df1.loc[df1.Player != player][['Player', 'Pos', 'Squad', 'Age']]

      # Password input
      password = st.text_input("Enter your Gemini API key", type="password")
      API = password

      # # Opening the key using API variable
      
      # with open("gemini_key.txt", "rt") as f:
      #     API = f.read()

      # # For safety
      # f.close()

      # This is how we load the model in!!!
      genai.configure(api_key = API) 
      model = genai.GenerativeModel("gemini-2.5-flash")

      if st.button('AI Recommendation'):
          with st.spinner('Generating AI response...'):
              try:
                response = model.generate_content(prompt)
                st.session_state['ai_recommendation'] = response.text
              except:
                 st.error("Input a valid API Key")

      if st.button('Generate graphs'):
          with st.spinner('Generating graphs...'):
              message1 = f'Given their statistics in the 24/25 soccer season, can you give me a streamlit script which generates graphs and figures of {player} and the players most suited to them. The stats and players are in {data_json}. Only choose the most suitable stats and types of graphs, for example radar charts and bar charts. Make sure to include the player names in the graphs and figures. The script should be able to run in streamlit and should not include any unnecessary comments or explanations. Create the graphs solely using streamlit'
              # prompt1 = f"""
              # {system_instructions1}{message1}
              # When creating graphs use {df1}
              # Only choose the most suitable stats and types of graphs, for example radar charts and bar charts.
              # Make sure to include the player names in the graphs and figures.
              # The script should be able to run in streamlit and should not include any unnecessary comments or explanations.
              # Create the graphs solely using streamlit.
              # Use the columns in this list only {df1.columns.tolist()}
              # I WANT NO ERRORS IN THE CODE YOU GENERATE!!!!!!!!!!
              # """
              prompt1 = f"""
              {system_instructions1}{message1}
              When creating graphs use {data_json}
              1st create a radar chart with the key stats of the players
              2nd create a bar charts for each key stats of the players
              Include {player} and their similar players in {data_json}
              Use the columns in this list only {df1.columns.tolist()}
              Create a valid response.text so I can use it in my own script
              Terminate all brackets and quotes and strings
              Complete all code
              Do not include any extra text, just produce python code
              Produce enough code to show show a maximum of 5 graphs
              I WANT NO ERRORS IN THE CODE YOU GENERATE!!!!!!!!!!
              """
              # prompt1 = f'{system_instructions1} {message1} Just note that the "cluster" and "Loss" columns are not important for the analysis, they are just for the clustering and similarity purposes. Please ONLY output valid Python code for a Streamlit app. I DO NOT want any syntax warnings within the code and NO ERRORS AT ALL. Complete every bit of code you generate. Do NOT include any explanations, markdown, or comments. The code should be ready to run with exec(). Also, do not include any import statements that require installation of additional packages, only use packages that are already installed in the environment. When creating new dataframes, do it the same way I have done it in this code, by using pd.DataFrame(). Also write it in a python formatting code block, starting with ```python and ending with ```. Do not create you\'re own dataframe, use the {df1}, and create graphs from there. I want no errors with this column "PSxG+/-"'
              try:
                response1 = model.generate_content(prompt1)
                t = response1.text
                # st.write(t)
                t = str(t).replace('```python', '').replace('```', '')
                st.session_state['graph_code'] = t
              except:
                  st.error("Input a valid API Key")

      if 'recommended_players' in st.session_state:
          st.write(st.session_state['recommended_players'])

      if 'ai_recommendation' in st.session_state:
          st.write(st.session_state['ai_recommendation'])

      if 'graph_code' in st.session_state:
          try:
              exec(st.session_state['graph_code'])
          except Exception as e:
              st.error(f"Error running generated code: {e}")
    else:
      st.write('Player not in database')
    
  elif search == 'broad':
    if player in all.Player.unique():
      x = all.loc[all.Player == player][['cluster']]
      c = x.iloc[0,0]
      df1 = all.loc[all.cluster == c].sort_values(by = 'Loss', ascending = True).copy()
      try:
        df1 = df1.loc[df1.Age <= age]
      except:
        st.write('')
      
      data_json = df1.to_json(orient='records')

      message = f'Given their statistics in the 24/25 soccer season, can you give me the best suited players to replace {player} out of these players in the JSON: {data_json}.'

      prompt = f'{system_instructions} {message} Just note that the "cluster" and "Loss" columns are not important for the analysis, they are just for the clustering and similarity purposes.'

      if st.button('Show Recommended Players'):
          with st.spinner('Generating recommendations...'):
              st.session_state['recommended_players'] = df1.loc[df1.Player != player][['Player', 'Pos', 'Squad', 'Age']]

      # Password input
      password = st.text_input("Enter your Gemini API key", type="password")
      API = password


      # with open("gemini_key.txt", "rt") as f:
      #     API = f.read()

      # # For safety
      # f.close()

      # This is how we load the model in!!!
      genai.configure(api_key = API) 
      model = genai.GenerativeModel("gemini-2.5-flash")

      if st.button('AI Recommendation'):
          with st.spinner('Generating AI response...'):
              response = model.generate_content(prompt)
              st.session_state['ai_recommendation'] = response.text

      if st.button('Generate graphs'):
          with st.spinner('Generating graphs...'):
              message1 = f'Given their statistics in the 24/25 soccer season, can you give me a streamlit script which generates graphs and figures of {player} and the players most suited to them. The stats and players are in {data_json}. Only choose the most suitable stats and types of graphs. Make sure to include the player names in the graphs and figures. The script should be able to run in streamlit and should not include any unnecessary comments or explanations.'
              prompt1 = f'{system_instructions} {message1} Just note that the "cluster" and "Loss" columns are not important for the analysis, they are just for the clustering and similarity purposes. Please ONLY output valid Python code for a Streamlit app. I DO NOT want any syntax warnings within the code or any errors. Do NOT include any explanations, markdown, or comments. The code should be ready to run with exec(). Also, do not include any import statements that require installation of additional packages, only use packages that are already installed in the environment matplotlib, plotly and streamlit are already installed. I want alll code to be completed so no errors, so always close brackets and quotes etc. For example I want no errors like this "unterminated string literal"'
              response1 = model.generate_content(prompt1)
              t = response1.text
              t = str(t).replace('```python', '').replace('```', '')
              st.session_state['graph_code'] = t

      if 'recommended_players' in st.session_state:
          st.write(st.session_state['recommended_players'])

      if 'ai_recommendation' in st.session_state:
          st.write(st.session_state['ai_recommendation'])

      if 'graph_code' in st.session_state:
          try:
              exec(st.session_state['graph_code'])
          except Exception as e:
              st.error(f"Error running generated code: {e}")
    else:
      st.write('Player not in database')

  else:
    st.write('Not a valid search')

# Tab 2: Stats Dictionary
with tab2:
   st.markdown("""
               # Dictionary of player stats in the 24/25 season

|Column|Definition|
|------|----------|
|Player|Name of the player|
|Pos|Position (Positions) of the player|
|Squad|The team the player plays for|
|MP|Matches played|
|Starts|How many times the player started|
|Min|How many minutes the player played|
|90s|How many 90 minutes played - Minutes played divded by 90 (1dp)|
|Gls|Goals scored|
|Ast|Assists made|
|G+A|Total goals and assists|
|G-PK|Non-penalty goals|
|PK|Penalty kicks made|
|PKatt|Penalty kicks attempted|
|CrdY|Yellow cards|
|CrdR|Red cards|
|xG|Expected Goals|
|npxG|Non-penalty expected goals|
|xAG|Expected goals assisted|
|npxG+xAG|Non-penalty expected goals plus expected assisted goals|
|PrgC|Progressive carries|
|PrgP|Progressive passes|
|PrgR|Progressive passes received|
|G+A-PK|Non-penalty goals and assists per 90|
|xG+xAG|Expected goals plus expected goals assisted per 90|
|Sh|Total shots (not incuding penalty kicks)|
|SoT|Shots on target (not including penalty kicks)|
|SoT%|Percentage of shots on target|
|Sh/90|Shots total per 90|
|SoT/90|Shots on target per 90|
|G/Sh|Goals per shot|
|G/SoT|Goals per shot on target|
|Dist|Average shot distance|
|FK|Shots from free kicks|
|npxG/Sh|Non-penalty expected goals per shot|
|G-xG|Goals minus expected goals|
|np:G-xG|Non-penalty goals minus non-penalty expected goals|
|Cmp|Passes completed|
|Att|Passess attempted|
|Cmp%|Pass completion percentage|
|TotDist|Total passing distance|
|PrgDist|Progressive passing distance|
|xA|Expected assists|
|A-xAG|Assists minus expected goals assisted|
|KP|Key Passes (passes leading to a shot)|
|1/3|Passes into the final third|
|PPA|Passes into the penalty area|
|CrsPA|Crosses into the penalty area|
|Live|Live-ball passes|
|Dead|Dead-ball passes|
|FK_stats_passing_types|Passes from free-kicks|
|TB|Through balls|
|Sw|Switches|
|Crs|Crosses|
|TI|Throw-ins taken|
|CK|Corner kicks|
|In|Inswinging corner kicks|
|Out|Outswinging corner kicks|
|Str|Straight corner kicks|
|Off|Passes offsides|
|Blocks|Passes blocked|
|SCA|Shot-creating actions|
|SCA90|Shot-creating actions per 90|
|PassLive|Completed live-ball passes that lead to a shot attempt|
|PassDead|Completed dead-ball passes that lead to a shot attempt|
|TO|Successful take-ons that lead to a shot attempt|
|Sh_stats_gca|Shots that lead to another shot attempt|
|Fld|Fouls drawn that lead to a shot attempt|
|Def|Defensive action that leads to a shot attempt|
|GCA|Goal-creating actions|
|GCA90|Goal-creating actions per 90|
|Tkl|Tackles|
|TklW|Tackles won|
|Def 3rd|Tackles in the defensive third|
|Mid 3rd|Tackels in the middle third|
|Att 3rd|Tackles in the attacking third|
|Att_stats_defense|Dribbles challenged|
|Tkl%|Percentage of dribblers challenged|
|Lost|Unsuccessful challenged|
|Blocks_stats_defense|Blocks|
|Sh_stats_defense|Shots blocked|
|Pass|Passes blocked|
|Int|Interceptions|
|Tkl+Int|number of tackles plus interceptions|
|Clr|Clearances|
|Err|Errors leading to an opponents shot|
|Touches|Number of touches|
|Def Pen|Touches in the defensive penalty area|
|Def 3rd_stats_possession|Touches in the defensive third|
|Mid 3rd_stats_possession|Touches in the midfield third|
|Att 3rd_stats_possession|Touches in the attacking third|
|Att Pen|Touches in the attacking penalty area|
|Live_stats_possession|Live-ball touches|
|Att_stats_possession|Take-ons attempted|
|Succ|Successful take-ons|
|Succ%|Percentage of successful take-ons|
|Tkld|Times tackled during a take-on|
|Tkld%|Tackled during a take-on percentage|
|Carries|Number of carries|
|TotDist_stats_possession|Total carrying distance|
|PrgDist_stats_possession|Progressive carrying distance|
|PrgC_stats_possession|Progressive carries|
|1/3_stats_possession|Carries into the final third|
|CPA|Carries into the penalty area|
|Mis|Miscontrols|
|Dis|Dispossessed|
|Rec|Passes received|
|PrgR_stats_possession|Progressive passes received|
|Mn/MP|Minutes per matches played|
|Min%|Percentage of minutes played|
|Mn/Start|Minutes per match started|
|Compl|Complete matches played|
|Subs|Substitute appearances|
|Mn/Sub|Minutes per substitution|
|unSub|Matches as unused sub|
|PPM|Points per match|
|onG|Goals scored by team while on the pitch|
|onGA|Goals allowed by team while on the pitch|
|+/-|Goals scored minus goals allowed while on the pitch|
|+/-90|Goals scored minus goals allowed while on the pitch per 90|
|On-Off|Net goals per 90 by the team while on the pitch|
|onxG|Expected goals by the team while on the pitch|
|onxGA|Expected goals allowed by the team while on the pitch|
|xG+/-|Expected goals minus expected goals allowed by team while on the pitch|
|xG+/-90|Expected goals minus expected goals allowed by team while on the pitch per 90|
|2CrdY|Second yellow cards|
|Fls|Fouls committed|
|Fld_stats_misc|Fouls drawn|
|Off_stats_misc|Offsides|
|Crs_stats_misc|Crosses|
|Int_stats_misc|Interceptions|
|TklW_stats_misc|Tackles won|
|PKwon|Penalty kicks won|
|PKcon|Penalty kicks conceded|
|OG|Own goals|
|Recov|Ball recoveries|
|Won|Aerial duels won|
|Lost_stats_misc|Aerial duels lost|
|Won%|Percentage of Aerial duels won|
|GA|Goals Against|
|GA90|Goals against per 90|
|SoTA|Shots on target against|
|Saves|Saves made|
|Save%|Save percentage|
|W|Wins|
|D|Draws|
|L|Losses|
|CS|Clean sheets|
|CS%|Clean sheet percentage|
|PKatt_stats_keeper|Penalty kicks attempted|
|PKA|Penalty kicks allowed|
|PKsv|Penalty kick saves|
|PKm|Penalty kicks missed|
|FK_stats_keeper_adv|Free-kick goals against|
|CK_stats_keeper_adv|Corner kick goals against|
|OG_stats_keeper_adv|Own goals scored against|
|PSxG|Post-shot expected goals|
|PSxG/SoT|Post-shot expected goals per shot on target|
|PSxG+/-|Post-shot expected goals minus goals allowed|
|/90|Post-shot expected goals minus goals allowed per 90|
|Cmp_stats_keeper_adv|Passes completed longer than 40 yards|
|Att_stats_keeper_adv|Passes attempted longer than 40 yards||
|Cmp%_stats_keeper_adv|Percentage pass completion (longer than 40 yards)|
|Att (GK)|Passes attempted (not including goal kicks)|
|Thr|Throws attempted|
|Launch%|Percentage of passes more than 40 yards (not including goal kicks)|
|AvgLen|Average length of pass, in yards (not including goal kicks)|
|Opp|Crosses faced|
|Stp|Crosses stopped|
|Stp%|Percentage of crosses stopped|
|#OPA|Defensive actions outside the penalty area|
|#OPA/90|Defensive actions outside the penalty area per 90|
|AvgDist|Average distance of defensive actions|

**From the [FBref.com](https://fbref.com/en/comps/Big5/2024-2025/stats/players/2024-2025-Big-5-European-Leagues-Stats) website and [Kaggle](https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025) dataset.**""")