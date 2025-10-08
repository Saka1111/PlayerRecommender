import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import streamlit as st
import plotly.express as px
import altair as alt
import plotly.graph_objects as go

import google.generativeai as genai
import json

st.set_page_config(layout = "wide")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

all_players = pd.read_csv('Files/k_means_players.csv')
New = pd.read_csv('Files/gmm_players.csv')

players = list(all_players.Player.sort_values())

def load_stat_mapping(filepath):
  with open(filepath, 'r') as f:
    string_keys_mapping = json.load(f)
    
  int_keys_mapping = {int(k): v for k, v in string_keys_mapping.items()}
  return int_keys_mapping

stat_mapping = load_stat_mapping('Files/s_stats.json')

big_stat_mapping = load_stat_mapping('Files/b_stats.json')

base_cols = ['Player', 'Pos', 'Squad', 'Age']

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

  try:
    player = st.selectbox('Input a player:', list(players), placeholder='Select a player')
  
  except:
    st.write('Player not in database')

  age = st.number_input('What is the max age of the recommended players?', min_value=16, max_value=40, value=25)

  search = st.selectbox('Do you want a specific or broad search?', ['specific', 'broad'])

  if search == 'specific':
    if player in all_players.Player.unique():
      x = all_players.loc[all_players.Player == player][['cluster']]
      c = x.iloc[0,0]
      df1 = all_players.loc[all_players.cluster == c].sort_values(by = 'Loss', ascending = True).copy()
      
      try:
        df1 = df1.loc[df1.Age <= age]

      except:
        st.write('')

      if player in df1.Player.unique():
        data_json = df1.to_json(orient='records')

      else:
        player_df = all_players.loc[all_players.Player == player]
        combined_df = pd.concat([player_df, df1])
        data_json = combined_df.to_json(orient='records')
  
      # Recommended Players
      if st.button('Show Recommended Players'):
        with st.spinner('Generating recommendations...'):
          if player in df1.Player.unique():
            if 'GK' in df1.loc[df1.Player == player]['Pos'].values[0]:
              st.session_state['recommended_players'] = df1[base_cols + ['GA', 'GA90', 'SoTA', 'Saves', 'Save%', 'W', 'D', 'L', 'CS', 'CS%', 'PSxG', 'PSxG/SoT', 'PSxG+/-', '/90', 'Cmp_stats_keeper_adv', 'Att_stats_keeper_adv', 'Cmp%_stats_keeper_adv', 'Att (GK)', 'Thr', 'Launch%', 'AvgLen', 'Opp', 'Stp', 'Stp%', '#OPA', '#OPA/90', 'AvgDist']]
            
            else:
              stats = stat_mapping.get(c, [])
              cluster_cols = base_cols + stats
              st.session_state['recommended_players'] = df1[cluster_cols]

          else:
            if 'GK' in combined_df.loc[combined_df.Player == player]['Pos'].values[0]:
              st.session_state['recommended_players'] = combined_df[base_cols + ['GA', 'GA90', 'SoTA', 'Saves', 'Save%', 'W', 'D', 'L', 'CS', 'CS%', 'PSxG', 'PSxG/SoT', 'PSxG+/-', '/90', 'Cmp_stats_keeper_adv', 'Att_stats_keeper_adv', 'Cmp%_stats_keeper_adv', 'Att (GK)', 'Thr', 'Launch%', 'AvgLen', 'Opp', 'Stp', 'Stp%', '#OPA', '#OPA/90', 'AvgDist']]
            
            else:
              stats = stat_mapping.get(c, [])
              cluster_cols = base_cols + stats
              st.session_state['recommended_players'] = combined_df[cluster_cols]

      # Graphs
      if st.button('Generate graphs'):
        with st.spinner('Generating graphs...'):

          graph_list = []

          if player in all_players.Player.unique():
            x = all_players.loc[all_players.Player == player][['cluster']]
            c = x.iloc[0,0]
            df1 = all_players.loc[all_players.cluster == c].sort_values(by = 'Loss', ascending = True).copy()

          try:
            df1 = df1.loc[df1.Age <= age]

          except:
            st.write('')

          if player in df1.Player.unique():
            data_json = df1.to_json(orient='records')
            selected_players_df = pd.read_json(data_json)

          else:
            player_df = all_players.loc[all_players.Player == player]
            combined_df = pd.concat([player_df, df1])
            data_json = combined_df.to_json(orient='records')
            selected_players_df = pd.read_json(data_json)

          # Creating radar chart
          if 'GK' in selected_players_df.loc[selected_players_df.Player == player]['Pos'].values[0]:
              bar_chart_metrics = ['GA', 'GA90', 'SoTA', 'Saves', 'Save%', 'W', 'D', 'L', 'CS', 'CS%', 'PSxG']

          else:
            bar_chart_metrics = stat_mapping.get(c, [])
          
          scaler = MinMaxScaler()
          scaled_data = scaler.fit_transform(selected_players_df[bar_chart_metrics])
          scaled_df = pd.DataFrame(scaled_data, columns=bar_chart_metrics)
          scaled_df['Player'] = selected_players_df['Player'].reset_index(drop=True)

          radar = go.Figure()

          labels = [stat.title() for stat in bar_chart_metrics]

          for i, r in scaled_df.iterrows():
            player_name = r['Player']
            values = r[bar_chart_metrics].tolist()

            values.append(values[0])

            radar.add_trace(go.Scatterpolar(
              r = values,
              theta = labels + [labels[0]],
              fill = 'toself',
              name = player_name
            ))

          radar.update_layout(
            polar = dict(
              radialaxis = dict(
                visible = True,
                range = [0, 1]
              )
          ),
          showlegend = True,
          title = f'Radar Chart of Key Stats for {player} and Similar Players'
          )

          graph_list.append(radar)

          # Creating bar charts
          if 'GK' in selected_players_df.loc[selected_players_df.Player == player]['Pos'].values[0]:
              bar_chart_metrics = ['GA', 'GA90', 'SoTA', 'Saves', 'Save%', 'W', 'D', 'L', 'CS', 'CS%', 'PSxG']

          else:
            bar_chart_metrics = stat_mapping.get(c, [])

          for metric in bar_chart_metrics:
              bar_chart = alt.Chart(selected_players_df).mark_bar().encode(
                y=alt.Y(metric, axis=alt.Axis(title=metric)),
                x=alt.X('Player', sort='x', axis=alt.Axis(title='Player')),
                color=alt.Color('Player', legend=None),
                tooltip=['Player', alt.Tooltip(metric, format=".2f")]
              ).properties(
                title=f'{metric}'
              )

              graph_list.append(bar_chart)

          st.session_state['graph_code'] = graph_list


      # Password input
      password = st.text_input("Enter your Gemini API key", type="password")
      API = password

      genai.configure(api_key = API) 
      model = genai.GenerativeModel("gemini-2.5-flash")

      # AI Recommendation
      if st.button('AI Recommendation'):
        with st.spinner('Generating AI response...'):

          message = f'Given their statistics in the 24/25 soccer season, can you give me the best suited players to replace {player} out of these players in the JSON: {data_json}.'

          prompt = f"""
          {system_instructions}{message}
          The players recommended should be similar to {player} but also be an improvement on them.
          Prioritise improvement.
          Do not recommend players that are in the same squad as {player}
          Just note that the "cluster" and "Loss" columns are not important for the analysis, they are just for the clustering and similarity purposes.
          I want no errors when I do response.text, make sure any error with this does not occur
          Please provide pros and cons of each player you recommend
          And give a final evaluation
          """

          try:
            response = model.generate_content(prompt)
            st.session_state['ai_recommendation'] = response.text
            text = response.text

          except:
            st.error("Input a valid API Key")

      # States
      if 'recommended_players' in st.session_state:
        st.write(st.session_state['recommended_players'])

      if 'graph_code' in st.session_state:
        for graph in st.session_state['graph_code']:
          if isinstance(graph, go.Figure):
            st.plotly_chart(graph, use_container_width=True)

          else:
            st.altair_chart(graph, use_container_width=True)

      if 'ai_recommendation' in st.session_state:
        st.write(st.session_state['ai_recommendation'])

    else:
      st.write('Player not in database')
    
  elif search == 'broad':
    if player in New.Player.unique():
      x = New.loc[New.Player == player][['cluster']]
      c = x.iloc[0,0]
      df1 = New.loc[New.cluster == c].sort_values(by = 'Player').copy()
      
      try:
        df1 = df1.loc[df1.Age <= age]

      except:
        st.write('')
      
      if player in df1.Player.unique():
        data_json = df1.to_json(orient='records')

      else:
        player_df = New.loc[New.Player == player]
        combined_df = pd.concat([player_df, df1])
        data_json = combined_df.to_json(orient='records')

      # Recommended Players
      if st.button('Show Recommended Players'):
        with st.spinner('Generating recommendations...'):

          if player in df1.Player.unique():
            if 'GK' in df1.loc[df1.Player == player]['Pos'].values[0]:
              st.session_state['recommended_players'] = df1[base_cols + ['GA', 'GA90', 'SoTA', 'Saves', 'Save%', 'W', 'D', 'L', 'CS', 'CS%', 'PSxG', 'PSxG/SoT', 'PSxG+/-', '/90', 'Cmp_stats_keeper_adv', 'Att_stats_keeper_adv', 'Cmp%_stats_keeper_adv', 'Att (GK)', 'Thr', 'Launch%', 'AvgLen', 'Opp', 'Stp', 'Stp%', '#OPA', '#OPA/90', 'AvgDist']]
            
            else:
              stats = big_stat_mapping.get(c, [])
              cluster_cols = base_cols + stats
              st.session_state['recommended_players'] = df1[cluster_cols]

          else:
            if 'GK' in combined_df.loc[combined_df.Player == player]['Pos'].values[0]:
              st.session_state['recommended_players'] = combined_df[base_cols + ['GA', 'GA90', 'SoTA', 'Saves', 'Save%', 'W', 'D', 'L', 'CS', 'CS%', 'PSxG', 'PSxG/SoT', 'PSxG+/-', '/90', 'Cmp_stats_keeper_adv', 'Att_stats_keeper_adv', 'Cmp%_stats_keeper_adv', 'Att (GK)', 'Thr', 'Launch%', 'AvgLen', 'Opp', 'Stp', 'Stp%', '#OPA', '#OPA/90', 'AvgDist']]
            
            else:
              stats = big_stat_mapping.get(c, [])
              cluster_cols = base_cols + stats
              st.session_state['recommended_players'] = combined_df[cluster_cols]

      # Graphs
      if st.button('Generate graphs'):
        with st.spinner('Generating graphs...'):

          graph_list = []
          
          if player in New.Player.unique():
            x = New.loc[New.Player == player][['cluster']]
            c = x.iloc[0,0]
            df1 = New.loc[New.cluster == c].sort_values(by = 'Player').copy()

          try:
            df1 = df1.loc[df1.Age <= age]

          except:
            st.write('')

          if player in df1.Player.unique():
            data_json = df1.to_json(orient='records')
            selected_players_df = pd.read_json(data_json)

          else:
            player_df = all_players.loc[all_players.Player == player]
            combined_df = pd.concat([player_df, df1])
            data_json = combined_df.to_json(orient='records')
            selected_players_df = pd.read_json(data_json)

          # Creating radar chart
          if 'GK' in selected_players_df.loc[selected_players_df.Player == player]['Pos'].values[0]:
              bar_chart_metrics = ['GA', 'GA90', 'SoTA', 'Saves', 'Save%', 'W', 'D', 'L', 'CS', 'CS%', 'PSxG']

          else:
            bar_chart_metrics = big_stat_mapping.get(c, [])
          
          scaler = MinMaxScaler()
          scaled_data = scaler.fit_transform(selected_players_df[bar_chart_metrics])
          scaled_df = pd.DataFrame(scaled_data, columns=bar_chart_metrics)
          scaled_df['Player'] = selected_players_df['Player'].reset_index(drop=True)

          radar = go.Figure()

          labels = [stat.title() for stat in bar_chart_metrics]

          for i, r in scaled_df.iterrows():
            player_name = r['Player']
            values = r[bar_chart_metrics].tolist()

            values.append(values[0])

            radar.add_trace(go.Scatterpolar(
              r = values,
              theta = labels + [labels[0]],
              fill = 'toself',
              name = player_name
            ))

          radar.update_layout(
            polar = dict(
              radialaxis = dict(
                visible = True,
                range = [0, 1]
              )
          ),
          showlegend = True,
          title = f'Radar Chart of Key Stats for {player} and Similar Players'
          )

          graph_list.append(radar)


          # Creating bar charts
          if 'GK' in selected_players_df.loc[selected_players_df.Player == player]['Pos'].values[0]:
              bar_chart_metrics = ['GA', 'GA90', 'SoTA', 'Saves', 'Save%', 'W', 'D', 'L', 'CS', 'CS%', 'PSxG']

          else:
            bar_chart_metrics = stat_mapping.get(c, [])

          for metric in bar_chart_metrics:
              bar_chart = alt.Chart(selected_players_df).mark_bar().encode(
                y=alt.Y(metric, axis=alt.Axis(title=metric)),
                x=alt.X('Player', sort='x', axis=alt.Axis(title='Player')),
                color=alt.Color('Player', legend=None),
                tooltip=['Player', alt.Tooltip(metric, format=".2f")]
              ).properties(
                title=f'{metric}'
              )

              graph_list.append(bar_chart)
          st.session_state['graph_code'] = graph_list

      # Password input
      password = st.text_input("Enter your Gemini API key", type="password")
      API = password

      genai.configure(api_key = API) 
      model = genai.GenerativeModel("gemini-2.5-flash")

      # AI Recommendation
      if st.button('AI Recommendation'):
        with st.spinner('Generating AI response...'):
          message = f'Given their statistics in the 24/25 soccer season, can you give me the best suited players to replace {player} out of these players in the JSON: {data_json}.'

          prompt = f"""
          {system_instructions}{message}
          The players recommended should be similar to {player} but also be an improvement on them.
          Prioritise improvement.
          Do not recommend players that are in the same squad as {player}
          Just note that the "cluster" and "Loss" columns are not important for the analysis, they are just for the clustering and similarity purposes.
          I want no errors when I do response.text, make sure any error with this does not occur
          Please provide pros and cons of each player you recommend
          And give a final evaluation
          """

          response = model.generate_content(prompt)
          st.session_state['ai_recommendation'] = response.text
          text = response.text

      # States
      if 'recommended_players' in st.session_state:
        st.write(st.session_state['recommended_players'])

      if 'graph_code' in st.session_state:
        for graph in st.session_state['graph_code']:
          if isinstance(graph, go.Figure):
            st.plotly_chart(graph, use_container_width=True)
          else:
            st.altair_chart(graph, use_container_width=True)

      if 'ai_recommendation' in st.session_state:
        st.write(st.session_state['ai_recommendation'])

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

              **From the [FBref.com](https://fbref.com/en/comps/Big5/2024-2025/stats/players/2024-2025-Big-5-European-Leagues-Stats) website and [Kaggle](https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025) dataset.**
              """)