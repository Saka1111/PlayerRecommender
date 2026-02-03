import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from io import StringIO

import streamlit as st
import plotly.express as px
import altair as alt
import plotly.graph_objects as go

import google.generativeai as genai
import json

st.set_page_config(page_title='Player Recommender', page_icon='⚽️', layout='wide')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

@st.cache_data
def load_all_players():
  return pd.read_csv('Files/k_means_players.csv')

@st.cache_data
def load_gmm_players():
  return pd.read_csv('Files/gmm_players.csv')

@st.cache_data
def load_stat_mapping(filepath):
  with open(filepath, 'r') as f:
    string_keys_mapping = json.load(f)
    
  int_keys_mapping = {int(k): v for k, v in string_keys_mapping.items()}
  return int_keys_mapping

all_players = load_all_players()
New = load_gmm_players()

players = list(all_players.Player.sort_values())

stat_mapping = load_stat_mapping('Files/s_stats.json')

big_stat_mapping = load_stat_mapping('Files/b_stats.json')

base_cols = ['Player', 'Pos', 'Squad', 'Age']

col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
  st.markdown("<h1 style='text-align: center;'>The Player Recommender ⚽️</h1>", unsafe_allow_html=True)
    
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
    

    col_left_btn, col_center_btn, col_right_btn = st.columns([1, 2, 1])
    
    with col_center_btn:
      col1, col2 = st.columns(2)

      with col1:
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

      with col2:
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
              selected_players_df = pd.read_json(StringIO(data_json))

            else:
              player_df = all_players.loc[all_players.Player == player]
              combined_df = pd.concat([player_df, df1])
              data_json = combined_df.to_json(orient='records')
              selected_players_df = pd.read_json(StringIO(data_json))

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

    col_left_btn, col_center_btn, col_right_btn = st.columns([1, 2, 1])

    with col_center_btn:
      col1, col2, col3 = st.columns(3)

      with col2:
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

    col_left_btns, col_center_btns, col_right_btns = st.columns([1, 2, 1])
    with col_center_btns:
      col1, col2 = st.columns(2)
      
      with col1:
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

      with col2:
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
              selected_players_df = pd.read_json(StringIO(data_json))

            else:
              player_df = all_players.loc[all_players.Player == player]
              combined_df = pd.concat([player_df, df1])
              data_json = combined_df.to_json(orient='records')
              selected_players_df = pd.read_json(StringIO(data_json))

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

    col_left_btn, col_center_btn, col_right_btn = st.columns([1, 2, 1])

    col1, col2, col3 = st.columns(3)

    with col_center_btn:

      with col2:
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

else:
  st.write('Not a valid search')
