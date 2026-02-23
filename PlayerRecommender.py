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

@st.cache_data
def load_all_players():
  return pd.read_csv('Files/Cleaned_PlayerData.csv')

@st.cache_data
def load_km():
  return pd.read_csv('Files/kmeans_players.csv')

@st.cache_data
def load_gmm():
  return pd.read_csv('Files/gmm_players.csv')

@st.cache_data
def load_stat_mapping(filepath):
  with open(filepath, 'r') as f:
    string_keys_mapping = json.load(f)
    
  int_keys_mapping = {int(k): v for k, v in string_keys_mapping.items()}
  return int_keys_mapping

all_players = load_all_players()
km = load_km()
gmm = load_gmm()

players = list(all_players.Player.sort_values())

attacking_stats = [
    'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'xG', 'npxG', 'xAG', 
    'npxG+xAG', 'G+A-PK', 'xG+xAG', 'Sh', 'SoT', 'SoT%', 'Sh/90', 'SoT/90', 
    'G/Sh', 'G/SoT', 'Dist', 'FK', 'npxG/Sh', 'G-xG', 'np:G-xG', 'xA', 
    'A-xAG', 'KP', '1/3', 'PPA', 'CrsPA', 'TB', 'Sw', 'Crs', 'CK', 'In', 
    'Out', 'Str', 'Off', 'Blocks', 'Att_stats_possession', 'Succ', 'Succ%', 
    'CPA', 'PKwon', 'Off_stats_misc'
]

midfield_stats = [
    'PrgC', 'PrgP', 'PrgR', 'Cmp', 'Att', 'Cmp%', 'TotDist', 'PrgDist', 
    'Live', 'Dead', 'FK_stats_passing_types', 'TI', 'Touches', 
    'Mid 3rd_stats_possession', 'Live_stats_possession', 'Carries', 
    'TotDist_stats_possession', 'PrgDist_stats_possession', 
    'PrgC_stats_possession', '1/3_stats_possession', 'Rec', 
    'PrgR_stats_possession'
]

defensive_stats = [
    'Tkl', 'TklW', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Att_stats_defense', 
    'Tkl%', 'Lost', 'Blocks_stats_defense', 'Sh_stats_defense', 'Pass', 
    'Int', 'Tkl+Int', 'Clr', 'Err', 'Def Pen', 'Def 3rd_stats_possession', 
    'Fls', 'Int_stats_misc', 'TklW_stats_misc', 'PKcon', 'OG', 'Recov', 
    'Won', 'Lost_stats_misc', 'Won%'
]

goalkeeping_stats = [
    'GA', 'GA90', 'SoTA', 'Saves', 'Save%', 'W', 'D', 'L', 'CS', 'CS%', 
    'PKatt_stats_keeper', 'PKA', 'PKsv', 'PKm', 'FK_stats_keeper_adv', 
    'CK_stats_keeper_adv', 'OG_stats_keeper_adv', 'PSxG', 'PSxG/SoT', 
    'PSxG+/-', '/90', 'Cmp_stats_keeper_adv', 'Att_stats_keeper_adv', 
    'Cmp%_stats_keeper_adv', 'Att (GK)', 'Thr', 'Launch%', 'AvgLen', 'Opp', 
    'Stp', 'Stp%', '#OPA', '#OPA/90', 'AvgDist'
]

other_stats = [
    'MP', 'Starts', 'Min', '90s', 'CrdY', 'CrdR', 
    'SCA', 'SCA90', 'PassLive', 'PassDead', 'TO', 'Sh_stats_gca', 'Fld', 
    'Def', 'GCA', 'GCA90', 'Att 3rd_stats_possession', 'Att Pen', 'Tkld', 
    'Tkld%', 'Mis', 'Dis', 'Mn/MP', 'Min%', 'Mn/Start', 'Compl', 'Subs', 
    'Mn/Sub', 'unSub', 'PPM', 'onG', 'onGA', '+/-', '+/-90', 'On-Off', 
    'onxG', 'onxGA', 'xG+/-', 'xG+/-90', '2CrdY', 'Fld_stats_misc', 
    'Crs_stats_misc'
]

attacking_stats_def = [
    'Goals scored',
    'Assists made',
    'Total goals and assists',
    'Non-penalty goals',
    'Penalty kicks made',
    'Penalty kicks attempted',
    'Expected Goals',
    'Non-penalty expected goals',
    'Expected goals assisted',
    'Non-penalty expected goals plus expected assisted goals',
    'Non-penalty goals and assists per 90',
    'Expected goals plus expected goals assisted per 90',
    'Total shots (not incuding penalty kicks)',
    'Shots on target (not including penalty kicks)',
    'Percentage of shots on target',
    'Shots total per 90',
    'Shots on target per 90',
    'Goals per shot',
    'Goals per shot on target',
    'Average shot distance',
    'Shots from free kicks',
    'Non-penalty expected goals per shot',
    'Goals minus expected goals',
    'Non-penalty goals minus non-penalty expected goals',
    'Expected assists',
    'Assists minus expected goals assisted',
    'Key Passes (passes leading to a shot)',
    'Passes into the final third',
    'Passes into the penalty area',
    'Crosses into the penalty area',
    'Through balls',
    'Switches',
    'Crosses',
    'Corner kicks',
    'Inswinging corner kicks',
    'Outswinging corner kicks',
    'Straight corner kicks',
    'Passes offsides',
    'Passes blocked',
    'Take-ons attempted',
    'Successful take-ons',
    'Percentage of successful take-ons',
    'Carries into the penalty area',
    'Penalty kicks won',
    'Offsides'
]

midfield_stats_def = [
    'Progressive carries',
    'Progressive passes',
    'Progressive passes received',
    'Passes completed',
    'Passess attempted',
    'Pass completion percentage',
    'Total passing distance',
    'Progressive passing distance',
    'Live-ball passes',
    'Dead-ball passes',
    'Passes from free-kicks',
    'Throw-ins taken',
    'Number of touches',
    'Touches in the midfield third',
    'Live-ball touches',
    'Number of carries',
    'Total carrying distance',
    'Progressive carrying distance',
    'Progressive carries',
    'Carries into the final third',
    'Passes received',
    'Progressive passes received'
]

defensive_stats_def = [
    'Tackles',
    'Tackles won',
    'Tackles in the defensive third',
    'Tackles in the middle third',
    'Tackles in the attacking third',
    'Dribbles challenged',
    'Percentage of dribblers challenged',
    'Unsuccessful challenged',
    'Blocks',
    'Shots blocked',
    'Passes blocked',
    'Interceptions',
    'number of tackles plus interceptions',
    'Clearances',
    'Errors leading to an opponents shot',
    'Touches in the defensive penalty area',
    'Touches in the defensive third',
    'Fouls committed',
    'Interceptions',
    'Tackles won',
    'Penalty kicks conceded',
    'Own goals',
    'Ball recoveries',
    'Aerial duels won',
    'Aerial duels lost',
    'Percentage of Aerial duels won'
]

goalkeeping_stats_def = [
    'Goals Against',
    'Goals against per 90',
    'Shots on target against',
    'Saves made',
    'Save percentage',
    'Wins',
    'Draws',
    'Losses',
    'Clean sheets',
    'Clean sheet percentage',
    'Penalty kicks attempted',
    'Penalty kicks allowed',
    'Penalty kick saves',
    'Penalty kicks missed',
    'Free-kick goals against',
    'Corner kick goals against',
    'Own goals scored against',
    'Post-shot expected goals',
    'Post-shot expected goals per shot on target',
    'Post-shot expected goals minus goals allowed',
    'Post-shot expected goals minus goals allowed per 90',
    'Passes completed longer than 40 yards',
    'Passes attempted longer than 40 yards',
    'Percentage pass completion (longer than 40 yards)',
    'Passes attempted (not including goal kicks)',
    'Throws attempted',
    'Percentage of passes more than 40 yards (not including goal kicks)',
    'Average length of pass, in yards (not including goal kicks)',
    'Crosses faced',
    'Crosses stopped',
    'Percentage of crosses stopped',
    'Defensive actions outside the penalty area',
    'Defensive actions outside the penalty area per 90',
    'Average distance of defensive actions'
]

other_stats_def = [
    'Matches played',
    'How many times the player started',
    'How many minutes the player played',
    'How many 90 minutes played - Minutes played divded by 90 (1dp)',
    'Yellow cards',
    'Red cards',
    'Shot-creating actions',
    'Shot-creating actions per 90',
    'Completed live-ball passes that lead to a shot attempt',
    'Completed dead-ball passes that lead to a shot attempt',
    'Successful take-ons that lead to a shot attempt',
    'Shots that lead to another shot attempt',
    'Fouls drawn that lead to a shot attempt',
    'Defensive action that leads to a shot attempt',
    'Goal-creating actions',
    'Goal-creating actions per 90',
    'Touches in the attacking third',
    'Touches in the attacking penalty area',
    'Times tackled during a take-on',
    'Tackled during a take-on percentage',
    'Miscontrols',
    'Dispossessed',
    'Minutes per matches played',
    'Percentage of minutes played',
    'Minutes per match started',
    'Complete matches played',
    'Substitute appearances',
    'Minutes per substitution',
    'Matches as unused sub',
    'Points per match',
    'Goals scored by team while on the pitch',
    'Goals allowed by team while on the pitch',
    'Goals scored minus goals allowed while on the pitch',
    'Goals scored minus goals allowed while on the pitch per 90',
    'Net goals per 90 by the team while on the pitch',
    'Expected goals by the team while on the pitch',
    'Expected goals allowed by the team while on the pitch',
    'Expected goals minus expected goals allowed by team while on the pitch',
    'Expected goals minus expected goals allowed by team while on the pitch per 90',
    'Second yellow cards',
    'Fouls drawn',
    'Crosses'
]

# Zip attacking stats with their definitions
attacking_zipped = list(zip(attacking_stats, attacking_stats_def))

# Zip midfield stats with their definitions
midfield_zipped = list(zip(midfield_stats, midfield_stats_def))

# Zip defensive stats with their definitions
defensive_zipped = list(zip(defensive_stats, defensive_stats_def))

# Zip goalkeeping stats with their definitions
goalkeeping_zipped = list(zip(goalkeeping_stats, goalkeeping_stats_def))

# Zip other stats with their definitions
other_zipped = list(zip(other_stats, other_stats_def))

base_cols = ['Player', 'Pos', 'Squad', 'Age']

norm_cols = ['Min', '90s', 'Gls', 'Ast', 'xG', 'xA', 'G+A', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR']

norm_cols_def = [
    'How many minutes the player played',
    'How many 90 minutes played - Minutes played divded by 90 (1dp)',
    'Goals scored',
    'Assists made',
    'Expected Goals',
    'Expected assists',
    'Total goals and assists',
    'Non-penalty goals',
    'Penalty kicks made',
    'Penalty kicks attempted',
    'Yellow cards',
    'Red cards'
]

norm_cols_zipped = list(zip(norm_cols, norm_cols_def))

col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
  st.markdown("<h1 style='text-align: center;'>The Player Recommender ⚽️</h1>", unsafe_allow_html=True)

st.markdown("<p style='position: absolute; top: 10px; right: 10px; font-size: 0.8em; color: gray;'>Data sourced from the top 5 leagues in the 24/25 football season.</p>", unsafe_allow_html=True)
    
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

def create_charts(player = player, all_players = all_players, age = age, search = search, cols = None):
    graph_list = []

    if search == 'specific':

      if player in all_players.Player.unique():
          km_cl = km.loc[km.Player == player][['KMeans_Cluster']]
          km_cl = km_cl.iloc[0, 0]

          gmm_cl = km.loc[km.Player == player][['Cluster']]
          gmm_cl = gmm_cl.iloc[0, 0]

          df1 = km.loc[(km.KMeans_Cluster == km_cl) & (km.Cluster == gmm_cl)].copy()

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

    elif search == 'broad':

      if player in all_players.Player.unique():
          x = gmm.loc[gmm.Player == player][['Cluster']]
          c = x.iloc[0, 0]
          df1 = gmm.loc[gmm.Cluster == c].sort_values(by='Player').copy()

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

    # Creating charts
    if 'GK' in selected_players_df.loc[selected_players_df.Player == player]['Pos'].values[0]:
        bar_chart_metrics = cols
    else:
        bar_chart_metrics = cols

    # scaler = MinMaxScaler()
    # scaled_data = scaler.fit_transform(selected_players_df[bar_chart_metrics])
    # scaled_df = pd.DataFrame(scaled_data, columns=bar_chart_metrics)
    # scaled_df['Player'] = selected_players_df['Player'].reset_index(drop=True)

    # radar = go.Figure()
    # labels = [stat.title() for stat in bar_chart_metrics]

    # for i, r in scaled_df.iterrows():
    #     player_name = r['Player']
    #     values = r[bar_chart_metrics].tolist()
    #     values.append(values[0])

    #     radar.add_trace(go.Scatterpolar(
    #         r=values,
    #         theta=labels + [labels[0]],
    #         fill='toself',
    #         name=player_name
    #     ))

    # radar.update_layout(
    #     polar=dict(
    #         radialaxis=dict(
    #             visible=True,
    #             range=[0, 1]
    #         )
    #     ),
    #     showlegend=True,
    #     title=f'Radar Chart of Key Stats for {player} and Similar Players'
    # )

    # graph_list.append(radar)

    # Creating bar charts
    for metric, defi in bar_chart_metrics:
        
        sanitized_metric = metric.replace(':', '\\:')  # Escape colon in column names

        bar_chart = alt.Chart(selected_players_df).mark_bar().encode(
            y=alt.Y(sanitized_metric, axis=alt.Axis(title=metric)),

            x=alt.X('Player', sort='x', axis=alt.Axis(title='Player')),

            color=alt.Color('Player', legend=None),

            tooltip=['Player', alt.Tooltip(sanitized_metric, format=".2f")]
        ).properties(
            title=f'{defi}'
        )

        graph_list.append(bar_chart)

    return graph_list


if search == 'specific':
  if player in km.Player.unique():
    km_cl = km.loc[km.Player == player][['KMeans_Cluster']]
    km_cl = km_cl.iloc[0,0]

    gmm_cl = km.loc[km.Player == player][['Cluster']]
    gmm_cl = gmm_cl.iloc[0,0]

    df1 = km.loc[(km.KMeans_Cluster == km_cl) & (km.Cluster == gmm_cl)].copy()
    
    try:
      df1 = df1.loc[df1.Age <= age]

    except:
      st.write('')

    if player in df1.Player.unique():
      data_json = df1.to_json(orient='records')

    else:
      player_df = km.loc[km.Player == player]
      combined_df = pd.concat([player_df, df1])
      data_json = combined_df.to_json(orient='records')
      selected_players_df = pd.read_json(StringIO(data_json))

    col_left_btn, col_center_btn, col_right_btn = st.columns([1, 2, 1])
    
    with col_center_btn:
      col1, col2 = st.columns(2)

      with col1:
        # Recommended Players
        if st.button('Show Recommended Players'):
          with st.spinner('Generating recommendations...'):
            if player in df1.Player.unique():
              if 'GK' in df1.loc[df1.Player == player]['Pos'].values[0]:

                st.session_state['recommended_players'] = df1[base_cols + goalkeeping_stats]
              
              else:
                # stats = stat_mapping.get(c, [])
                # cluster_cols = base_cols + stats
                st.session_state['recommended_players'] = df1[base_cols + attacking_stats + midfield_stats + defensive_stats + goalkeeping_stats + other_stats]

            else:
              if 'GK' in combined_df.loc[combined_df.Player == player]['Pos'].values[0]:

                st.session_state['recommended_players'] = combined_df[base_cols + goalkeeping_stats]
              
              else:
                # stats = stat_mapping.get(c, [])
                # cluster_cols = base_cols + stats
                st.session_state['recommended_players'] = combined_df[base_cols + attacking_stats + midfield_stats + defensive_stats + other_stats]

      with col2:
        # Graphs
        if st.button('Generate graphs'):
          with st.spinner('Generating graphs...'):

            st.session_state['graph_code'] = create_charts(cols=norm_cols_zipped)

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
        filtered_players = st.session_state['recommended_players'].loc[st.session_state['recommended_players'].Player == player]
        if not filtered_players.empty and 'GK' in filtered_players['Pos'].values[0]:
          try:
            base, gk = st.tabs(['Base stats', 'Goalkeeping stats'])
            with base:
              st.write(st.session_state['recommended_players'][base_cols])
            with gk:
              st.write(st.session_state['recommended_players'][base_cols + goalkeeping_stats])
          except:
            st.write('')
        else:
          try:
            base, att, mid, defi, other = st.tabs(['Base stats', 'Offensive stats', 'Midfield stats', 'Defensive stats', 'Other stats'])
            with base:
              st.write(st.session_state['recommended_players'][base_cols])
              # st.write(st.session_state['recommended_players'])
            with att:
              st.write(st.session_state['recommended_players'][base_cols + attacking_stats])
            with mid:
              st.write(st.session_state['recommended_players'][base_cols + midfield_stats])
            with defi:
              st.write(st.session_state['recommended_players'][base_cols + defensive_stats])
            with other:
              st.write(st.session_state['recommended_players'][base_cols + other_stats])
          except:
            st.write('')

    if 'graph_code' in st.session_state:

      if player in all_players.Player.unique():
        if 'GK' in all_players.loc[all_players.Player == player]['Pos'].values[0]:
          st.session_state['graph_code'] = create_charts(cols=goalkeeping_zipped)

          st.markdown("<h2 style='text-align: center;'>Goalkeeping Stats</h2>", unsafe_allow_html=True)

          for graph in st.session_state['graph_code']:
            if isinstance(graph, go.Figure):
              st.plotly_chart(graph, use_container_width=True)
            else:
              st.altair_chart(graph, use_container_width=True)

        else:
          att, mid, defi, other = st.tabs(['Offensive stats', 'Midfield stats', 'Defensive stats', 'Other stats'])
          with att:
            st.session_state['graph_code'] = create_charts(cols=attacking_zipped)

            st.markdown("<h2 style='text-align: center;'>Offensive Stats</h2>", unsafe_allow_html=True)

            for graph in st.session_state['graph_code']:
              if isinstance(graph, go.Figure):
                st.plotly_chart(graph, use_container_width=True)
              else:
                st.altair_chart(graph, use_container_width=True)

          with mid:
            st.session_state['graph_code'] = create_charts(cols=midfield_zipped)

            st.markdown("<h2 style='text-align: center;'>Midfield Stats</h2>", unsafe_allow_html=True)

            for graph in st.session_state['graph_code']:
              if isinstance(graph, go.Figure):
                st.plotly_chart(graph, use_container_width=True)
              else:
                st.altair_chart(graph, use_container_width=True)

          with defi:
            st.session_state['graph_code'] = create_charts(cols=defensive_zipped)

            st.markdown("<h2 style='text-align: center;'>Defensive Stats</h2>", unsafe_allow_html=True)

            for graph in st.session_state['graph_code']:
              if isinstance(graph, go.Figure):
                st.plotly_chart(graph, use_container_width=True)
              else:
                st.altair_chart(graph, use_container_width=True)

          with other:
            st.session_state['graph_code'] = create_charts(cols=other_zipped)

            st.markdown("<h2 style='text-align: center;'>Other Stats</h2>", unsafe_allow_html=True)

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
  if player in gmm.Player.unique():
    x = gmm.loc[gmm.Player == player][['Cluster']]
    c = x.iloc[0,0]
    df1 = gmm.loc[gmm.Cluster == c].sort_values(by = 'Player').copy()
    
    try:
      df1 = df1.loc[df1.Age <= age]

    except:
      st.write('')
    
    if player in df1.Player.unique():
      data_json = df1.to_json(orient='records')

    else:
      player_df = gmm.loc[gmm.Player == player]
      combined_df = pd.concat([player_df, df1])
      data_json = combined_df.to_json(orient='records')
      selected_players_df = pd.read_json(StringIO(data_json))

    col_left_btns, col_center_btns, col_right_btns = st.columns([1, 2, 1])
    with col_center_btns:
      col1, col2 = st.columns(2)
      
      with col1:
        # Recommended Players
        if st.button('Show Recommended Players'):
          with st.spinner('Generating recommendations...'):

            if player in df1.Player.unique():
              if 'GK' in df1.loc[df1.Player == player]['Pos'].values[0]:
                # st.session_state['recommended_players'] = df1[base_cols + ['GA', 'GA90', 'SoTA', 'Saves', 'Save%', 'W', 'D', 'L', 'CS', 'CS%', 'PSxG', 'PSxG/SoT', 'PSxG+/-', '/90', 'Cmp_stats_keeper_adv', 'Att_stats_keeper_adv', 'Cmp%_stats_keeper_adv', 'Att (GK)', 'Thr', 'Launch%', 'AvgLen', 'Opp', 'Stp', 'Stp%', '#OPA', '#OPA/90', 'AvgDist']]
                st.session_state['recommended_players'] = df1[base_cols + goalkeeping_stats]
              
              else:
                # stats = stat_mapping.get(c, [])
                # cluster_cols = base_cols + stats
                st.session_state['recommended_players'] = df1[base_cols + attacking_stats + midfield_stats + defensive_stats + other_stats]

            else:
              if 'GK' in combined_df.loc[combined_df.Player == player]['Pos'].values[0]:
                # st.session_state['recommended_players'] = combined_df[base_cols + ['GA', 'GA90', 'SoTA', 'Saves', 'Save%', 'W', 'D', 'L', 'CS', 'CS%', 'PSxG', 'PSxG/SoT', 'PSxG+/-', '/90', 'Cmp_stats_keeper_adv', 'Att_stats_keeper_adv', 'Cmp%_stats_keeper_adv', 'Att (GK)', 'Thr', 'Launch%', 'AvgLen', 'Opp', 'Stp', 'Stp%', '#OPA', '#OPA/90', 'AvgDist']]
                st.session_state['recommended_players'] = combined_df[base_cols + goalkeeping_stats]
              
              else:
                # stats = stat_mapping.get(c, [])
                # cluster_cols = base_cols + stats
                st.session_state['recommended_players'] = combined_df[base_cols + attacking_stats + midfield_stats + defensive_stats + other_stats]

      with col2:
        # Graphs
        if st.button('Generate graphs'):
          with st.spinner('Generating graphs...'):

            st.session_state['graph_code'] = create_charts(cols=norm_cols_zipped)

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
      # st.write(st.session_state['recommended_players'])
      filtered_players = st.session_state['recommended_players'].loc[st.session_state['recommended_players'].Player == player]
      if not filtered_players.empty and 'GK' in filtered_players['Pos'].values[0]:
        try:
          base, gk = st.tabs(['Base stats', 'Goalkeeping stats'])
          with base:
            st.write(st.session_state['recommended_players'][base_cols])
          with gk:
            st.write(st.session_state['recommended_players'][base_cols + goalkeeping_stats])
        except:
          st.write('')

      else:
        try:
          base, att, mid, defi, other = st.tabs(['Base stats', 'Offensive stats', 'Midfield stats', 'Defensive stats', 'Other stats'])
          with base:
            st.write(st.session_state['recommended_players'][base_cols])
            # st.write(st.session_state['recommended_players'])
          with att:
            st.write(st.session_state['recommended_players'][base_cols + attacking_stats])
          with mid:
            st.write(st.session_state['recommended_players'][base_cols + midfield_stats])
          with defi:
            st.write(st.session_state['recommended_players'][base_cols + defensive_stats])
          with other:
            st.write(st.session_state['recommended_players'][base_cols + other_stats])
        except:
          st.write('')

    if 'graph_code' in st.session_state:

      if player in all_players.Player.unique():
        if 'GK' in all_players.loc[all_players.Player == player]['Pos'].values[0]:
          st.session_state['graph_code'] = create_charts(cols=goalkeeping_zipped)

          st.markdown("<h2 style='text-align: center;'>Goalkeeping Stats</h2>", unsafe_allow_html=True)

          for graph in st.session_state['graph_code']:
            if isinstance(graph, go.Figure):
              st.plotly_chart(graph, use_container_width=True)
            else:
              st.altair_chart(graph, use_container_width=True)

        else:
          att, mid, defi, other = st.tabs(['Offensive stats', 'Midfield stats', 'Defensive stats', 'Other stats'])
          with att:
            st.session_state['graph_code'] = create_charts(cols=attacking_zipped)

            st.markdown("<h2 style='text-align: center;'>Offensive Stats</h2>", unsafe_allow_html=True)

            for graph in st.session_state['graph_code']:
              if isinstance(graph, go.Figure):
                st.plotly_chart(graph, use_container_width=True)
              else:
                st.altair_chart(graph, use_container_width=True)

          with mid:
            st.session_state['graph_code'] = create_charts(cols=midfield_zipped)

            st.markdown("<h2 style='text-align: center;'>Midfield Stats</h2>", unsafe_allow_html=True)

            for graph in st.session_state['graph_code']:
              if isinstance(graph, go.Figure):
                st.plotly_chart(graph, use_container_width=True)
              else:
                st.altair_chart(graph, use_container_width=True)

          with defi:
            st.session_state['graph_code'] = create_charts(cols=defensive_zipped)

            st.markdown("<h2 style='text-align: center;'>Defensive Stats</h2>", unsafe_allow_html=True)

            for graph in st.session_state['graph_code']:
              if isinstance(graph, go.Figure):
                st.plotly_chart(graph, use_container_width=True)
              else:
                st.altair_chart(graph, use_container_width=True)

          with other:
            st.session_state['graph_code'] = create_charts(cols=other_zipped)

            st.markdown("<h2 style='text-align: center;'>Other Stats</h2>", unsafe_allow_html=True)

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