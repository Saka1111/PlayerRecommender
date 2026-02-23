import pandas as pd

import streamlit as st
import plotly.express as px
import altair as alt
import plotly.graph_objects as go

st.set_page_config(page_title='Player Comparisons', page_icon='⚖️', layout='wide')

st.markdown("<h1 style='text-align: center;'>Player Comparisons</h1>", unsafe_allow_html=True)

st.markdown("<p style='position: absolute; top: 10px; right: 10px; font-size: 0.8em; color: gray;'>Data sourced from the top 5 leagues in the 24/25 football season.</p>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('Files/Cleaned_PlayerData.csv')
    
    return df

df = load_data()

player_names = df['Player']

gk_player_names = df[df['Pos'] == 'GK']['Player']

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

# Neutral or better to have more stats
nuetral_or_better_to_have_more = [
    'Player', 'Pos', 'Squad', 'MP', 'Starts', 'Min', '90s', 'Dist', 
    'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Att_stats_defense', 'Def 3rd_stats_possession', 
    'Mid 3rd_stats_possession', 'Att 3rd_stats_possession', 'Live_stats_possession', 
    'Mn/MP', 'Min%', 'Mn/Start', 'Subs', 'Mn/Sub', 'unSub', 'Launch%',
    'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'xG', 'npxG', 'xAG', 
    'npxG+xAG', 'PrgC', 'PrgP', 'PrgR', 'G+A-PK', 'xG+xAG', 'Sh', 'SoT', 
    'SoT%', 'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT', 'FK', 'npxG/Sh', 'G-xG', 
    'np:G-xG', 'Cmp', 'Att', 'Cmp%', 'TotDist', 'PrgDist', 'xA', 'A-xAG', 
    'KP', '1/3', 'PPA', 'CrsPA', 'Live', 'Dead', 'FK_stats_passing_types', 
    'TB', 'Sw', 'Crs', 'TI', 'CK', 'In', 'Out', 'Str', 'SCA', 'SCA90', 
    'PassLive', 'PassDead', 'TO', 'Sh_stats_gca', 'Fld', 'Def', 'GCA', 
    'GCA90', 'Tkl', 'TklW', 'Blocks_stats_defense', 'Int', 'Tkl+Int', 
    'Clr', 'Touches', 'Att_stats_possession', 'Succ', 'Succ%', 'Carries', 
    'TotDist_stats_possession', 'PrgDist_stats_possession', 
    'PrgC_stats_possession', '1/3_stats_possession', 'CPA', 'Rec', 
    'PrgR_stats_possession', 'Compl', 'PPM', 'onG', '+/-', '+/-90', 'On-Off', 
    'onxG', 'xG+/-', 'xG+/-90', 'Fld_stats_misc', 'PKwon', 'Recov', 'Won', 
    'Won%', 'Saves', 'Save%', 'W', 'D', 'CS', 'CS%', 'PSxG', 'PSxG/SoT', 
    'PSxG+/-', '/90', 'Cmp_stats_keeper_adv', 'Att_stats_keeper_adv', 
    'Cmp%_stats_keeper_adv', 'Thr', 'Stp', 'Stp%', '#OPA', '#OPA/90', 'AvgDist'
]

# better to have less stats
better_to_have_less = [
    'CrdY', 'CrdR', 'Lost', 'Blocks', 'Off', 'Err', 'Def Pen', 'Tkld', 
    'Tkld%', 'Mis', 'Dis', 'onGA', '2CrdY', 'Fls', 'Off_stats_misc', 
    'PKcon', 'OG', 'Lost_stats_misc', 'GA', 'GA90', 'SoTA', 'L', 
    'PKatt_stats_keeper', 'PKA', 'FK_stats_keeper_adv', 'CK_stats_keeper_adv', 
    'OG_stats_keeper_adv', 'Opp', 'AvgLen'
]

def create_colored_bar_chart(data, x_col, y_col, title):
    colors = []
    
    if len(data) == 2:
        val1, val2 = data[y_col].values[0], data[y_col].values[1]

        if y_col in nuetral_or_better_to_have_more:
            if val1 > val2:
                colors = ['green', 'red']
            elif val1 < val2:
                colors = ['red', 'green']
            elif val1 == val2:
                colors = ['yellow', 'yellow']
        elif y_col in better_to_have_less:
            if val1 > val2:
                colors = ['red', 'green']
            elif val1 < val2:
                colors = ['green', 'red']
            elif val1 == val2:
                colors = ['yellow', 'yellow']
        elif val1 == val2:
            colors = ['yellow', 'yellow']
    else:
        colors = ['blue'] * len(data)
    
    fig = go.Figure(data=[
        go.Bar(
            x=data[x_col],
            y=data[y_col],
            marker_color=colors,
            text=data[y_col],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        height=500
    )
    
    return fig

OPC, GKC = st.tabs(["Outfield Player Comparisons", "Goalkeeper Comparisons"])

with OPC:

    col1, col2 = st.columns(2)

    with col1:
        player1 = st.selectbox('Select Player 1', player_names, index=0, key="player1_selectbox")
    with col2:
        player2 = st.selectbox('Select Player 2', player_names, index=1, key="player2_selectbox")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {player1}")
        st.markdown(f"#### {df[df['Player'] == player1]['Squad'].values[0]}")
        st.markdown(f"#### Position: {df[df['Player'] == player1]['Pos'].values[0]}")

    with col2:
        st.markdown(f"### {player2}")
        st.markdown(f"#### {df[df['Player'] == player2]['Squad'].values[0]}")
        st.markdown(f"#### Position: {df[df['Player'] == player2]['Pos'].values[0]}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Base Stats", "Offensive Stats", "Midfield Stats", "Defensive Stats", "Other Stats"])

    with tab1:
        avg_stats = ['MP', 'Min', '90s', 'Gls', 'Ast', 'G+A', 'CrdY', 'CrdR']
        df_avg = df[['Player'] + avg_stats]

        # chart_data = df_avg[df_avg['Player'].isin([player1, player2])]
        chart_data1 = df_avg[df_avg['Player'].isin([player1])]
        chart_data2 = df_avg[df_avg['Player'].isin([player2])]
        chart_data = pd.concat([chart_data1, chart_data2])

        col1, col2, col3 = st.columns(3)

        with col1:
            # Matches Played
            st.plotly_chart(create_colored_bar_chart(chart_data, 'Player', 'MP', 'Matches Played'), use_container_width=True, key="matches_played_chart")

            # Minutes Played
            st.plotly_chart(create_colored_bar_chart(chart_data, 'Player', 'Min', 'Minutes Played'), use_container_width=True, key="minutes_played_chart")

            # Yellow Cards
            st.plotly_chart(create_colored_bar_chart(chart_data, 'Player', 'CrdY', 'Yellow Cards'), use_container_width=True, key="yellow_cards_chart")

        with col2:
            # Goals
            st.plotly_chart(create_colored_bar_chart(chart_data, 'Player', 'Gls', 'Goals'), use_container_width=True, key="goals_chart")

            # Goal + Assists
            st.plotly_chart(create_colored_bar_chart(chart_data, 'Player', 'G+A', 'Goal + Assists'), use_container_width=True, key="goal_assists_chart")

        with col3:
            # Assists
            st.plotly_chart(create_colored_bar_chart(chart_data, 'Player', 'Ast', 'Assists'), use_container_width=True, key="assists_chart")

            # 90s Played
            st.plotly_chart(create_colored_bar_chart(chart_data, 'Player', '90s', '90s Played'), use_container_width=True, key="90s_played_chart")

            # Red Cards
            st.plotly_chart(create_colored_bar_chart(chart_data, 'Player', 'CrdR', 'Red Cards'), use_container_width=True, key="red_cards_chart")

    with tab2:
        df_off = df[['Player'] + attacking_stats]

        chart_data_off1 = df_off[df_off['Player'].isin([player1])]
        chart_data_off2 = df_off[df_off['Player'].isin([player2])]

        chart_data_off = pd.concat([chart_data_off1, chart_data_off2])

        col1, col2, col3 = st.columns(3)

        with col1:
            for stat, defi in attacking_zipped[:15]:  # Displaying the first 10 stats for brevity
                st.plotly_chart(create_colored_bar_chart(chart_data_off, 'Player', stat, defi), use_container_width=True, key=f"{stat}_chart")

        with col2:
            for stat, defi in attacking_zipped[15:30]:  # Next 10 stats
                st.plotly_chart(create_colored_bar_chart(chart_data_off, 'Player', stat, defi), use_container_width=True, key=f"{stat}_chart")

        with col3:
            for stat, defi in attacking_zipped[30:]:  # Remaining stats
                st.plotly_chart(create_colored_bar_chart(chart_data_off, 'Player', stat, defi), use_container_width=True, key=f"{stat}_chart")
            

    with tab3:
        df_mid = df[['Player'] + midfield_stats]

        chart_data_mid1 = df_mid[df_mid['Player'].isin([player1])]
        chart_data_mid2 = df_mid[df_mid['Player'].isin([player2])]

        chart_data_mid = pd.concat([chart_data_mid1, chart_data_mid2])

        col1, col2, col3 = st.columns(3)

        with col1:
            for stat, defi in midfield_zipped[:7]:  # Displaying the first 10 stats for brevity
                st.plotly_chart(create_colored_bar_chart(chart_data_mid, 'Player', stat, defi), use_container_width=True, key=f"{stat}_chart")

        with col2:
            for stat, defi in midfield_zipped[7:14]:  # Next 10 stats
                st.plotly_chart(create_colored_bar_chart(chart_data_mid, 'Player', stat, defi), use_container_width=True, key=f"{stat}_chart")

        with col3:
            for stat, defi in midfield_zipped[14:]:  # Remaining stats
                st.plotly_chart(create_colored_bar_chart(chart_data_mid, 'Player', stat, defi), use_container_width=True, key=f"{stat}_chart")

    with tab4:
        df_def = df[['Player'] + defensive_stats]

        chart_data_def1 = df_def[df_def['Player'].isin([player1])]
        chart_data_def2 = df_def[df_def['Player'].isin([player2])]

        chart_data_def = pd.concat([chart_data_def1, chart_data_def2])

        col1, col2, col3 = st.columns(3)

        with col1:
            for stat, defi in defensive_zipped[:8]:  # Displaying the first 10 stats for brevity
                st.plotly_chart(create_colored_bar_chart(chart_data_def, 'Player', stat, defi), use_container_width=True, key=f"{stat}_chart")

        with col2:
            for stat, defi in defensive_zipped[8:17]:  # Next 10 stats
                st.plotly_chart(create_colored_bar_chart(chart_data_def, 'Player', stat, defi), use_container_width=True, key=f"{stat}_chart")

        with col3:
            for stat, defi in defensive_zipped[17:]:  # Remaining stats
                st.plotly_chart(create_colored_bar_chart(chart_data_def, 'Player', stat, defi), use_container_width=True, key=f"{stat}_chart")
    with tab5:
        df_other = df[['Player'] + other_stats]

        chart_data_other1 = df_other[df_other['Player'].isin([player1])]
        chart_data_other2 = df_other[df_other['Player'].isin([player2])]

        chart_data_other = pd.concat([chart_data_other1, chart_data_other2])

        col1, col2, col3 = st.columns(3)

        with col1:
            for stat, defi in other_zipped[:14]:  # Displaying the first 10 stats for brevity
                st.plotly_chart(create_colored_bar_chart(chart_data_other, 'Player', stat, defi), use_container_width=True, key=f"{stat}_chart")

        with col2:
            for stat, defi in other_zipped[14:28]:  # Next 10 stats
                st.plotly_chart(create_colored_bar_chart(chart_data_other, 'Player', stat, defi), use_container_width=True, key=f"{stat}_chart")

        with col3:
            for stat, defi in other_zipped[28:]:  # Remaining stats
                st.plotly_chart(create_colored_bar_chart(chart_data_other, 'Player', stat, defi), use_container_width=True, key=f"{stat}_chart")

with GKC:
    col1, col2 = st.columns(2)

    with col1:
        gk_player1 = st.selectbox('Select Goalkeeper 1', gk_player_names, index=0, key="gk_player1_selectbox")
    with col2:
        gk_player2 = st.selectbox('Select Goalkeeper 2', gk_player_names, index=1, key="gk_player2_selectbox")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {gk_player1}")
        st.markdown(f"#### {df[df['Player'] == gk_player1]['Squad'].values[0]}")
        st.markdown(f"#### Position: {df[df['Player'] == gk_player1]['Pos'].values[0]}")

    with col2:
        st.markdown(f"### {gk_player2}")
        st.markdown(f"#### {df[df['Player'] == gk_player2]['Squad'].values[0]}")
        st.markdown(f"#### Position: {df[df['Player'] == gk_player2]['Pos'].values[0]}")

    
    df_gk = df[['Player'] + goalkeeping_stats]

    chart_data_gk1 = df_gk[df_gk['Player'].isin([gk_player1])]
    chart_data_gk2 = df_gk[df_gk['Player'].isin([gk_player2])]
    chart_data_gk = pd.concat([chart_data_gk1, chart_data_gk2])

    col1, col2, col3 = st.columns(3)

    with col1:
        for stat, defi in goalkeeping_zipped[:11]:  # Displaying the first 10 stats for brevity
            st.plotly_chart(create_colored_bar_chart(chart_data_gk, 'Player', stat, defi), use_container_width=True, key=f"{stat}_chart")

    with col2:
        for stat, defi in goalkeeping_zipped[11:22]:  # Next 10 stats
            st.plotly_chart(create_colored_bar_chart(chart_data_gk, 'Player', stat, defi), use_container_width=True, key=f"{stat}_chart")

    with col3:
        for stat, defi in goalkeeping_zipped[22:]:  # Remaining stats
            st.plotly_chart(create_colored_bar_chart(chart_data_gk, 'Player', stat, defi), use_container_width=True, key=f"{stat}_chart")