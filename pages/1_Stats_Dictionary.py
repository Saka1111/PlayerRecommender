import streamlit as st

st.set_page_config(page_title='Stats Dictionary', page_icon='📊', layout='wide')

st.markdown("<h1 style='text-align: center;'>Dictionary of player stats in the 24/25 season</h1>", unsafe_allow_html=True)
st.markdown("""
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
