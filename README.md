<div style = 'text-align: center;'>
  <h1>PlayerRecommender</h1>
  
  The football/soccer transfer market can get dynamic and messy. The Player Recommender is a system which provides recommendations to the user, to point out who the best players are to sign.

  This system uses a Gaussian Mixture Model and a K-Means model to put players into clusters based on their statistics in the 2024/25 season.
  Using these clusters, a Gemini API produces personalised recommendations which suggest which players are the best fit.
  Also, Gemini generates graphs and figures to visualise the different statistics and compare them different statistics between other players.

  This dataset is made by Hubert Sidorowicz on <a href='https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025'>Kaggle</a>, and this dataset is originally on the <a href = 'https://fbref.com/en/comps/Big5/2024-2025/stats/players/2024-2025-Big-5-European-Leagues-Stats'>FBref</a> website.

  <h2>Key Requirements</h2>
  <ul>
    <li>If you cloned this repo, install the required packages:
    <pre>
    <code>
    pip install streamlit plotly
    </code>
    </pre>
    </li>
    <li>You need a GCP account and access to Google's Gemini API. This is so that you can input an API key</li>
  </ul>
</div>
