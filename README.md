<div style = 'text-align: center;'>
  <h1>The Player Recommender ⚽️ </h1>

  To use the app click here <a href = 'https://playerrecommender-x4jpuv4jxfwpvcvapvk8d7.streamlit.app'>here</a>
  
  The football/soccer transfer market can get dynamic and messy. The Player Recommender is a system which provides recommendations to the user, to point out who the best players are to sign 💰

  <p>This system uses a Gaussian Mixture Model and a K-Means model to put players into clusters based on their statistics in the 2024/25 season. 📊</p>
  <p>Using these clusters, a Gemini API produces personalised recommendations which suggest which players are the best fit. 🤖</p>
  <p>Also, Gemini generates graphs and figures to visualise the different statistics and compare them different statistics between other players. 📈</p>

  This dataset is made by <i><b>Hubert Sidorowicz</b></i> on <b><a href='https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025'>Kaggle</a></b>, and this dataset is originally on the <b><a href = 'https://fbref.com/en/comps/Big5/2024-2025/stats/players/2024-2025-Big-5-European-Leagues-Stats'>FBref</a></b> website.

  <h2>Key Requirements 🔐</h2>
  <ul>
    <li>If you cloned this repo, install the required packages:
    <pre>
    <code>
    pip install seaborn pandas numpy matplotlib scikit-learn streamlit google.generativeai plotly
    </code>
    </pre>
    </li>
    <li>You need a GCP account and access to Google's Gemini API. This is so that you can input an <b>API key</b> 🔑</li>
  </ul>

  <h2>How to Run 🏃‍♂️</h2>
  <p>To run on Streamlit, run this in the terminal
  <pre>
    <code>
    streamlit run PlayerRecommender.py
    </code>
    </pre>
  </p>
</div>
