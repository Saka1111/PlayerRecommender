# Import libraries
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Load the data
PlayerData = pd.read_csv('Files/Cleaned_PlayerData.csv')

# GMM Clustering

# Prepare data for GMM clustering
New = PlayerData.copy()

# Store categorical columns
player = New.Player
nation = New.Nation
squad = New.Squad
comp = New.Comp
age = New.Age

# Drop categorical columns
New = New.drop(columns=['Player', 'Nation', 'Squad', 'Comp', 'Age'])

# Encode position
le = LabelEncoder()
New['Pos'] = le.fit_transform(New['Pos'])

# Scale features
scaler = MinMaxScaler()
N_scaled = scaler.fit_transform(New)

covariance_types = ["diag", "tied", "full", "spherical"]

results1 = []

for cov_type in covariance_types:
    gmm = GaussianMixture(
        n_components=30,
        covariance_type=cov_type,
        reg_covar=1e-6,
        n_init=10,
        max_iter=500,
        random_state=42
    )

    gmm.fit(N_scaled)

    results1.append({
        "n_components": 30,
        "covariance_type": cov_type,
        "bic": gmm.bic(N_scaled),
        "aic": gmm.aic(N_scaled),
        "log_likelihood": gmm.score(N_scaled),
        "model": gmm
    })

best_bic_model1 = min(results1, key=lambda x: x["bic"])

New['Cluster'] = best_bic_model1['model'].predict(N_scaled)

New['Pos'] = le.inverse_transform(New['Pos'])
New['Player'] = player
New['Nation'] = nation
New['Squad'] = squad
New['Comp'] = comp
New['Age'] = age

# Save GMM results
New.to_csv('Files/gmm_players.csv', index=False)

# KMeans Clustering

cluster0 = New[New['Cluster'] == 0]
cluster1 = New[New['Cluster'] == 1]
cluster2 = New[New['Cluster'] == 2]
cluster3 = New[New['Cluster'] == 3]
cluster4 = New[New['Cluster'] == 4]
cluster5 = New[New['Cluster'] == 5]
cluster6 = New[New['Cluster'] == 6]
cluster7 = New[New['Cluster'] == 7]
cluster8 = New[New['Cluster'] == 8]
cluster9 = New[New['Cluster'] == 9]
cluster10 = New[New['Cluster'] == 10]
cluster11 = New[New['Cluster'] == 11]
cluster12 = New[New['Cluster'] == 12]
cluster13 = New[New['Cluster'] == 13]
cluster14 = New[New['Cluster'] == 14]
cluster15 = New[New['Cluster'] == 15]
cluster16 = New[New['Cluster'] == 16]
cluster17 = New[New['Cluster'] == 17]
cluster18 = New[New['Cluster'] == 18]
cluster19 = New[New['Cluster'] == 19]
cluster20 = New[New['Cluster'] == 20]
cluster21 = New[New['Cluster'] == 21]
cluster22 = New[New['Cluster'] == 22]
cluster23 = New[New['Cluster'] == 23]
cluster24 = New[New['Cluster'] == 24]
cluster25 = New[New['Cluster'] == 25]
cluster26 = New[New['Cluster'] == 26]
cluster27 = New[New['Cluster'] == 27]
cluster28 = New[New['Cluster'] == 28]
cluster29 = New[New['Cluster'] == 29]

# Function which does the whole k-means process

def km_cl(data):
    # Store categorical columns
    player = data.Player
    nation = data.Nation
    squad = data.Squad
    comp = data.Comp
    age = data.Age
    gmm_cluster = data.Cluster

    # Drop categorical columns
    data = data.drop(columns=['Player', 'Nation', 'Squad', 'Comp', 'Age', 'Cluster'])

    # Encode position
    le = LabelEncoder()
    data['Pos'] = le.fit_transform(data['Pos'])

    # Scale features
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    ks = list(range(2,8))

    results = []

    for k in ks:
        inertias = []
        silhouettes = []
        labels_list = []
        
        for seed in range(30):  # lots of restarts
            kmeans = KMeans(
                n_clusters=k,
                n_init=1,
                max_iter=500,
                random_state=42
            )
            
            labels = kmeans.fit_predict(data_scaled)
            inertias.append(kmeans.inertia_)
            
            if k < len(data_scaled):
                silhouettes.append(silhouette_score(data_scaled, labels))
            
            labels_list.append(labels)
        
        results.append({
            "k": k,
            "mean_inertia": np.mean(inertias),
            "std_inertia": np.std(inertias),
            "mean_silhouette": np.mean(silhouettes),
            "std_silhouette": np.std(silhouettes),
            "model": kmeans
        })

    return data_scaled, results

# Cluster 0
cluster0_scaled, cluster0_results = km_cl(cluster0)
best_c0_model = cluster0_results[3]['model']
# Updating cluster0 with kmeans labels
cluster0['KMeans_Cluster'] = best_c0_model.predict(cluster0_scaled)


# Cluster 1
cluster1_scaled, cluster1_results = km_cl(cluster1)
best_c1_model = cluster1_results[1]['model']
# Updating cluster1 with kmeans labels
cluster1['KMeans_Cluster'] = best_c1_model.predict(cluster1_scaled)

# Cluster 2
cluster2_scaled, cluster2_results = km_cl(cluster2)
best_c2_model = cluster2_results[1]['model']
# Updating cluster2 with kmeans labels
cluster2['KMeans_Cluster'] = best_c2_model.predict(cluster2_scaled)

# Cluster 3
cluster3_scaled, cluster3_results = km_cl(cluster3)
best_c3_model = cluster3_results[1]['model']
# Updating cluster3 with kmeans labels
cluster3['KMeans_Cluster'] = best_c3_model.predict(cluster3_scaled)

# Cluster 4
cluster4_scaled, cluster4_results = km_cl(cluster4)
best_c4_model = cluster4_results[1]['model']
# Updating cluster4 with kmeans labels
cluster4['KMeans_Cluster'] = best_c4_model.predict(cluster4_scaled)

# Cluster 5
cluster5_scaled, cluster5_results = km_cl(cluster5)
best_c5_model = cluster5_results[0]['model']
# Updating cluster5 with kmeans labels
cluster5['KMeans_Cluster'] = best_c5_model.predict(cluster5_scaled)

# Cluster 6
cluster6_scaled, cluster6_results = km_cl(cluster6)
best_c6_model = cluster6_results[1]['model']
# Updating cluster6 with kmeans labels
cluster6['KMeans_Cluster'] = best_c6_model.predict(cluster6_scaled)

# Cluster 7
cluster7_scaled, cluster7_results = km_cl(cluster7)
best_c7_model = cluster7_results[2]['model']
# Updating cluster7 with kmeans labels
cluster7['KMeans_Cluster'] = best_c7_model.predict(cluster7_scaled)

# Cluster 8
cluster8_scaled, cluster8_results = km_cl(cluster8)
best_c8_model = cluster8_results[2]['model']
# Updating cluster8 with kmeans labels
cluster8['KMeans_Cluster'] = best_c8_model.predict(cluster8_scaled)

# Cluster 9
cluster9_scaled, cluster9_results = km_cl(cluster9)
best_c9_model = cluster9_results[5]['model']
# Updating cluster9 with kmeans labels
cluster9['KMeans_Cluster'] = best_c9_model.predict(cluster9_scaled)

# Cluster 10
cluster10_scaled, cluster10_results = km_cl(cluster10)
best_c10_model = cluster10_results[2]['model']
# Updating cluster10 with kmeans labels
cluster10['KMeans_Cluster'] = best_c10_model.predict(cluster10_scaled)

# Cluster 11
cluster11_scaled, cluster11_results = km_cl(cluster11)
best_c11_model = cluster11_results[2]['model']
# Updating cluster11 with kmeans labels
cluster11['KMeans_Cluster'] = best_c11_model.predict(cluster11_scaled)

# Cluster 12
cluster12_scaled, cluster12_results = km_cl(cluster12)
best_c12_model = cluster12_results[2]['model']
# Updating cluster12 with kmeans labels
cluster12['KMeans_Cluster'] = best_c12_model.predict(cluster12_scaled)

# Cluster 13
cluster13_scaled, cluster13_results = km_cl(cluster13)
best_c13_model = cluster13_results[1]['model']
# Updating cluster13 with kmeans labels
cluster13['KMeans_Cluster'] = best_c13_model.predict(cluster13_scaled)

# Cluster 14
cluster14_scaled, cluster14_results = km_cl(cluster14)
best_c14_model = cluster14_results[2]['model']
# Updating cluster14 with kmeans labels
cluster14['KMeans_Cluster'] = best_c14_model.predict(cluster14_scaled)

# Cluster 15
cluster15_scaled, cluster15_results = km_cl(cluster15)
best_c15_model = cluster15_results[2]['model']
# Updating cluster15 with kmeans labels
cluster15['KMeans_Cluster'] = best_c15_model.predict(cluster15_scaled)

# Cluster 16
cluster16_scaled, cluster16_results = km_cl(cluster16)
best_c16_model = cluster16_results[3]['model']
# Updating cluster16 with kmeans labels
cluster16['KMeans_Cluster'] = best_c16_model.predict(cluster16_scaled)

# Cluster 17
cluster17_scaled, cluster17_results = km_cl(cluster17)
best_c17_model = cluster17_results[2]['model']
# Updating cluster17 with kmeans labels
cluster17['KMeans_Cluster'] = best_c17_model.predict(cluster17_scaled)

# Cluster 18
cluster18['KMeans_Cluster'] = 0

# Cluster 19
cluster19_scaled, cluster19_results = km_cl(cluster19)
best_c19_model = cluster19_results[0]['model']
# Updating cluster19 with kmeans labels
cluster19['KMeans_Cluster'] = best_c19_model.predict(cluster19_scaled)

# Cluster 20
cluster20_scaled, cluster20_results = km_cl(cluster20)
best_c20_model = cluster20_results[1]['model']
# Updating cluster20 with kmeans labels
cluster20['KMeans_Cluster'] = best_c20_model.predict(cluster20_scaled)

# Cluster 21
cluster21_scaled, cluster21_results = km_cl(cluster21)
best_c21_model = cluster21_results[1]['model']
# Updating cluster21 with kmeans labels
cluster21['KMeans_Cluster'] = best_c21_model.predict(cluster21_scaled)

# Cluster 22
cluster22_scaled, cluster22_results = km_cl(cluster22)
best_c22_model = cluster22_results[0]['model']
# Updating cluster22 with kmeans labels
cluster22['KMeans_Cluster'] = best_c22_model.predict(cluster22_scaled)

# Cluster 23
cluster23_scaled, cluster23_results = km_cl(cluster23)
best_c23_model = cluster23_results[1]['model']
# Updating cluster23 with kmeans labels
cluster23['KMeans_Cluster'] = best_c23_model.predict(cluster23_scaled)

# Cluster 24
cluster24_scaled, cluster24_results = km_cl(cluster24)
best_c24_model = cluster24_results[0]['model']
# Updating cluster24 with kmeans labels
cluster24['KMeans_Cluster'] = best_c24_model.predict(cluster24_scaled)

# Cluster 25
cluster25_scaled, cluster25_results = km_cl(cluster25)
best_c25_model = cluster25_results[2]['model']
# Updating cluster25 with kmeans labels
cluster25['KMeans_Cluster'] = best_c25_model.predict(cluster25_scaled)

# Cluster 26
cluster26_scaled, cluster26_results = km_cl(cluster26)
best_c26_model = cluster26_results[3]['model']
# Updating cluster26 with kmeans labels
cluster26['KMeans_Cluster'] = best_c26_model.predict(cluster26_scaled)

# Cluster 27
cluster27_scaled, cluster27_results = km_cl(cluster27)
best_c27_model = cluster27_results[1]['model']
# Updating cluster27 with kmeans labels
cluster27['KMeans_Cluster'] = best_c27_model.predict(cluster27_scaled)

# Cluster 28
cluster28_scaled, cluster28_results = km_cl(cluster28)
best_c28_model = cluster28_results[2]['model']
# Updating cluster28 with kmeans labels
cluster28['KMeans_Cluster'] = best_c28_model.predict(cluster28_scaled)

# Cluster 29
cluster29_scaled, cluster29_results = km_cl(cluster29)
best_c29_model = cluster29_results[2]['model']
# Updating cluster29 with kmeans labels
cluster29['KMeans_Cluster'] = best_c29_model.predict(cluster29_scaled)

KMeans_players = pd.concat([cluster0, cluster1, cluster2, cluster3, cluster4, cluster5, cluster6, cluster7, cluster8, cluster9,
                            cluster10, cluster11, cluster12, cluster13, cluster14, cluster15, cluster16, cluster17, cluster18,
                            cluster19, cluster20, cluster21, cluster22, cluster23, cluster24, cluster25, cluster26, cluster27,
                            cluster28, cluster29])

# Saving to files
KMeans_players.to_csv('Files/kmeans_players.csv', index=False)