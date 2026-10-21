import base64
from html import escape
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from demo_salesdatabase.dal.customerDAO import CustomerDAO

df = CustomerDAO.getCustomers()
print(df)
print(df.columns)
def showHistogram(df, columns):
    plt.figure(1, figsize=(7, 8))
    n = 0
    for column in columns:
        n += 1
        plt.subplot(3, 1, n)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        sns.histplot(df[column], bins=32)
        plt.title(f'Histogram of {column}')
    plt.show()

showHistogram(df, df.columns[1:])

def elbowMethod(df, columnsForElbow):
    X = df.loc[:, columnsForElbow].values
    cluster_counts = np.arange(1, min(10, len(X)) + 1)
    inertia = []
    for n in cluster_counts:
        model = KMeans(n_clusters = n,
                       init='k-means++',
                       max_iter=500,
                       random_state=42,
                       n_init=10)
        model.fit(X)
        inertia.append(model.inertia_)

    if len(cluster_counts) >= 3:
        relative_improvements = (
            np.array(inertia[:-1]) - np.array(inertia[1:])
        ) / np.array(inertia[:-1])
        improvement_drops = relative_improvements[:-1] - relative_improvements[1:]
        optimal_k = int(cluster_counts[1:-1][np.argmax(improvement_drops)])
    else:
        optimal_k = int(cluster_counts[-1])

    plt.figure(1, figsize = (15, 6))
    plt.plot(cluster_counts, inertia, 'o-')
    plt.axvline(
        optimal_k,
        color='red',
        linestyle='--',
        label=f'Optimal k = {optimal_k}'
    )
    plt.legend()
    plt.xlabel('Number of Clusters'), plt.ylabel('Cluster sum of squared distances')
    plt.show()
    return optimal_k

columns=['Age','Spending_Score']
optimal_k = elbowMethod(df, columns)
print(f'Cụm tối ưu ta nên gom = {optimal_k}')

def runKMeans(X, cluster):
    model = KMeans(n_clusters=cluster,
                   init='k-means++',
                   max_iter=500,
                   random_state=42)
    model.fit(X)
    labels = model.labels_
    centroids = model.cluster_centers_
    y_kmeans = model.fit_predict(X)
    return y_kmeans, centroids, labels

X = df.loc[:, columns].values
cluster = 4
colors = ["red", "green", "blue", "purple", "black", "pink", "orange"]

y_kmeans, centroids, labels = runKMeans(X, cluster)
print(y_kmeans)
print(centroids)
print(labels)
df["cluster"] = labels

def visualizeKMeans(X,y_kmeans,cluster,title,xlabel,ylabel,colors):
    plt.figure(figsize=(10, 10))
    for i in range(cluster):
        plt.scatter(X[y_kmeans == i, 0],
                    X[y_kmeans == i, 1],
                    s=100,
                    c=colors[i],
                    label='Cluster %i'%(i+1))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

visualizeKMeans(X,
                y_kmeans,
                cluster,
                "Clusters of Customers - Age X Spending Score",
                "Age",
                "Spending_Score",
                colors)
print(df.columns)

def export_df2excel(df, filename):
    if "cluster" not in df.columns:
        raise ValueError("DataFrame must contain a 'cluster' column")
    if df.empty:
        raise ValueError("DataFrame is empty; no cluster sheets can be created")

    with pd.ExcelWriter(filename) as writer:
        for cluster_id, cluster_df in df.groupby("cluster", sort=True):
            sheet_name = f"Cluster_{cluster_id}"
            cluster_df.to_excel(writer, sheet_name=sheet_name, index=False)
export_df2excel(df, r"G:\2. Machine Learning\ML_261BIM401402\demo_salesdatabase\results\customer_clusters.xlsx")

def export_df2web(df, filename):
    if "cluster" not in df.columns:
        raise ValueError("DataFrame must contain a 'cluster' column")
    if df.empty:
        raise ValueError("DataFrame is empty; no cluster page can be created")
    cluster_ids = range(4)
    if not df["cluster"].isin(cluster_ids).all():
        raise ValueError("The 'cluster' column must contain only cluster IDs 0, 1, 2, or 3")

    cluster_groups = [
        (cluster_id, df.loc[df["cluster"] == cluster_id])
        for cluster_id in cluster_ids
    ]
    chart_groups = [
        (cluster_id, cluster_df)
        for cluster_id, cluster_df in cluster_groups
        if not cluster_df.empty
    ]

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.pie(
        [len(cluster_df) for _, cluster_df in chart_groups],
        labels=[f"Cluster {cluster_id}" for cluster_id, _ in chart_groups],
        autopct="%1.1f%%",
        startangle=90,
    )
    axis.set_title("Customer distribution by cluster")
    axis.axis("equal")

    image_buffer = BytesIO()
    figure.savefig(image_buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    chart = base64.b64encode(image_buffer.getvalue()).decode("ascii")

    tab_buttons = []
    cluster_panels = []
    for cluster_id, cluster_df in cluster_groups:
        panel_id = f"cluster-panel-{cluster_id}"
        active = cluster_id == 0
        tab_buttons.append(
            f'<button type="button" class="cluster-tab{" active" if active else ""}" '
            f'id="cluster-tab-{cluster_id}" role="tab" aria-controls="{panel_id}" '
            f'aria-selected="{str(active).lower()}" onclick="showCluster({cluster_id})">'
            f'Cluster {cluster_id}</button>'
        )
        table = cluster_df.to_html(index=False, border=0, classes="customer-table")
        cluster_panels.append(
            f'<section class="cluster-panel" id="{panel_id}" role="tabpanel" '
            f'aria-labelledby="cluster-tab-{cluster_id}"{" hidden" if not active else ""}>'
            f'<h2>Cluster {cluster_id} ({len(cluster_df)} customers)</h2>{table}</section>'
        )

    html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Customer clusters</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; }}
        img {{ display: block; margin: 0 auto 2rem; max-width: 100%; height: auto; }}
        .cluster-tabs {{ display: flex; gap: 0.5rem; border-bottom: 1px solid #ccc; }}
        .cluster-tab {{ background: #f2f2f2; border: 1px solid #ccc; border-bottom: 0; border-radius: 0.4rem 0.4rem 0 0; cursor: pointer; padding: 0.7rem 1rem; }}
        .cluster-tab.active {{ background: #fff; font-weight: bold; }}
        .cluster-panel {{ margin: 1rem 0 2rem; overflow-x: auto; }}
        .customer-table {{ border-collapse: collapse; width: 100%; }}
        .customer-table th, .customer-table td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; }}
        .customer-table th {{ background: #f2f2f2; }}
        .customer-table tr:nth-child(even) {{ background: #fafafa; }}
    </style>
    <script>
        function showCluster(clusterId) {{
            document.querySelectorAll(".cluster-panel").forEach(panel => {{
                panel.hidden = panel.id !== `cluster-panel-${{clusterId}}`;
            }});
            document.querySelectorAll(".cluster-tab").forEach(tab => {{
                const selected = tab.id === `cluster-tab-${{clusterId}}`;
                tab.classList.toggle("active", selected);
                tab.setAttribute("aria-selected", String(selected));
            }});
        }}
    </script>
</head>
<body>
    <h1>Customer clusters</h1>
    <h2>Customer distribution</h2>
    <img src="data:image/png;base64,{chart}" alt="Pie chart of customers by cluster">
    <div class="cluster-tabs" role="tablist" aria-label="Customer clusters">
        {''.join(tab_buttons)}
    </div>
    {''.join(cluster_panels)}
</body>
</html>
"""

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_document, encoding="utf-8")

export_df2web(df,r"G:\2. Machine Learning\ML_261BIM401402\demo_salesdatabase\results\index.html")