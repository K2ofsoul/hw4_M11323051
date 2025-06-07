import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import numpy as np

np.random.seed(42)

drinks = ['Coke', 'Pepsi', 'Sprite', '7Up', 'Cappuccino', 'Espresso', 'Latte', 'Mocha']
classes = list("ABCDEFGH")
counts = [100, 200, 100, 400, 400, 200, 100, 100]
amount_means = [150, 200, 200, 400, 700, 700, 800, 900]
amount_stds = [20, 10, 10, 100, 10, 10, 300, 400]
quantity_ranges = [(500, 1000), (500, 1000), (500, 1000), (500, 1000),
                   (1, 500), (1, 500), (1, 500), (1, 500)]
ranks = [8, 7, 6, 5, 4, 3, 2, 1]

rows = []
for i in range(len(drinks)):
    count = counts[i]
    for _ in range(count):
        rows.append({
            'Class': classes[i],
            'Drink': drinks[i],
            'Rank': ranks[i],
            'Amount': np.random.normal(amount_means[i], amount_stds[i]),
            'Quantity': np.random.randint(*quantity_ranges[i])
        })
df = pd.DataFrame(rows)

df_encoded = pd.get_dummies(df, columns=["Drink"])


features = df_encoded.drop(columns=["Class"])
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(features)

tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(X_scaled)
df['TSNE-1'] = tsne_result[:, 0]
df['TSNE-2'] = tsne_result[:, 1]

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("📊 t-SNE Visualization with Linked Selection"),
    dcc.Graph(
        id='scatter-plot',
        figure=px.scatter(df, x='TSNE-1', y='TSNE-2', color='Class', hover_data=['Drink']),
        style={'height': '600px'}
    ),
    html.H4("📋 Selected Points:"),
    html.Div(id='selected-data')
])


@app.callback(
    Output('selected-data', 'children'),
    Input('scatter-plot', 'selectedData')
)
def display_selected_data(selectedData):
    if selectedData is None:
        return "請使用滑鼠圈選區域來查看資料點資訊。"
    else:
        indices = [point['pointIndex'] for point in selectedData['points']]
        selected_rows = df.iloc[indices]
        return html.Pre(selected_rows[['Drink', 'Class', 'Rank', 'Amount', 'Quantity']].to_string(index=False))


if __name__ == '__main__':
    app.run(debug=True)
