# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


sns.set_palette('icefire')
sns.set_style('darkgrid')
sns.set_context('paper')

# Load dataset
data = pd.read_csv('/home/brodney01/mysite/data/mktcmpdata.csv')

# Clean any whitespace from column names
data.columns = data.columns.str.replace(' ', '')

# Removing erronous entries from Year of Birth [YOB] (<= 1900)
data = data[data['YOB'] > 1900].reset_index(drop=True)

plt.figure(figsize=(3,4))
data['YOB'].plot(kind='box', patch_artist=True)

# Cleaning the Income feature

data['Income'] = data['Income'].str.replace('$', '')
data['Income'] = data['Income'].str.replace(',', '').astype('float')
data['Income'] = data['Income'].fillna(data['Income'].median())

# Variable consolidation

ts_col = [col for col in data.columns if "Amt" in col]
pur_col = [col for col in data.columns if "Purchase" in col]
cmp_col = [col for col in data.columns if "Camp" in col]

data['Children'] = data['Kidhome'] + data['Teenhome']
data['Account_Year'] = pd.DatetimeIndex(data['Orig_Customer']).year
data['Total_Spent'] = data[ts_col].sum(axis=1)
data['Total_Purchases'] = data[pur_col].sum(axis=1)
data['Total_Campaign'] = data[cmp_col].sum(axis=1)

data['Country_code'] = data['Country'].replace({'SP': 'ESP', 'CA': 'CAN', 'US': 'USA', 'SA': 'ZAF', 'ME': 'MEX'})

cam_success = pd.DataFrame(data[['Camp1', 'Camp2', 'Camp3', 'Camp4', 'Camp5', 'RecentCamp']].mean()*100, 
                           columns=['Percent']).reset_index()

# success of campaigns by country code
df_cam = data[['Country_code', 'Camp1', 'Camp2', 'Camp3', 'Camp4', 'Camp5', 'RecentCamp']].melt(
    id_vars='Country_code', var_name='Campaign', value_name='Accepted (%)')
df_cam = pd.DataFrame(df_cam.groupby(['Country_code', 'Campaign'])['Accepted (%)'].mean()*100).reset_index(drop=False)

# rename the campaign variables so they're easier to interpret
df_cam['Campaign'] = df_cam['Campaign'].replace({'Camp1': '1',
                                                'Camp2': '2',
                                                'Camp3': '3',
                                                'Camp4': '4',
                                                'Camp5': '5',
                                                 'RecentCamp': 'Most recent'
                                                })

# choropleth plot

fig1 = px.choropleth(df_cam, locationmode='ISO-3', color='Accepted (%)', facet_col='Campaign', facet_col_wrap=2,
                    facet_row_spacing=0.05, facet_col_spacing=0.01, width=700,
                    locations='Country_code', projection='natural earth', title='Marketing Campaign Success Rate by Country'
                   )


# Merging original country codes into dataset
countries = data[['Country', 'Country_code']].drop_duplicates().reset_index(drop=True)
df_cam2 = df_cam.merge(countries, how='left', on='Country_code')


# Graphing results
fig3 = px.bar(df_cam2, x='Country', y='Accepted (%)', facet_col='Campaign', color = 'Campaign',
title='Regional Effects on Campaign Effectiveness')

chart1 = px.pie(cam_success, values='Percent', names='index',
               title='Success Rates of Marketing Campaigns',
                height = 500)
chart1.update(layout=dict(title=dict(x=0.5)))

chart2 = px.pie(data, values='Total_Campaign', names='Country',
               title='Overall Positive Campaign Response by Country',
                height = 500)
chart2.update(layout=dict(title=dict(x=0.5)))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

fig1 = dcc.Graph(
        id='fig1',
        figure=fig1,
        className="four columns"
    )
chart1 = dcc.Graph(
        id='chart1',
        figure=chart1,
        className="four columns"
    )
chart2 = dcc.Graph(
        id='chart2',
        figure=chart2,
        className="four columns"
    )
fig3 = dcc.Graph(
        id='fig3',
        figure=fig3,
        className="twelve columns"
    )

table = dash_table.DataTable(
    id="table",
    columns=[{"name": i, "id": i} for i in data.columns],
    data=data.to_dict("records"),
    page_size=20,  
    style_table={'height': '300px', 'overflowY': 'auto'},
    export_format='xlsx',
    export_headers='display',
    merge_duplicate_headers=True
)
# setup the header
header = html.H2(children="Marketing Campaign Results")

# setup to rows, graph 1-3 in the first row, and graph 4 in the second:
row1 = html.Div(children=[chart1, chart2])
row2 = html.Div(children=[fig1, fig3])

# setup & apply the layout
layout = html.Div(children=[header, row1, row2, table], style={"text-align": "center"})
app.layout = layout

if __name__ == '__main__':
    app.run_server(debug=False)