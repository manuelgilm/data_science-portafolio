from dash import Dash 
import dash_core_components as dcc 
import dash_html_components as html 
import dash_bootstrap_components as dbc 
from dash.dependencies import Input, Output 
import plotly.graph_objs as go

import pandas as pd 
import numpy as np 

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# read dataset

df = pd.read_csv("data/WHO-COVID-19-global-data.csv")
df['Date_reported'] = pd.to_datetime(df['Date_reported'])
df = df[['Date_reported',' Country',' New_cases',' Cumulative_cases',' New_deaths',' Cumulative_deaths']]
df.columns = ['Date_reported','Country','New_cases','Cumulative_cases','New_deaths','Cumulative_deaths']

options = [{'label':op.replace("_"," "),'value':n} for n,op in enumerate(df.columns[2:].values)]
countries = [{'label':country,'value':n} for n,country in enumerate(df['Country'].unique())]

global_cases = pd.DataFrame()
count = []
for f in df['Date_reported'].unique():
    count.append((f,df[df['Date_reported']==f]['New_cases'].sum(),df[df['Date_reported']==f]['New_deaths'].sum()))
global_cases['Date_reported'] = df['Date_reported'].unique()
global_cases['Cases'] = [c for _,c,_ in count]
global_cases['Deaths'] = [c for _,_,c in count]
total_cases = global_cases['Cases'].sum()
total_deaths = global_cases['Deaths'].sum()

controls = [
    dbc.FormGroup([
        html.Div(html.H4('Options'),style = {'padding':1}),
        dcc.Dropdown(
            id = 'opt',
            options = options,
            value = 1,
            style ={'background':'rgb(218,218,218)'},
        ),
        html.Hr(),
        html.Div(html.H4('Countries'),style={'padding':1}),
        dcc.Dropdown(
            id = 'countries',
            options = countries,
            value = 0,
            style = {'background':'rgb(218,218,218)'},
            multi = True            
        ),
        html.Hr(),
        html.Div(html.H4('Graph Options')),
        dcc.Dropdown(
            id = 'gopt',
            options = [
                {'label':'Marker','value':0},
                {'label':'Line','value':1}
            ],
            value = 0,
            style = {'background':'rgb(218,218,218)'}
        )
    ]
    )
]

principal_graph =  [
    dcc.Graph(
        id = 'principal'
    )
]

graph_global_cases = [
    dcc.Graph(
        id = 'second',
        figure = {
            'data':[go.Bar(
                x = global_cases['Date_reported'],
                y = global_cases['Cases'],
                text = 'Cases'
            )],
            'layout':go.Layout(
                xaxis = {'title':'Date Reported'},
                yaxis = {'title':'New Cases'}
            )
        }       
    )
]

graph_global_deaths = [
    dcc.Graph(
        id = 'third',
        figure = {
            'data':[go.Bar(
                x = global_cases['Date_reported'],
                y = global_cases['Deaths'],
                text = 'Deaths'
            )],
            'layout':go.Layout(
                xaxis = {'title':'Date Reported'},
                yaxis = {'title':'Deaths'}
            )
        }
    )
]
app.layout = dbc.Container(
    fluid = True,
    children = [
        dbc.Row([
            html.Img(className="logo", src=app.get_asset_url("logo.png")),
            html.Div(html.H2('Search by Country')),
        ]),
        dbc.Row([
            html.Div(html.H1("Coronavirus Disease Dashboard"))
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col(controls,md=3),
            dbc.Col(principal_graph,md=9)
        ]),
        dbc.Row([
            html.Div(html.H1("Global Situation"))
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Col([html.Div(html.H1(total_cases),style = {'color':'rgb(128,128,128)'}),html.Div(html.H4("Confirmed Cases"))],md = 3),
            dbc.Col(graph_global_cases,md = 9)
        ]),
        html.Hr(),        
        dbc.Row([
            dbc.Col([html.Div(html.H1(total_deaths),style = {'color':'rgb(128,128,128)'}),html.Div(html.H4("Deaths"))],md = 3),
            dbc.Col(graph_global_deaths,md = 9)
        ])
    ]
)

@app.callback(
    Output(component_id = 'principal', component_property = 'figure'),
    [Input(component_id = 'opt', component_property = 'value'),
     Input(component_id = 'countries', component_property = 'value'),
     Input(component_id = 'gopt', component_property = 'value')]
    )
def create_graph(option, country,grap_options):

    feature = options[option]['label']
    
    country = np.array([country]).flatten()
    country = [countries[c]['label'] for c in country]
        
    if grap_options == 0:
        mode = 'markers'
    else:
        mode = 'lines'
        
    traces = []
        
    for count in country:
        traces.append(go.Scatter(
            x = df["Date_reported"],
            y = df[df['Country']==count][feature.replace(" ","_")],
            mode = mode,
            name = count,
            marker = {
                'line_width':3
            },
            text = count
        ))
    
    layout = go.Layout(
        title = feature,
        xaxis = {'title':'Date Reported'},
        yaxis = {'title':feature}
    )
    return {'data':traces,'layout':layout}

if __name__ == "__main__":
    app.run_server()
    