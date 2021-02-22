import os
import base64
import datetime
import io

import plotly.graph_objects as go

import pandas as pd
import numpy as np

from dash import Dash
import dash_core_components as dcc
import dash_html_components as html 
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_table

from dash_extensions import Download
from dash_extensions.snippets import send_data_frame

from utils import parse_contents, make_simple_graph, make_graph_with_anomalies
from detectors import make_outlier_detector, create_clean_time_serie
from adtk.data import validate_series


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Time Series - Outlier Detection"
server = app.server
i = 0
i2 = 0

root = os.getcwd()
image_filename =os.path.join(root,"data/images/keyboard.png")
encoded_image = base64.b64encode(open(image_filename, 'rb').read())
save_path = os.path.join("data/files")

NAVBAR = dbc.Navbar(
    children = [
        html.A(
            dbc.Row(
                [
                    dbc.Col(html.Img(src='data:image/png;base64,{}'.format(encoded_image), height = "60px")),
                    dbc.Col(
                        dbc.NavbarBrand("Time Series - Outlier Detection", className="ml-2",style = {"color":"white","font-size":46})
                    ),
                ],
                align="center",
                no_gutters = True,
            ),
            href = "https://www.google.com",
        )
    ],
    fixed = "top",
    color = "black",
    dark = True,
    sticky = "top",
)

controls = [
    dbc.Card([
        html.H4("Controls"),
        html.Label("Upload data"),
        dcc.Upload(
            id = "upload-data",
            children = html.A('Select a File'),
            # allow multiple files to be uploaded
            multiple = True,
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'solid',
                'borderRadius': '5px',
                'textAlign': 'center'
            }
        ),
        html.Hr(),
        html.Div(dbc.Input(id="high-parameter", placeholder="high", type="number"),style={"padding":"10px 0px 10px 0px"}),
        html.Div(dbc.Input(id="low-parameter", placeholder="low", type="number"),style={"padding":"10px 0px 10px 0px"}),
        html.Hr(),
        html.Div([
            html.Div(dbc.Button('Apply', id='run-button', n_clicks=0),style={"padding":"10px 0px 10px 0px"}),
            html.Div(dbc.Button('Download csv', id='download-button', n_clicks=0),style={"padding":"10px 0px 10px 0px","align":"center"})      
        ], style = {"justify":"center"}),
        html.Div(Download(id="download")),
        html.Div(id="intermediate-level", style={'display': 'none'}),
        html.Div(id="intermediate-level-2", style={'display': 'none'})
    ],
    body = True
    ),
]


graphs = [
    html.Div(
            className = 'panel panel-default',
            children = [
                html.Div(html.P("Time Series"),className="panel-heading2"),
                html.Div(className='panel-heading'),
                dcc.Graph(
                        id = 'origin-ts',
                        config = {'displaylogo':False},
                    ),
            ]
        ) 
]

@app.callback([
    Output(component_id = 'intermediate-level-2', component_property = 'children'),
    Output(component_id = 'origin-ts', component_property = 'figure')
],
[
    Input(component_id = 'intermediate-level', component_property = 'children'),
    Input(component_id = 'high-parameter', component_property = 'value'),
    Input(component_id = 'low-parameter', component_property = 'value'),
    Input(component_id = 'run-button', component_property = 'n_clicks')
])
def run_detection(df_data, high, low, n_clicks):

    global i
    
    try:
        time_serie_df = pd.read_json(df_data, orient = 'split')
        original_ts = make_simple_graph(time_serie_df, "Original Time Series", "blue", "lines",1)
    except:
        original_ts = []

    if n_clicks > i:
        i = n_clicks

        outlier_detector = make_outlier_detector('ThresholdAD', high, low)

        s = validate_series(pd.Series(data = time_serie_df['y'].values,index = pd.to_datetime(time_serie_df['ds'])))
        anomalies = outlier_detector.detect(s)
        anomaly_graph, shapes = make_graph_with_anomalies(s,anomalies)
        
        clean_time_serie = create_clean_time_serie(s,anomalies)
        df_clean_time_serie = pd.DataFrame({"y":clean_time_serie,"ds":clean_time_serie.index})
        clean_trace = make_simple_graph(df_clean_time_serie, "Clean Time Series", "red", "lines",1)
        original_ts = make_simple_graph(time_serie_df, "Original Time Series", "blue", "lines",0.5)


        traces = [clean_trace[0],original_ts[0]]

        graph = {
            'data':traces,
            'layout':go.Layout( 
                yaxis = {'title':s.name},
                hovermode = 'closest',
                showlegend = True,
            )
        }    

        return df_clean_time_serie.to_json(date_format='iso', orient='split'), graph

    graph = {
        'data':original_ts,
        'layout':go.Layout( 
            yaxis = {'title':"Y Axis"},
            hovermode = 'closest',
            showlegend = False
        )
    }
    
    return df_data, graph 


@app.callback(Output(component_id = 'intermediate-level', component_property = 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]

        time_serie_df = children[0][0]
        filename = children[0][1]

        return time_serie_df.to_json(date_format='iso', orient='split')

@app.callback(Output(component_id="download",component_property = "data"),
             [Input(component_id = "download-button", component_property = "n_clicks"),
              Input(component_id = 'intermediate-level-2', component_property = "children")])
def generate_download(n_clicks,json_data):
    
    global i2

    if n_clicks > i2:
        df_data = pd.read_json(json_data, orient='split')
        i2 = n_clicks
        return send_data_frame(df_data.to_csv,filename = "processed_data.csv")

BODY = dbc.Container(
    fluid = True,
    children = [
        html.Div(),
        dbc.Row([
            dbc.Col(dbc.CardBody(controls), md = 2),
            dbc.Col(
                [   
                    graphs[0],
                ],
                md = 10)
        ]),
    ]
)


app.layout = html.Div(children=[NAVBAR,BODY])


if __name__ == "__main__":
    app.run_server(debug=True)