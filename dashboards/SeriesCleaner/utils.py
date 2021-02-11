import os
import base64
import datetime
import io

import plotly.graph_objects as go

import pandas as pd 

actual_color = 'black'
marker_size = 3


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    return df, filename

def make_simple_graph(df, name, color, mode,alpha):

    return [go.Scatter(
        name = name,
        x = df["ds"],
        y = df["y"],
        marker = {
            'color':color,
            'size':marker_size
        },
        mode = mode,
        opacity = alpha
    )]

def make_graph_with_anomalies(s,anomalies):

    lines = []
    ymin = min(s)
    ymax = max(s)
    for val , idx in zip(anomalies.values,anomalies.index):
        if val:
            lines.append({
                'type':'line',
                'xref':'x',
                'yref':'y',
                'x0':idx,
                'y0':ymin,
                'x1':idx,
                'y1':ymax,
                'line':{
                    'color':'red',
                    'width':1
                }
            })
    
    trace = [
        go.Scatter(
            name = s.name,
            x = s.index,
            y = s.values,
            mode = 'lines',
            marker = {
                'color':anomalies.values,
                'size':marker_size
            }
        )
    ]

    return trace, lines


def create_clean_time_serie(ts,anomalies):
    
    anomalies = anomalies.fillna(0)
    anomalies = list(map(bool,anomalies))
    ts.mask(cond = anomalies, other = None,inplace = True)
    return  ts

