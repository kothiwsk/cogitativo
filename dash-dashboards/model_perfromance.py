#!/usr/bin/env python
# coding: utf-8
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd

app = dash.Dash()

df_projections = pd.read_feather("july_projections_all.feather")
df_daily = pd.read_feather("july_daily_predictions_all.feather")

available_entities = df_projections.level_1.unique()
available_metrics = df_daily.surgery_type.unique()


app.layout = html.Div([
    
    html.Link(
        rel='stylesheet',
        href='/static/dash.css'
    ),

    html.Div([

        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i} for i in available_entities],
                value='Enterprise'
            ),
            dcc.RadioItems(
                id='crossfilter-xaxis-type',
                options=[{'label': i, 'value': i} for i in \
                         ['2020-07-01', '2020-07-07', '2020-07-14', \
                          '2020-07-21', '2020-07-28']],
                value='2020-07-14',
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i} for i in available_metrics],
                value='elective'
            ),
            dcc.RadioItems(
                id='crossfilter-yaxis-type',
                options=[{'label': i, 'value': i} for i in ['IP', 'OP','ALL']],
                value='IP',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': 'East Florida'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='x-time-series'),
        dcc.Graph(id='y-time-series'),
    ], style={'display': 'inline-block', 'width': '45%','padding': '0 20'})

#     html.Div(dcc.Slider(
#         id='crossfilter-year--slider',
#         min=df['Year'].min(),
#         max=df['Year'].max(),
#         value=df['Year'].max(),
#         step=None,
#         marks={str(year): str(year) for year in df['Year'].unique()}
#     ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
])

@app.callback(
    dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
    [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-type', 'value')])
     #dash.dependencies.Input('crossfilter-year--slider', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type):
                 #year_value):
    dff = df_projections[df_projections['level_1'] == xaxis_column_name]
    dff = dff[dff['forecast_date'] == xaxis_type]
    dff = dff[dff['pattype'] == yaxis_type]
    x_col = "qmris_"+yaxis_column_name+"_%"
    y_col = "ds_"+yaxis_column_name+"_%"

    return {
        'data': [go.Scatter(
            x=dff[x_col],
            y=dff[y_col],
            text=dff['level_2'],
            customdata=dff['level_2'],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            }
        )],
        'layout': go.Layout(
            xaxis={
                'title': x_col,
                'type': 'linear',
                'range': [-100,100],
                'dtick': 20
            },
            yaxis={
                'title': y_col,
                'type': 'linear',
                'range': [-100,100],
                'dtick': 20
            },
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450,
            hovermode='closest'
        )
    }

def create_time_series(dff, title):
    return {
        'data': [go.Scatter(
            x=dff['appointment_date'],
            y=dff['actual_counts'],
            mode='lines+markers',
            name='actual'
        ),
                 go.Scatter(
            x=dff['appointment_date'],
            y=dff['daily_counts'],
            mode='lines+markers',
            name="predicted"
        )],
        'layout': {
            'height': 225,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': title
            }],
            'yaxis': {'type': 'linear'},
            'xaxis': {'showgrid': False}
        }
    }

@app.callback(
    dash.dependencies.Output('x-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-type', 'value')])
def update_y_timeseries(hoverData, xaxis_column_name, yaxis_column_name,xaxis_type,yaxis_type):
    entity_name = hoverData['points'][0]['customdata']
    dff = df_daily[df_daily['level_2'] == entity_name]
    dff = dff[dff['report_ts'] == xaxis_type]
    dff = dff[dff['pattype'] == yaxis_type]
    dff = dff[dff['surgery_type'] == yaxis_column_name]
    dff = dff.sort_values("appointment_date")

    
    title = entity_name+" "+yaxis_column_name+" "+yaxis_type
    return create_time_series(dff, title)

@app.callback(
    dash.dependencies.Output('y-time-series', 'figure'),
    [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
     dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
     dash.dependencies.Input('crossfilter-yaxis-type', 'value')])
def update_x_timeseries(hoverData, xaxis_column_name, yaxis_column_name,xaxis_type, yaxis_type):
    dff = df_daily[df_daily["level_2"]==xaxis_column_name]
    dff = dff[dff['report_ts'] == xaxis_type]
    dff = dff[dff['pattype'] == yaxis_type]
    dff = dff[dff['surgery_type'] == yaxis_column_name]
    dff = dff.sort_values("appointment_date")

    title = xaxis_column_name+" "+yaxis_column_name+" "+yaxis_type
    return create_time_series(dff, title)

# app.css.append_css({
#     'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
# })

if __name__ == '__main__':
    app.run_server(
        host='10.230.10.96',
        port=8053)