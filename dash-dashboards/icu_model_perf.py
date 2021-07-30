#!/usr/bin/env python
# coding: utf-8
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import joblib
import shap
import numpy as np


static_image_route = '/static/'
image_directory = '/opt/app/SandeepProjects/icu_predictor/python/large_set_4hrs/assets/'

train_cols_humanreadable = joblib.load("training_cols_humanreadable.pkl")

app = dash.Dash()

predictions = pd.read_csv("validationset_predictions_medvitalslabresultsloc.csv",dtype={"patient_dw_id":object})
predictions["consolidated_arrival"] = pd.to_datetime(predictions["consolidated_arrival"])
predictions["label_icunonicu"] = predictions.label.replace(["ICU","non-ICU","non-ICU-72"],[1,0,1])

estimator = joblib.load("model_allclinic.pkl") 
valid_x = pd.read_feather("validation_dataset_dashboard.csv")
valid_mapped = pd.read_feather("validation_dataset_dashboard_mapped.csv")
# valid_mapped["row_ts"] = valid_mapped["row_ts"].dt.strftime('%Y-%m-%d %H:%M:%S')
# valid_x["row_ts"] = valid_x["row_ts"].dt.strftime('%Y-%m-%d %H:%M:%S')

valid_x=valid_x.merge(predictions[["patient_dw_id","consolidated_arrival"]].drop_duplicates(),how="inner",on=["patient_dw_id"])
valid_mapped=valid_mapped.merge(predictions[["patient_dw_id","consolidated_arrival"]].drop_duplicates(),how="inner",on=["patient_dw_id"])


valid_x["row_ts_dt"] = valid_x.apply(lambda x:x.consolidated_arrival + pd.Timedelta(hours=x.row_ts),axis=1)
valid_x["consolidated_arrival"] = valid_x["consolidated_arrival"].dt.strftime('%Y-%m-%d %H:%M:%S' )
valid_x["row_ts_dt"] = valid_x["row_ts_dt"].dt.strftime('%Y-%m-%d %H:%M:%S')

valid_mapped["row_ts_dt"] = valid_mapped.apply(lambda x:x.consolidated_arrival + pd.Timedelta(hours=x.row_ts),axis=1)
valid_mapped["consolidated_arrival"] = valid_mapped["consolidated_arrival"].dt.strftime('%Y-%m-%d %H:%M:%S' )
valid_mapped["row_ts_dt"] = valid_mapped["row_ts_dt"].dt.strftime('%Y-%m-%d %H:%M:%S')



explainer = shap.TreeExplainer(estimator)

# population_locations = pd.read_csv("../../queries_python/population.csv.gz",dtype=object)
# population_locations.columns=[x.lower() for x in population_locations.columns]
# predictions = pd.read_csv("validationset_predictions_medvitalslabresultsloc.csv",dtype={"patient_dw_id":object})

# merged=population_locations.merge(predictions[["patient_dw_id","consolidated_arrival"]]\
# .drop_duplicates(),how="inner",on=["patient_dw_id"])

# merged = merged.sort_values(["patient_dw_id","loc_start_ts"])

# merged.reset_index(drop=True).to_feather("tmp_merged_dash.f")
# print(sdfsf)

merged=pd.read_feather("tmp_merged_dash.f")
predictions["row_ts_dt"] = predictions.apply(lambda x:x.consolidated_arrival + pd.Timedelta(hours=x.row_ts),axis=1)
predictions["consolidated_arrival"] = predictions["consolidated_arrival"].dt.strftime('%Y-%m-%d %H:%M:%S' )
predictions["row_ts_dt"] = predictions["row_ts_dt"].dt.strftime('%Y-%m-%d %H:%M:%S')

#image_filename = '/opt/app/SandeepProjects/icu_predictor/python/24hr_2pm/static/shap_example.png' # replace with your own image
#encoded_image = base64.b64encode(open(image_filename, 'rb').read())

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}

app.layout = html.Div([
    
    html.Link(
        rel='stylesheet',
        href='/static/dash.css'
    ),

    html.Div([
        html.H2('ICU predictor model performance', style=TEXT_STYLE),
        html.H4("Select Threshold", style=TEXT_STYLE),
        dcc.Slider(
            id='threshold-slider',
            min=0,
            max=1.0,
            step=0.01,
            value=0.5,
            tooltip = { 'always_visible': True }
        ),
        
        html.Div([
            html.H3("Performance at patient level", style=TEXT_STYLE),
            dcc.Graph(
                id='confusion-matrix-patlevel',
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),        
        
        html.Div([
            dcc.RadioItems(
            id='radiobutton-errors',
            options=[{'label': i, 'value': i} for i in ['TP', 'FP','FN','EarlyICUPrediction']],
            value='FP',
            labelStyle={'display': 'inline-block'}
        ),
            dcc.Graph(id='bar-plot-pat')
        ], style={'width': '49%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("Shapley importance plots", style=TEXT_STYLE),
            html.Img(src = app.get_asset_url('shapley_feat_importance.png'))
        ]),

        html.Div([
                dcc.RadioItems(
                    id='radiobutton-pattype',
                    options=[{'label': i, 'value': i} for i in ['M/S','PCU','ICU','Deceased','EarlyICUPrediction']],
                    value='M/S',
                    labelStyle={'display': 'inline-block'}
                ),
                dcc.Dropdown( id='patient-selector'),
                dcc.Graph( id='gantt-chart',hoverData={'points': [{}]}),
        ], style={'display': 'inline-block', 'width': '49%'}),
        
        html.Div([
                html.H3("Feature importance", style=TEXT_STYLE),
                dcc.Graph( id='shap-patient'),
        ], style={'display': 'inline-block', 'width': '49%','height':1600}),

        
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),
])


@app.callback(
    dash.dependencies.Output('confusion-matrix-patlevel', 'figure'),
    [dash.dependencies.Input('threshold-slider', 'value')])
def update_graph(threshold):
                 #year_value):
    predictions["prediction_label"] = (predictions["icu_prob"] > threshold)*1
    
    tmp1 = predictions.sort_values("icu_prob",ascending=False).drop_duplicates("patient_dw_id")
    tmp = pd.crosstab(tmp1.label_icunonicu, tmp1.prediction_label, \
                      rownames=['True'], colnames=['Predicted'], margins=True).values
    x=["non-icu","icu","total"]
    y=["non_icu","icu","total"]
    z_text = [[str(y) for y in x] for x in tmp]
    # set up figure 
    colorscale = [[0, 'orange'], [1, 'orange']]
    fig = ff.create_annotated_heatmap(tmp, x=x, y=y, annotation_text=z_text, colorscale=colorscale)
    fig['layout']['yaxis']['autorange'] = "reversed"

    return fig

@app.callback(
    dash.dependencies.Output('bar-plot-pat', 'figure'),
    [dash.dependencies.Input('threshold-slider', 'value'),
     dash.dependencies.Input('radiobutton-errors', 'value')])
def update_graph( threshold, rb_choice):
                 #year_value):
    predictions["prediction_label"] = (predictions["icu_prob"] > threshold)*1
    
    tmp1 = predictions.sort_values("icu_prob",ascending=False).drop_duplicates("patient_dw_id")


    if rb_choice == "FP":
        tmp = tmp1[((tmp1.prediction_label==1) & (tmp1.label_icunonicu==0))]
    elif rb_choice == "TP":
        tmp = tmp1[((tmp1.prediction_label==1) & (tmp1.label_icunonicu==1))]
    elif rb_choice == "FN":
        tmp = tmp1[((tmp1.prediction_label==0) & (tmp1.label_icunonicu==1))]
    elif rb_choice == "EarlyICUPrediction":
        tmp = tmp1[((tmp1.prediction_label==1) & (tmp1.label_4hr==0) & (tmp1.label_icunonicu==1))]

        
    tmp = tmp[["patient_dw_id","consolidated_arrival","label","discharge_summary"]].\
    sort_values(["patient_dw_id"]).drop_duplicates(["patient_dw_id"])

    tmp1 = tmp.merge(merged[["patient_dw_id","loc_acuity","loc_start_ts"]],how="inner")

    tmp1.loc_start_ts=pd.to_datetime(tmp1.loc_start_ts)

    tmp1.consolidated_arrival=pd.to_datetime(tmp1.consolidated_arrival)

    tmp1["time_elapsed"] = (tmp1.loc_start_ts - tmp1.consolidated_arrival).astype('timedelta64[h]')


    pat_counts = []
    deceased_pats=tmp1[((tmp1.discharge_summary=="Deceased"))].patient_dw_id.unique()

    ms_only = len(set(tmp1.patient_dw_id.unique()) - set(deceased_pats) - \
        set(tmp1[tmp1.loc_acuity.isin(["ICU","PCU","OR_Rad_NonBilled"])].patient_dw_id.unique()))

    pat_counts.append([ms_only,0])
    for items in ["ICU","PCU"]:
        tmp = [0]*2
        tmp[0]=len(tmp1[((tmp1.loc_acuity==items))].patient_dw_id.unique())
        tmp[1]=len(tmp1[((tmp1.loc_acuity==items) & (tmp1.time_elapsed < 80))].patient_dw_id.unique())
        pat_counts.append(tmp)

    pat_counts.append([len(deceased_pats),0])


    condition=['arrival-departure', 'arrival to 80 hours']

    fig = go.Figure(data=[
        go.Bar(name='M/S', x=condition, y=pat_counts[0]),
        go.Bar(name='PCU', x=condition, y=pat_counts[2]),
        go.Bar(name='ICU', x=condition, y=pat_counts[1]),
        go.Bar(name='Deceased', x=condition, y=pat_counts[3])
    ])
    # Change the bar mode
    fig.update_layout(barmode='group')
    return fig


# Load patients into dropdown
@app.callback(
    dash.dependencies.Output(component_id='patient-selector', component_property='options'),
    [
        dash.dependencies.Input('threshold-slider', 'value'),
        dash.dependencies.Input('radiobutton-errors', 'value'),
         dash.dependencies.Input('radiobutton-pattype', 'value')])
def populate_patient_selector( threshold, rb_choice, pattype):
    predictions["prediction_label"] = (predictions["icu_prob"] > threshold)*1
    
    tmp1 = predictions.sort_values("icu_prob",ascending=False).drop_duplicates("patient_dw_id")
    

    if rb_choice == "FP":
        tmp = tmp1[((tmp1.prediction_label==1) & (tmp1.label_icunonicu==0))]
    elif rb_choice == "TP":
        tmp = tmp1[((tmp1.prediction_label==1) & (tmp1.label_icunonicu==1))]
    elif rb_choice == "FN":
        tmp = tmp1[((tmp1.prediction_label==0) & (tmp1.label_icunonicu==1))]
    elif rb_choice == "EarlyICUPrediction":
        tmp = tmp1[((tmp1.prediction_label==1) & (tmp1.label_4hr==0) & (tmp1.label_icunonicu==1))]

        
    if pattype=="EarlyICUPrediction":
        tmp = tmp[((tmp.prediction_label==1) & (tmp.label_4hr==0) & (tmp.label_icunonicu==1))]
 
    tmp = tmp[["patient_dw_id","icu_prob","discharge_summary"]].\
    sort_values(["patient_dw_id"]).drop_duplicates(["patient_dw_id"])

    tmp = tmp.merge(merged[["patient_dw_id","loc_acuity"]],how="inner")
    tmp = tmp.sort_values("icu_prob",ascending=False)

        
    deceased_pats = tmp[((tmp.discharge_summary=="Deceased"))].patient_dw_id.unique()
    if pattype=="Deceased":
        patients=deceased_pats
    elif pattype=="M/S":
        tmp2 = []
        patients = set(tmp.patient_dw_id.unique()) - set(deceased_pats) - \
        set(tmp[tmp.loc_acuity.isin(["ICU","PCU","OR_Rad_NonBilled"])].patient_dw_id.unique())
        patients = [x for x in tmp.patient_dw_id.unique() if x in patients]

    elif pattype=="PCU":
        tmp2=[]
        tmp2=tmp[((tmp.loc_acuity=="PCU"))].patient_dw_id.unique()
        patients = set(tmp2) - set(deceased_pats) - set(tmp[tmp.loc_acuity.isin(["ICU"])].patient_dw_id.unique())
        patients = [x for x in tmp2 if x in patients]
    elif pattype=="ICU":
        tmp2=[]
        tmp2=tmp[((tmp.loc_acuity=="ICU"))].patient_dw_id.unique()
        patients = set(tmp2) - set(deceased_pats)
        patients = [x for x in tmp2 if x in patients]

    elif pattype=="EarlyICUPrediction":
        patients = set(tmp.patient_dw_id.unique()) - set(deceased_pats)
        patients = [x for x in tmp.patient_dw_id.unique() if x in patients]


        
    return [
        {'label': patient, 'value': patient}
        for patient in patients
    ]

# Load patients into dropdown
@app.callback(
    dash.dependencies.Output('gantt-chart','figure'),
    [
        dash.dependencies.Input('patient-selector', 'value'),
        dash.dependencies.Input('threshold-slider', 'value'),
    ]
)
def plot_gantt_chart( patient,threshold):
    print(patient)
    
    tmp=merged[merged.patient_dw_id==patient]
    print(tmp)
    arr = tmp.consolidated_arrival.values[0]
    tmp=tmp[["loc_start_ts","loc_end_ts","loc_acuity"]]
    tmp.columns=["Start","Finish","Task"]

    first_ts = tmp.Start.values[0].split(".")[0]
    if arr!=first_ts:
        tmp1 = pd.DataFrame([[arr,first_ts,"ER"]],
                            columns=["Start","Finish","Task"])
        tmp = tmp1.append(tmp)
    tmp = tmp.sort_values("Start")
    loc_names = tmp.Task.values.tolist()
    tmp=tmp.to_dict(orient='records')
    print(tmp)
    
    fig_tmp=ff.create_gantt(tmp)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    

    #fig_tmp=ff.create_gantt(tmp)

    for x in range(len(fig_tmp["data"])):
        fig.add_trace(fig_tmp["data"][x],secondary_y=False)

    preds=predictions[predictions.patient_dw_id==patient]
    print(preds[["patient_dw_id","row_ts_dt","icu_prob"]])

    preds = preds.sort_values("row_ts")
    fig.add_trace(dict(type='scatter',
                  x=preds.row_ts_dt.values.tolist(),
                  y=preds.icu_prob.values.tolist()),secondary_y=True)
        
    fig['layout']['yaxis2'].update( range=[0, 1])
    #fig['layout']['yaxis1'].update(tickvals=loc_names,ticktext = list(range(0,len(loc_names))))

    return fig


# Load patients into dropdown
@app.callback(
    dash.dependencies.Output('shap-patient','figure'),
    [
        dash.dependencies.Input('patient-selector', 'value'),
        dash.dependencies.Input('gantt-chart', 'hoverData')
    ]
)
def plot_shapley_importance( patient, row_ts):
    
    tmp = valid_x[((valid_x.patient_dw_id==patient))]
    tmp.row_ts_dt = tmp.row_ts_dt.apply(lambda x:x[0:16])
    tmp = tmp[(tmp.row_ts_dt==row_ts['points'][0]['x'][0:16])]
    del tmp["patient_dw_id"], tmp["row_ts_dt"],tmp["index"]
    del tmp["consolidated_arrival"]


    tmp_mapped = valid_mapped[((valid_mapped.patient_dw_id==patient))]
    tmp_mapped.row_ts_dt = tmp_mapped.row_ts_dt.apply(lambda x:x[0:16])
    tmp_mapped = tmp_mapped[(tmp_mapped.row_ts_dt==row_ts['points'][0]['x'][0:16])]
    del tmp_mapped["patient_dw_id"], tmp_mapped["row_ts_dt"],tmp_mapped["index"]
    del tmp_mapped["consolidated_arrival"]

    shap_values = explainer.shap_values(tmp)
    plot_df = pd.DataFrame(shap_values, columns=train_cols_humanreadable).T
    plot_df["values"] = tmp_mapped.values[0]

    lower = plot_df.sort_values(0).head(10)
    higher = plot_df.sort_values(0).tail(10)

    plot_df = pd.concat([higher,lower])
    plot_df=plot_df.reset_index()
    plot_df = plot_df[["index",0,"values"]]
    plot_df.columns=["feature","importance","value"]
    plot_df.feature = plot_df.apply(lambda x:x.feature+"-"+str(x.value) if type(x.value)==str or x.value\
                                    >-100 else x.feature+"-UNK" ,axis=1)


    plot_df["Color"] = np.where(plot_df["importance"]<0, 'red', 'blue')

    # Plot
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name='Net',
               x=plot_df["feature"],
               y=plot_df['importance'],
               marker_color=plot_df['Color']))
    fig.update_layout(barmode='stack')

    return fig

if __name__ == '__main__':
    app.run_server(
        host='10.230.10.96',
        #debug=True,
        port=8057)
