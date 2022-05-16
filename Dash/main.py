# https://towardsdatascience.com/dash-for-beginners-create-interactive-python-dashboards-338bfcb6ffa4#:~:text=Dash%20is%20a%20python%20framework,dashboards%2C%20you%20only%20need%20python.

import dash
from dash import html
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri, globalenv
from rpy2.robjects.packages import importr
from rpy2.robjects import globalenv

app = dash.Dash()


lr_variables = ["ClimSST", "Temperature_Kelvin", "Temperature_Kelvin_Standard_Deviation", "SSTA_Frequency", "SSTA_Frequency_Standard_Deviation" ,"TSA_Frequency_Standard_Deviation" ,"mean_cur"]

df = px.data.stocks()

# Still map https://plotly.com/python/maps/
map_df = px.data.gapminder().query("year==2007")
map_fig = px.scatter_geo(map_df, locations="iso_alpha", color="continent",
                         hover_name="country", size="pop",
                         projection="natural earth")
map_fig.update_layout(title='Example world map')

# Animated map
ani_df = px.data.gapminder()
ani_fig = px.scatter_geo(ani_df, locations="iso_alpha", color="continent",
                         hover_name="country", size="pop",
                         animation_frame="year",
                         projection="natural earth")
ani_fig.update_layout(title="Example world map with animation")

# Try with our data
our_df = pd.read_csv("./lr_version_merged_mean.csv")
our_df['Date'] = our_df['Month'].astype(str) + '-' + our_df['Year'].astype(str)
our_df = our_df.sort_values(['Year', 'Month'], ascending=True)
our_fig = px.scatter_geo(our_df, lon="reef_longitude", lat="reef_latitude", color="mean_cur",
                         hover_name="index", size=(our_df["Average_bleaching"] + 10) * 100,
                         projection="natural earth",
                         animation_frame="Date",
                         center={"lon": 147.6992, "lat": -18.2871},
                         width=1000,
                         height=1000,
                         custom_data=['mean_cur', 'Average_bleaching', 'index', 'Date'])
our_fig.update_layout(title="Example map with our data")
# Update information shown in each frame when hovering over bubbles
# Need this nasty code cause dash is finicky https://stackoverflow.com/questions/67958555/plotly-express-animation-changes-in-hoverlabel-only-applied-for-the-first-frame
our_fig.update_traces(hovertemplate='<br>'.join(["Current: %{customdata[0]}",
                                                 "Average Bleaching: %{customdata[1]}",
                                                 "Index: %{customdata[2]}",
                                                 "Date: %{customdata[3]}"
                                                 ])
                      )
for f in our_fig.frames:
    f.data[0].update(hovertemplate='<br>'.join(["Current: %{customdata[0]}",
                                                "Average Bleaching: %{customdata[1]}",
                                                "Index: %{customdata[2]}",
                                                "Date: %{customdata[3]}"
                                                ])
                     )

# App
app.layout = html.Div(children=[
    # One graph
    html.Div(id='parent', children=[
        html.H1(id='H1', children='Styling using html components', style={'textAlign': 'center', \
                                                                          'marginTop': 40, 'marginBottom': 40}),

        dcc.Dropdown(id='dropdown',
                     options=[
                         {'label': 'Google', 'value': 'GOOG'},
                         {'label': 'Apple', 'value': 'AAPL'},
                         {'label': 'Amazon', 'value': 'AMZN'},
                     ],
                     value='GOOG'),
        dcc.Graph(id='bar_plot')
    ]),
    # Second graph
    html.Div("Hello! nested html div as second box."),

    # Trying out maps
    html.Div(children=[dcc.Graph(id="example_map", figure=map_fig)]),

    # Trying out animated maps
    html.Div(children=[dcc.Graph(id="animated_example_map", figure=ani_fig)]),

    # Trying map with callback
    html.Div(id="map_callback_test", children=[
        dcc.Dropdown(id='map_dropdown',
                     options=[
                         {'label': str(x), 'value': str(x)} for x in range(1952, 2008, 5)
                     ],
                     value='1952'),
        dcc.Graph(id='map_callback_test_map')
    ]),

    # Our graph: all data
    html.Div(id="our_fig_all_data", children=[dcc.Graph(id="our_fig_figure", figure=our_fig)]),

    # # Our graph: for prediction
    html.Div(id="lr_prediction", children=[
        html.Div([
                    dcc.Input(
                        id=_,
                        type="number",
                        placeholder="input " + _,
                        debounce=True
                    ) for _ in lr_variables
                ]),
        html.Br(),
        html.Div(id="lr_predicted")
    ])
]
)


r = robjects.r
pandas2ri.activate()
lr_model_path = "../Models/logistic.rds"
lr_model = r.readRDS(lr_model_path)
globalenv['lr_model'] = lr_model
# Show prediction result with logistic regression model
@app.callback(Output('lr_predicted', 'children'),
              Input('ClimSST', 'value'),
              Input('Temperature_Kelvin', 'value'),
              Input("Temperature_Kelvin_Standard_Deviation", "value"),
              Input("SSTA_Frequency", 'value'),
              Input("SSTA_Frequency_Standard_Deviation", "value"),
              Input("TSA_Frequency_Standard_Deviation", "value"),
              Input("mean_cur", "value")
              )
def lr_figure(Climsst, Temperature_Kelvin, Temperature_Kelvin_Standard_Deviation, SSTA_Frequency,
              SSTA_Frequency_Standard_Deviation, TSA_Frequency_Standard_Deviation, mean_cur):
    # Create r test data frame

    r('test_data <- data.frame(clim_sst=c(' + str(Climsst) + '),' +
      'temperature_kelvin=c(' + str(Temperature_Kelvin) + '),' +
      'temperature_kelvin_standard_deviation=c(' + str(Temperature_Kelvin_Standard_Deviation) + '),' +
      'ssta_frequency=c(' + str(SSTA_Frequency) + '),' +
      ' ssta_frequency_standard_deviation=c(' + str(SSTA_Frequency_Standard_Deviation) + '),' +
      ' tsa_frequency_standard_deviation=c(' + str(TSA_Frequency_Standard_Deviation) + '),' +
      ' mean_cur=c(' + str(mean_cur) + '))')
    r('pred <- predict(lr_model, test_data, type="raw", probability=FALSE)')
    predicted = r('pred')[0]
    if predicted > 0.50:
        #Bleach
        return "Bleaching probability: " + str(predicted) + ". Bleaching should occur!!"
    else:
        #no bleach
        return "Bleaching probability: " + str(predicted) + ". Bleaching should not occur!!"

# Function for map with callback
@app.callback(Output("map_callback_test_map", "figure"),
              Input('map_dropdown', 'value'))
def map_callback(year):
    df = px.data.gapminder().query("year==" + str(year))
    fig = px.scatter_geo(df, locations="iso_alpha", color="continent",
                         hover_name="country", size="pop",
                         projection="natural earth")
    fig.update_layout(title='Example world map in ' + str(year))
    return fig


@app.callback(Output(component_id='bar_plot', component_property='figure'),
              [Input(component_id='dropdown', component_property='value')])
def graph_update(dropdown_value):
    print(dropdown_value)
    fig = go.Figure([go.Scatter(x=df['date'], y=df['{}'.format(dropdown_value)], \
                                line=dict(color='firebrick', width=4))
                     ])

    fig.update_layout(title='Stock prices of ' + dropdown_value + ' over time',
                      xaxis_title='Dates',
                      yaxis_title='Prices'
                      )
    return fig


if __name__ == '__main__':
    app.run_server(port=8000)
