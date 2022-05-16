import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html

app = dash.Dash()

app.layout = html.Div([
    dcc.Input(id='my-id', value='initial value', type="text"),
    dcc.Input(id='my-id2', value='initial value', type="text"),
    html.Button('Click Me', id='button'),
    html.Div(id='my-div')
])

@app.callback(
    Output(component_id='my-div', component_property='children'),
    Input('button', 'n_clicks'),
    State(component_id='my-id', component_property='value'),
    State("my-id2", 'value')
)
def update_output_div(n_clicks, input_value, input_value2):
    return 'You\'ve entered "{}" "{}" and clicked {} times'.format(input_value, input_value2, n_clicks)

if __name__ == '__main__':
    app.run_server()
