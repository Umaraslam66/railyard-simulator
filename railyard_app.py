import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import networkx as nx
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)

# --------------------------------------------------

# switches as nodes to the graph
switches = [11, 16, 12, 13, 52, 90, 91, 92, 93,94,95,96,97,98,99,100,101, 102, 112, 114, 116, 125, 123, 121, 129, 131, 132, 133, 134, 135, 152, 154, 156, 160, 162, 142, 193, 191, 200, 164]

coordinates = {
  11: (0.0, 1502.3), 16: (0.0, 0),
  12: (829.82, 751.15), 13: (1907.09, 751.15),
  52: (2081.00, 751.15), 90: (3223.09, 300),
  91: (1600, 650), 92: (3550, 751.5),
  93: (3620, 1400), 94: (3550, 5000.0),
  95: (3200, 4700), 96: (3125, 4700),
  97: (3050, 4700), 98: (2800, 4900),
  99: (2580, 4900), 100: (2966, 1300),
  101: (2966, 1200), 102: (2163.05, 751.15),
  112: (2520.80, 751.15), 114: (2650, 650),
  116: (2900, 500.0), 121: (3352.29, 751.15),
  123: (3223.09, 650), 125: (2966.45, 650),
  129: (3550, 1500.0), 131: (3450, 1750),
  132: (3550, 2000.0), 133: (3450, 2115),
  134: (3450, 2250.0), 135: (3550, 2400.0),
  152: (3550, 3100), 154: (3450, 3300.0),
  156: (3350, 3500), 160: (3200, 3900.0), 162:(3050,4200), 164:(3125,4100),
  142: (2941.72, 3650), 193: (3450, 5200.0),
  191: (3350, 5400.0), 164:(3125,4000)
}

# edges and their lengths
edges_with_lengths = [
    (11, 12, 468),  # (node1, node2, length)
    (16, 12, 468),
    (12, 13, 1132),
    (13,91,112), # end point, not a switch (Sky 13)
    (13,52,176),
    (52,90,2713), # end point, not a switch
    (52,102,89),
    (102,112,427),
    (112,121,1265),
    (112,114,49),
    (114,116,624),
    (116,123,555),
    (123,121,44),
    (121,92,75), # end point, not a switch
    (114,125,730),
    (125,123,447),
    (125,116,104),
    (102,129,1913),
    (129,93,135), # end point, not a switch (loco lineup 100)
    (129,132,375),
    (132,135,171),
    (135,134,64),
    (135,152,69), 
    (152,94,900) , # end point, not a switch
    (94,193,67),
    (193,191,40),
    (152,154,40),
    (154,193,900),
    (154,156,40),
    (156,160,102),
    (164,96,250),
    (162,97,300), # end point, not a switch (future track 103)
    (160,95,290), # end point, not a switch (Track 5 endpoint)
    (156,191,900),
    (132,133,100),
    (133,134,9),
    (133,131,280),
    (160,164,40),
    (164,162,100),
    (131,100,530), # end point, not a switch (102 track)
    (131,101,535), # end point, not a switch (101 track)
    (134,142,125), 
    (142,98,1000), # end point, not a switch ( coil loading track 7)
    (142,99,1000)  # end point, not a switch (coil loading track 8)
,]

polyline_edges = {
    (13, 91): {'segments': 2, 'offset': -50}, 
    (52, 90): {'segments': 2, 'offset': 240},
    (114, 116): {'segments': 2, 'offset':90}, 
    (116, 123): {'segments': 2, 'offset':90}
}

curve_edges = {
    (129, 93): {'control_offset': (1, 30)},
    # (142, 98): {'control_offset': (1, -500)},
    # (142, 99): {'control_offset': (1, -600)},
    #(131, 101): {'control_offset': (1, -400)},
}

mixed_edges = {
    (102, 129): {'straight_length': 250, 'control_offset': (1.5, -400),  'angle': np.radians(35), 'curve_first': False}, 
    (134, 142): {'straight_length': 300, 'control_offset': (1.5, 400),  'angle': np.radians(90), 'curve_first': False}, 
    (131, 100): {'straight_length': 150, 'control_offset': (1.5, -200),  'angle': np.radians(225), 'curve_first': False}, 
    (131, 101): {'straight_length': 200, 'control_offset': (1.5, -200),  'angle': np.radians(270), 'curve_first': False}, 
    (142, 98): {'straight_length': 450, 'control_offset': (1.5, -300),  'angle': np.radians(90), 'curve_first': True}, 
    (142, 99): {'straight_length': 600, 'control_offset': (1.5, -300),  'angle': np.radians(90), 'curve_first': True}, 
}

special_switches = [200] 
special_tracks = [] 

# --------------------------------------------------

# Function to create the Plotly figure using your code
def create_figure():
    # Create a new figure
    fig = go.Figure()

    # Define colors for different edge types
    polyline_color = '#1f77b4'
    curve_color = '#ff7f0e'
    mixed_color = '#2ca02c'
    node_color = 'lightblue'
    node_text_color = '#333333'
    background_color = '#f5f5f5'

    # Update the node plotting section with custom markers
    for switch in switches:
        if switch in coordinates:
            x, y = coordinates[switch]
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=12, color=node_color, line=dict(width=2, color='darkblue')),
                text=[f"{switch}"],
                textposition="bottom center",
                textfont=dict(color=node_text_color),
                hoverinfo='text',
                hovertext=[f"Switch: {switch}"],
                showlegend=False
            ))

    # Function definitions from your code
    # (Include the create_polyline, create_quadratic_bezier, create_mixed_edge functions here)

    # Function to create evenly spaced polyline segments between two nodes
    def create_polyline(x1, y1, x2, y2, segments, offset):
        xs = [x1]
        ys = [y1]
        
        for i in range(1, segments):
            t = i / segments
            mid_x = (1 - t) * x1 + t * x2
            mid_y = (1 - t) * y1 + t * y2

            # Perpendicular direction for offset (90-degree rotation)
            dx = x2 - x1
            dy = y2 - y1
            length = (dx ** 2 + dy ** 2) ** 0.5
            perp_x = -dy / length  # Perpendicular x direction
            perp_y = dx / length   # Perpendicular y direction

            # Apply offset in perpendicular direction
            mid_x += perp_x * offset * (-1)**i  # Alternate offsets for bending
            mid_y += perp_y * offset * (-1)**i
            
            xs.append(mid_x)
            ys.append(mid_y)

        xs.append(x2)
        ys.append(y2)
        
        return xs, ys

    # Function to create a quadratic Bézier curve
    def create_quadratic_bezier(x1, y1, x2, y2, control_offset):
        # Control point (midpoint with offset)
        control_x = (x1 + x2) / 2 + control_offset[0]
        control_y = (y1 + y2) / 2 + control_offset[1]
        
        t_values = np.linspace(0, 1, 100)
        bezier_x = (1 - t_values)**2 * x1 + 2 * (1 - t_values) * t_values * control_x + t_values**2 * x2
        bezier_y = (1 - t_values)**2 * y1 + 2 * (1 - t_values) * t_values * control_y + t_values**2 * y2
        
        return bezier_x, bezier_y

    # Function to create a mixed edge with a curve and a straight part
    def create_mixed_edge(x1, y1, x2, y2, straight_length, control_offset, angle=None, curve_first=False):
        if curve_first:
            # Curve followed by a straight line
            control_x = x1 + control_offset[0]
            control_y = y1 + control_offset[1]
            
            # Intersection point
            intersect_x = x2 - straight_length * np.cos(angle) if angle is not None else x1
            intersect_y = y2 - straight_length * np.sin(angle) if angle is not None else y1
            
            # Create the curve
            bezier_xs, bezier_ys = create_quadratic_bezier(x1, y1, intersect_x, intersect_y, control_offset)
            
            # Create the straight line
            straight_xs = [bezier_xs[-1], x2]
            straight_ys = [bezier_ys[-1], y2]
            
            xs = list(bezier_xs) + straight_xs
            ys = list(bezier_ys) + straight_ys
        else:
            # Straight line followed by a curve
            if angle is None:
                ratio = straight_length / np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                straight_x = x1 + ratio * (x2 - x1)
                straight_y = y1 + ratio * (y2 - y1)
            else:
                straight_x = x1 + straight_length * np.cos(angle)
                straight_y = y1 + straight_length * np.sin(angle)
            
            # Create the straight line
            straight_xs = [x1, straight_x]
            straight_ys = [y1, straight_y]
            
            # Create the curve
            bezier_xs, bezier_ys = create_quadratic_bezier(straight_x, straight_y, x2, y2, control_offset)
            
            xs = straight_xs + list(bezier_xs)[1:]
            ys = straight_ys + list(bezier_ys)[1:]

        return xs, ys

    # Add edges with lengths, using polyline, Bézier curve, or mixed edges
    for node1, node2, length in edges_with_lengths:
        if node1 in coordinates and node2 in coordinates:
            x1, y1 = coordinates[node1]
            x2, y2 = coordinates[node2]
            
            if (node1, node2) in polyline_edges or (node2, node1) in polyline_edges:
                key = (node1, node2) if (node1, node2) in polyline_edges else (node2, node1)
                segments = polyline_edges[key]['segments']
                offset = polyline_edges[key]['offset']
                xs, ys = create_polyline(x1, y1, x2, y2, segments, offset)
                edge_color = polyline_color
                line_style = dict(width=3, dash='dash')
            
            elif (node1, node2) in curve_edges or (node2, node1) in curve_edges:
                key = (node1, node2) if (node1, node2) in curve_edges else (node2, node1)
                control_offset = curve_edges[key]['control_offset']
                xs, ys = create_quadratic_bezier(x1, y1, x2, y2, control_offset)
                edge_color = curve_color
                line_style = dict(width=3, dash='dot')
            
            elif (node1, node2) in mixed_edges or (node2, node1) in mixed_edges:
                key = (node1, node2) if (node1, node2) in mixed_edges else (node2, node1)
                straight_length = mixed_edges[key]['straight_length']
                control_offset = mixed_edges[key]['control_offset']
                angle = mixed_edges[key].get('angle')
                curve_first = mixed_edges[key].get('curve_first', False)
                xs, ys = create_mixed_edge(x1, y1, x2, y2, straight_length, control_offset, angle, curve_first)
                edge_color = mixed_color
                line_style = dict(width=3, dash='longdash')
            
            else:
                xs, ys = [x1, x2], [y1, y2]
                edge_color = 'black'
                line_style = dict(width=2)
            
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode='lines',
                line=line_style,
                line_color=edge_color,
                name=f"Edge {node1}-{node2}",
                hoverinfo='text',
                hovertext=[f"Edge: {node1}-{node2}, Length: {length}"],
                showlegend=False
            ))
    
    # Update layout with custom background and grid
    fig.update_layout(
        title="Railyard Network",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            scaleanchor="y", 
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            scaleanchor="x"
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        width=600,
        height=800
    )

    return fig

# --------------------------------------------------

# Define the app layout
app.layout = html.Div(style={'margin': '20px'}, children=[
    html.H1('Interactive Railyard Visualization and Analysis Tool'),
    html.Div(style={'display': 'flex'}, children=[
        # Left panel for controls
        html.Div(style={'width': '25%', 'padding': '10px'}, children=[
            html.H2('Controls'),
            html.Label('Select Start Node:'),
            dcc.Dropdown(
                id='start-node-dropdown',
                options=[
                    {'label': f'Node {i}', 'value': i} for i in sorted(coordinates.keys())
                ],
                placeholder='Select Start Node'
            ),
            html.Br(),
            html.Label('Select End Node:'),
            dcc.Dropdown(
                id='end-node-dropdown',
                options=[
                    {'label': f'Node {i}', 'value': i} for i in sorted(coordinates.keys())
                ],
                placeholder='Select End Node'
            ),
            html.Br(),
            html.Label('Train Speed (km/h):'),
            dcc.Slider(
                id='train-speed-slider',
                min=0,
                max=120,
                step=5,
                value=60,
                marks={i: str(i) for i in range(0, 121, 20)}
            ),
            html.Div(id='train-speed-output', style={'marginTop': '20px'}),
            html.Br(),
            html.Label('Traffic Intensity (%):'),
            dcc.Slider(
                id='traffic-intensity-slider',
                min=0,
                max=100,
                step=5,
                value=50,
                marks={i: f'{i}%' for i in range(0, 101, 20)}
            ),
            html.Div(id='traffic-intensity-output', style={'marginTop': '20px'}),
            html.Br(),
            html.Label('Track Availability:'),
            dcc.Checklist(
                id='track-availability',
                options=[
                    {'label': f'Track {i}', 'value': i} for i in range(1, 6)
                ],
                value=[1, 2, 3, 4, 5],
                labelStyle={'display': 'block'}
            ),
            html.Br(),
            html.Button('Calculate Shortest Path', id='calculate-path-button', n_clicks=0),
            html.Br(),
            html.Button('Run Capacity Analysis', id='capacity-analysis-button', n_clicks=0),
            html.Br(),
            html.Button('Optimize Maintenance Schedule', id='maintenance-button', n_clicks=0),
        ]),
        # Right panel for graph and outputs
        html.Div(style={'width': '75%', 'padding': '10px'}, children=[
            html.H2('Railyard Visualization'),
            dcc.Graph(
                id='railyard-graph',
                figure=create_figure()  # Use the figure created by your code
            ),
            html.Div(id='analysis-results', style={'marginTop': '20px'})
        ])
    ])
])

# Placeholder callbacks (no functionality implemented yet)
@app.callback(
    Output('train-speed-output', 'children'),
    Input('train-speed-slider', 'value')
)
def update_train_speed_output(value):
    return f'Train Speed: {value} km/h'

@app.callback(
    Output('traffic-intensity-output', 'children'),
    Input('traffic-intensity-slider', 'value')
)
def update_traffic_intensity_output(value):
    return f'Traffic Intensity: {value}%'

# Run the Dash app
if __name__ == '__main__':
    app.run(debug=True)
