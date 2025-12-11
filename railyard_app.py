import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import bisect
import random
from datetime import datetime

# Initialize the Dash app with external stylesheets
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# --------------------------------------------------
# Data Structures
# --------------------------------------------------

switches = [11, 16, 12, 13, 52, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 112, 114, 116, 125, 123, 121, 129, 131, 132, 133, 134, 135, 152, 154, 156, 160, 162, 142, 193, 191, 200, 164]

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
    156: (3350, 3500), 160: (3200, 3900.0), 162: (3050, 4200), 164: (3125, 4100),
    142: (2941.72, 3650), 193: (3450, 5200.0),
    191: (3350, 5400.0), 164: (3125, 4000)
}

edges_with_lengths = [
    (11, 12, 468), (16, 12, 468), (12, 13, 1132), (13, 91, 112), (13, 52, 176),
    (52, 90, 2713), (52, 102, 89), (102, 112, 427), (112, 121, 1265), (112, 114, 49),
    (114, 116, 624), (116, 123, 555), (123, 121, 44), (121, 92, 75), (114, 125, 730),
    (125, 123, 447), (125, 116, 104), (102, 129, 1913), (129, 93, 135), (129, 132, 375),
    (132, 135, 171), (135, 134, 64), (135, 152, 69), (152, 94, 900), (94, 193, 67),
    (193, 191, 40), (152, 154, 40), (154, 193, 900), (154, 156, 40), (156, 160, 102),
    (164, 96, 250), (162, 97, 300), (160, 95, 290), (156, 191, 900), (132, 133, 100),
    (133, 134, 9), (133, 131, 280), (160, 164, 40), (164, 162, 100), (131, 100, 530),
    (131, 101, 535), (134, 142, 125), (142, 98, 1000), (142, 99, 1000),
]

polyline_edges = {
    (13, 91): {'segments': 2, 'offset': -50},
    (52, 90): {'segments': 2, 'offset': 240},
    (114, 116): {'segments': 2, 'offset': 90},
    (116, 123): {'segments': 2, 'offset': 90}
}

curve_edges = {
    (129, 93): {'control_offset': (1, 30)},
}

mixed_edges = {
    (102, 129): {'straight_length': 250, 'control_offset': (1.5, -400), 'angle': np.radians(35), 'curve_first': False},
    (134, 142): {'straight_length': 300, 'control_offset': (1.5, 400), 'angle': np.radians(90), 'curve_first': False},
    (131, 100): {'straight_length': 150, 'control_offset': (1.5, -200), 'angle': np.radians(225), 'curve_first': False},
    (131, 101): {'straight_length': 200, 'control_offset': (1.5, -200), 'angle': np.radians(270), 'curve_first': False},
    (142, 98): {'straight_length': 450, 'control_offset': (1.5, -300), 'angle': np.radians(90), 'curve_first': True},
    (142, 99): {'straight_length': 600, 'control_offset': (1.5, -300), 'angle': np.radians(90), 'curve_first': True},
}

# Train colors for multiple trains
TRAIN_COLORS = [
    {'color': '#e74c3c', 'name': 'Red Express'},
    {'color': '#3498db', 'name': 'Blue Line'},
    {'color': '#2ecc71', 'name': 'Green Cargo'},
    {'color': '#f39c12', 'name': 'Orange Freight'},
    {'color': '#9b59b6', 'name': 'Purple Metro'},
]

# Default simulation routes
DEFAULT_ROUTES = [
    (11, 99), (16, 93), (11, 191), (16, 97), (11, 92),
]

# --------------------------------------------------
# Geometry Functions
# --------------------------------------------------

def create_polyline(x1, y1, x2, y2, segments, offset):
    xs, ys = [x1], [y1]
    for i in range(1, segments):
        t = i / segments
        mid_x = (1 - t) * x1 + t * x2
        mid_y = (1 - t) * y1 + t * y2
        dx, dy = x2 - x1, y2 - y1
        length = (dx ** 2 + dy ** 2) ** 0.5
        perp_x, perp_y = -dy / length, dx / length
        mid_x += perp_x * offset * (-1)**i
        mid_y += perp_y * offset * (-1)**i
        xs.append(mid_x)
        ys.append(mid_y)
    xs.append(x2)
    ys.append(y2)
    return xs, ys


def create_quadratic_bezier(x1, y1, x2, y2, control_offset, num_points=100):
    control_x = (x1 + x2) / 2 + control_offset[0]
    control_y = (y1 + y2) / 2 + control_offset[1]
    t_values = np.linspace(0, 1, num_points)
    bezier_x = (1 - t_values)**2 * x1 + 2 * (1 - t_values) * t_values * control_x + t_values**2 * x2
    bezier_y = (1 - t_values)**2 * y1 + 2 * (1 - t_values) * t_values * control_y + t_values**2 * y2
    return bezier_x, bezier_y


def create_mixed_edge(x1, y1, x2, y2, straight_length, control_offset, angle=None, curve_first=False):
    if curve_first:
        intersect_x = x2 - straight_length * np.cos(angle) if angle else x1
        intersect_y = y2 - straight_length * np.sin(angle) if angle else y1
        bezier_xs, bezier_ys = create_quadratic_bezier(x1, y1, intersect_x, intersect_y, control_offset)
        xs = list(bezier_xs) + [bezier_xs[-1], x2]
        ys = list(bezier_ys) + [bezier_ys[-1], y2]
    else:
        if angle is None:
            ratio = straight_length / np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            straight_x = x1 + ratio * (x2 - x1)
            straight_y = y1 + ratio * (y2 - y1)
        else:
            straight_x = x1 + straight_length * np.cos(angle)
            straight_y = y1 + straight_length * np.sin(angle)
        bezier_xs, bezier_ys = create_quadratic_bezier(straight_x, straight_y, x2, y2, control_offset)
        xs = [x1, straight_x] + list(bezier_xs)[1:]
        ys = [y1, straight_y] + list(bezier_ys)[1:]
    return xs, ys


def get_edge_path_points(node1, node2):
    if node1 not in coordinates or node2 not in coordinates:
        return None, None
    x1, y1 = coordinates[node1]
    x2, y2 = coordinates[node2]
    
    if (node1, node2) in polyline_edges or (node2, node1) in polyline_edges:
        key = (node1, node2) if (node1, node2) in polyline_edges else (node2, node1)
        offset = polyline_edges[key]['offset'] * (-1 if key != (node1, node2) else 1)
        xs, ys = create_polyline(x1, y1, x2, y2, polyline_edges[key]['segments'], offset)
        return list(xs), list(ys)
    elif (node1, node2) in curve_edges or (node2, node1) in curve_edges:
        key = (node1, node2) if (node1, node2) in curve_edges else (node2, node1)
        xs, ys = create_quadratic_bezier(x1, y1, x2, y2, curve_edges[key]['control_offset'])
        return list(xs), list(ys)
    elif (node1, node2) in mixed_edges or (node2, node1) in mixed_edges:
        key = (node1, node2) if (node1, node2) in mixed_edges else (node2, node1)
        me = mixed_edges[key]
        if key != (node1, node2):
            ox1, oy1 = coordinates[key[0]]
            ox2, oy2 = coordinates[key[1]]
            xs, ys = create_mixed_edge(ox1, oy1, ox2, oy2, me['straight_length'], me['control_offset'], me.get('angle'), me.get('curve_first', False))
            return list(reversed(xs)), list(reversed(ys))
        xs, ys = create_mixed_edge(x1, y1, x2, y2, me['straight_length'], me['control_offset'], me.get('angle'), me.get('curve_first', False))
        return list(xs), list(ys)
    else:
        return np.linspace(x1, x2, 50).tolist(), np.linspace(y1, y2, 50).tolist()


def compute_cumulative_distances(xs, ys):
    distances = [0.0]
    for i in range(1, len(xs)):
        dist = np.sqrt((xs[i] - xs[i-1])**2 + (ys[i] - ys[i-1])**2)
        distances.append(distances[-1] + dist)
    return distances


def precompute_edge_paths():
    edge_paths = {}
    for node1, node2, length in edges_with_lengths:
        for n1, n2 in [(node1, node2), (node2, node1)]:
            if (n1, n2) in edge_paths:
                continue
            xs, ys = get_edge_path_points(n1, n2)
            if xs is None:
                continue
            cum_dist = compute_cumulative_distances(xs, ys)
            edge_paths[(n1, n2)] = {
                'points': list(zip(xs, ys)),
                'cumulative_dist': cum_dist,
                'total_length': cum_dist[-1] if cum_dist else 0,
                'edge_length': length
            }
    return edge_paths


def get_position_on_edge(edge_key, distance, edge_paths_dict):
    if edge_key not in edge_paths_dict:
        return None, None
    path_data = edge_paths_dict[edge_key]
    points, cum_dist = path_data['points'], path_data['cumulative_dist']
    if not points:
        return None, None
    distance = max(0, min(distance, cum_dist[-1]))
    idx = max(0, min(bisect.bisect_right(cum_dist, distance) - 1, len(points) - 2))
    d0, d1 = cum_dist[idx], cum_dist[idx + 1]
    t = (distance - d0) / (d1 - d0) if (d1 - d0) > 1e-9 else 0
    return points[idx][0] + t * (points[idx + 1][0] - points[idx][0]), points[idx][1] + t * (points[idx + 1][1] - points[idx][1])


def create_railyard_graph():
    G = nx.Graph()
    for node in coordinates.keys():
        G.add_node(node, pos=coordinates[node])
    for node1, node2, length in edges_with_lengths:
        G.add_edge(node1, node2, weight=length)
    return G


def compute_route(start_node, end_node, graph):
    try:
        path_nodes = nx.shortest_path(graph, source=start_node, target=end_node, weight='weight')
        return [(path_nodes[i], path_nodes[i + 1]) for i in range(len(path_nodes) - 1)], path_nodes
    except nx.NetworkXNoPath:
        return None, None


# Precompute at module load
edge_paths = precompute_edge_paths()
railyard_graph = create_railyard_graph()

# --------------------------------------------------
# Figure Creation
# --------------------------------------------------

def create_figure(trains=None, highlighted_switches=None, all_route_edges=None):
    """Create figure with multiple train support."""
    fig = go.Figure()
    
    # Colors
    polyline_color, curve_color, mixed_color = '#4a90d9', '#f5a623', '#7ed321'
    node_color, background_color = '#e8f4f8', '#1a1a2e'
    
    highlighted_switches = highlighted_switches or []
    all_route_edges = all_route_edges or []
    trains = trains or []
    
    # Build route set
    route_edge_set = set()
    for edge in all_route_edges:
        route_edge_set.add(tuple(edge))
        route_edge_set.add((edge[1], edge[0]))
    
    # Draw edges
    for node1, node2, length in edges_with_lengths:
        if node1 not in coordinates or node2 not in coordinates:
            continue
        x1, y1 = coordinates[node1]
        x2, y2 = coordinates[node2]
        is_route = (node1, node2) in route_edge_set
        
        if (node1, node2) in polyline_edges or (node2, node1) in polyline_edges:
            key = (node1, node2) if (node1, node2) in polyline_edges else (node2, node1)
            xs, ys = create_polyline(x1, y1, x2, y2, polyline_edges[key]['segments'], polyline_edges[key]['offset'])
            color = '#9b59b6' if is_route else polyline_color
            style = dict(width=4 if is_route else 2, dash='dash')
        elif (node1, node2) in curve_edges or (node2, node1) in curve_edges:
            key = (node1, node2) if (node1, node2) in curve_edges else (node2, node1)
            xs, ys = create_quadratic_bezier(x1, y1, x2, y2, curve_edges[key]['control_offset'])
            color = '#9b59b6' if is_route else curve_color
            style = dict(width=4 if is_route else 2, dash='dot')
        elif (node1, node2) in mixed_edges or (node2, node1) in mixed_edges:
            key = (node1, node2) if (node1, node2) in mixed_edges else (node2, node1)
            me = mixed_edges[key]
            xs, ys = create_mixed_edge(x1, y1, x2, y2, me['straight_length'], me['control_offset'], me.get('angle'), me.get('curve_first', False))
            color = '#9b59b6' if is_route else mixed_color
            style = dict(width=4 if is_route else 2, dash='longdash')
        else:
            xs, ys = [x1, x2], [y1, y2]
            color = '#9b59b6' if is_route else '#6c757d'
            style = dict(width=3 if is_route else 1.5)
        
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line={**style, 'color': color},
                                  hoverinfo='text', hovertext=f"Edge {node1}-{node2} ({length}m)", showlegend=False))
    
    # Draw nodes
    for switch in switches:
        if switch not in coordinates:
            continue
        x, y = coordinates[switch]
        is_hl = switch in highlighted_switches
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(size=14 if is_hl else 10, color='#e74c3c' if is_hl else node_color,
                       line=dict(width=2, color='#3498db'), symbol='circle'),
            text=[str(switch)], textposition='bottom center',
            textfont=dict(color='#ecf0f1', size=9), hoverinfo='text',
            hovertext=f"Switch {switch}", showlegend=False
        ))
    
    # Draw trains
    for i, train in enumerate(trains):
        if train.get('position'):
            tx, ty = train['position']
            color_info = TRAIN_COLORS[i % len(TRAIN_COLORS)]
            fig.add_trace(go.Scatter(
                x=[tx], y=[ty], mode='markers',
                marker=dict(size=18, color=color_info['color'], symbol='triangle-up',
                           line=dict(width=2, color='white')),
                name=color_info['name'], hoverinfo='text',
                hovertext=f"{color_info['name']}", showlegend=False
            ))
    
    fig.update_layout(
        paper_bgcolor=background_color, plot_bgcolor=background_color,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y', scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10),
        uirevision='constant', showlegend=False,
        height=900
    )
    return fig


# --------------------------------------------------
# Styles
# --------------------------------------------------

STYLES = {
    'container': {
        'display': 'flex', 'flexDirection': 'column', 'minHeight': '100vh',
        'backgroundColor': '#0f0f1a', 'fontFamily': "'Segoe UI', Roboto, sans-serif", 'color': '#ecf0f1'
    },
    'header': {
        'background': 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
        'padding': '15px 30px', 'borderBottom': '2px solid #3498db',
        'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between'
    },
    'title': {
        'margin': '0', 'fontSize': '24px', 'fontWeight': '600',
        'background': 'linear-gradient(90deg, #3498db, #2ecc71)',
        'WebkitBackgroundClip': 'text', 'WebkitTextFillColor': 'transparent'
    },
    'main': {
        'display': 'flex', 'flex': '1', 'gap': '0', 'padding': '0'
    },
    'leftPanel': {
        'width': '280px', 'backgroundColor': '#16213e', 'padding': '20px',
        'borderRight': '1px solid #2c3e50', 'overflowY': 'auto'
    },
    'centerPanel': {
        'flex': '1', 'backgroundColor': '#1a1a2e', 'display': 'flex',
        'flexDirection': 'column', 'minWidth': '0'
    },
    'rightPanel': {
        'width': '300px', 'backgroundColor': '#16213e', 'padding': '20px',
        'borderLeft': '1px solid #2c3e50', 'overflowY': 'auto'
    },
    'panelTitle': {
        'fontSize': '14px', 'fontWeight': '600', 'color': '#3498db',
        'textTransform': 'uppercase', 'letterSpacing': '1px',
        'marginBottom': '20px', 'paddingBottom': '10px',
        'borderBottom': '1px solid #2c3e50'
    },
    'controlGroup': {
        'marginBottom': '20px'
    },
    'label': {
        'fontSize': '12px', 'color': '#95a5a6', 'marginBottom': '8px',
        'display': 'block', 'fontWeight': '500'
    },
    'button': {
        'width': '100%', 'padding': '12px 20px', 'border': 'none',
        'borderRadius': '8px', 'cursor': 'pointer', 'fontSize': '14px',
        'fontWeight': '600', 'transition': 'all 0.3s ease',
        'marginBottom': '10px'
    },
    'primaryBtn': {
        'background': 'linear-gradient(135deg, #2ecc71 0%, #27ae60 100%)',
        'color': 'white'
    },
    'dangerBtn': {
        'background': 'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)',
        'color': 'white'
    },
    'secondaryBtn': {
        'background': 'linear-gradient(135deg, #3498db 0%, #2980b9 100%)',
        'color': 'white'
    },
    'statsCard': {
        'backgroundColor': '#1a1a2e', 'borderRadius': '8px', 'padding': '15px',
        'marginBottom': '15px', 'border': '1px solid #2c3e50'
    },
    'statValue': {
        'fontSize': '24px', 'fontWeight': '700', 'color': '#3498db'
    },
    'statLabel': {
        'fontSize': '11px', 'color': '#7f8c8d', 'textTransform': 'uppercase'
    },
    'logEntry': {
        'padding': '10px 12px', 'backgroundColor': '#1a1a2e', 'borderRadius': '6px',
        'marginBottom': '8px', 'borderLeft': '3px solid #3498db', 'fontSize': '12px'
    },
    'logTime': {
        'color': '#7f8c8d', 'fontSize': '10px', 'marginBottom': '4px'
    },
    'logMessage': {
        'color': '#ecf0f1'
    },
    'trainIndicator': {
        'display': 'flex', 'alignItems': 'center', 'padding': '10px',
        'backgroundColor': '#1a1a2e', 'borderRadius': '8px', 'marginBottom': '10px'
    },
    'trainDot': {
        'width': '12px', 'height': '12px', 'borderRadius': '50%', 'marginRight': '10px'
    }
}

# --------------------------------------------------
# Layout
# --------------------------------------------------

app.layout = html.Div(style=STYLES['container'], children=[
    # Hidden stores
    dcc.Store(id='simulation-state', data={'trains': [], 'logs': [], 'running': False}),
    dcc.Interval(id='animation-interval', interval=50, n_intervals=0, disabled=True),
    
    # Header
    html.Div(style=STYLES['header'], children=[
        html.H1('Railyard Simulation Control Center', style=STYLES['title']),
        html.Div(style={'display': 'flex', 'gap': '20px', 'alignItems': 'center'}, children=[
            html.Div(id='sim-status', children=[
                html.Span('●', style={'color': '#e74c3c', 'marginRight': '8px'}),
                html.Span('Idle', style={'color': '#95a5a6'})
            ])
        ])
    ]),
    
    # Main content
    html.Div(style=STYLES['main'], children=[
        # Left Panel - Controls
        html.Div(style=STYLES['leftPanel'], children=[
            html.Div(style=STYLES['panelTitle'], children='Simulation Controls'),
            
            # Train count
            html.Div(style=STYLES['controlGroup'], children=[
                html.Label('Number of Trains', style=STYLES['label']),
                dcc.Slider(id='train-count-slider', min=1, max=5, step=1, value=3,
                          marks={i: {'label': str(i), 'style': {'color': '#95a5a6'}} for i in range(1, 6)},
                          className='custom-slider')
            ]),
            
            # Speed control
            html.Div(style=STYLES['controlGroup'], children=[
                html.Label('Train Speed', style=STYLES['label']),
                dcc.Slider(id='speed-slider', min=10, max=200, step=10, value=80,
                          marks={i: {'label': f'{i}', 'style': {'color': '#95a5a6'}} for i in [10, 50, 100, 150, 200]},
                          className='custom-slider'),
                html.Div(id='speed-display', style={'textAlign': 'center', 'marginTop': '8px', 'color': '#3498db', 'fontSize': '14px'})
            ]),
            
            # Buttons
            html.Div(style={'marginTop': '30px'}, children=[
                html.Button('Run Simulation', id='run-btn', n_clicks=0,
                           style={**STYLES['button'], **STYLES['primaryBtn']}),
                html.Button('Stop All', id='stop-btn', n_clicks=0,
                           style={**STYLES['button'], **STYLES['dangerBtn']}),
                html.Button('Add Train', id='add-train-btn', n_clicks=0,
                           style={**STYLES['button'], **STYLES['secondaryBtn']}),
            ]),
            
            # Active trains
            html.Div(style={'marginTop': '30px'}, children=[
                html.Div(style=STYLES['panelTitle'], children='Active Trains'),
                html.Div(id='train-list')
            ])
        ]),
        
        # Center Panel - Visualization
        html.Div(style=STYLES['centerPanel'], children=[
            dcc.Graph(id='railyard-graph', figure=create_figure(), 
                     style={'height': '100%'}, config={'displayModeBar': False, 'scrollZoom': True})
        ]),
        
        # Right Panel - Updates
        html.Div(style=STYLES['rightPanel'], children=[
            html.Div(style=STYLES['panelTitle'], children='Live Updates'),
            
            # Stats
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '10px', 'marginBottom': '20px'}, children=[
                html.Div(style=STYLES['statsCard'], children=[
                    html.Div(id='active-trains-count', style=STYLES['statValue'], children='0'),
                    html.Div(style=STYLES['statLabel'], children='Active Trains')
                ]),
                html.Div(style=STYLES['statsCard'], children=[
                    html.Div(id='total-distance', style=STYLES['statValue'], children='0'),
                    html.Div(style=STYLES['statLabel'], children='Total Distance')
                ]),
            ]),
            
            # Log feed
            html.Div(style=STYLES['panelTitle'], children='Event Log'),
            html.Div(id='log-feed', style={'maxHeight': '500px', 'overflowY': 'auto'})
        ])
    ])
])

# --------------------------------------------------
# Callbacks
# --------------------------------------------------

@app.callback(
    Output('speed-display', 'children'),
    Input('speed-slider', 'value')
)
def update_speed_display(value):
    return f'{value} km/h'


@app.callback(
    [Output('simulation-state', 'data'),
     Output('animation-interval', 'disabled')],
    [Input('run-btn', 'n_clicks'),
     Input('stop-btn', 'n_clicks'),
     Input('add-train-btn', 'n_clicks')],
    [State('train-count-slider', 'value'),
     State('speed-slider', 'value'),
     State('simulation-state', 'data')],
    prevent_initial_call=True
)
def control_simulation(run_clicks, stop_clicks, add_clicks, train_count, speed, state):
    from dash import ctx
    triggered = ctx.triggered_id
    
    if triggered == 'stop-btn':
        state['trains'] = []
        state['running'] = False
        state['logs'] = [{'time': datetime.now().strftime('%H:%M:%S'), 'msg': 'Simulation stopped'}] + state.get('logs', [])[:20]
        return state, True
    
    if triggered == 'run-btn':
        # Create multiple trains with different routes
        state['trains'] = []
        used_routes = []
        for i in range(train_count):
            # Pick different routes for each train
            available = [r for r in DEFAULT_ROUTES if r not in used_routes]
            if not available:
                available = DEFAULT_ROUTES
            route = random.choice(available)
            used_routes.append(route)
            
            route_edges, route_nodes = compute_route(route[0], route[1], railyard_graph)
            if route_edges:
                total_len = sum(edge_paths.get(e, edge_paths.get((e[1], e[0]), {})).get('edge_length', 0) for e in route_edges)
                state['trains'].append({
                    'id': i,
                    'route_edges': [list(e) for e in route_edges],
                    'route_nodes': route_nodes,
                    'current_edge_index': 0,
                    'distance_on_edge': 0.0,
                    'speed': speed,
                    'status': 'moving',
                    'visited_switches': [route[0]],
                    'total_length': total_len,
                    'position': coordinates.get(route[0])
                })
        
        state['running'] = True
        state['logs'] = [{'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'Started {train_count} train(s)'}] + state.get('logs', [])[:20]
        return state, False
    
    if triggered == 'add-train-btn' and state.get('running'):
        route = random.choice(DEFAULT_ROUTES)
        route_edges, route_nodes = compute_route(route[0], route[1], railyard_graph)
        if route_edges:
            total_len = sum(edge_paths.get(e, edge_paths.get((e[1], e[0]), {})).get('edge_length', 0) for e in route_edges)
            new_id = len(state['trains'])
            state['trains'].append({
                'id': new_id,
                'route_edges': [list(e) for e in route_edges],
                'route_nodes': route_nodes,
                'current_edge_index': 0,
                'distance_on_edge': 0.0,
                'speed': speed,
                'status': 'moving',
                'visited_switches': [route[0]],
                'total_length': total_len,
                'position': coordinates.get(route[0])
            })
            state['logs'] = [{'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'Added {TRAIN_COLORS[new_id % len(TRAIN_COLORS)]["name"]}'}] + state.get('logs', [])[:20]
        return state, False
    
    return state, not state.get('running', False)


@app.callback(
    [Output('railyard-graph', 'figure'),
     Output('simulation-state', 'data', allow_duplicate=True),
     Output('train-list', 'children'),
     Output('active-trains-count', 'children'),
     Output('total-distance', 'children'),
     Output('log-feed', 'children'),
     Output('sim-status', 'children')],
    Input('animation-interval', 'n_intervals'),
    [State('simulation-state', 'data'),
     State('speed-slider', 'value')],
    prevent_initial_call=True
)
def update_animation(n_intervals, state, speed):
    if not state or not state.get('trains'):
        return create_figure(), state, [], '0', '0m', [], [
            html.Span('●', style={'color': '#e74c3c', 'marginRight': '8px'}),
            html.Span('Idle', style={'color': '#95a5a6'})
        ]
    
    all_route_edges = []
    highlighted = []
    active_count = 0
    total_dist = 0
    
    # Update each train - FIXED PHYSICS: much faster movement
    for train in state['trains']:
        if train.get('status') == 'stopped':
            continue
        
        active_count += 1
        train['speed'] = speed
        
        route_edges = [tuple(e) for e in train['route_edges']]
        all_route_edges.extend(route_edges)
        idx = train['current_edge_index']
        dist = train['distance_on_edge']
        
        # FIXED: Much higher delta for visible movement
        # At 100 km/h = 27.7 m/s, at 50ms interval = 1.38m per frame
        # Our coordinate units are roughly meters, so multiply by 20 for good visual speed
        delta = (speed / 3.6) * 0.05 * 25  # speed in m/s * interval_sec * scale factor
        
        if idx >= len(route_edges):
            train['status'] = 'stopped'
            final_node = route_edges[-1][1] if route_edges else train['route_nodes'][-1]
            train['position'] = coordinates.get(final_node)
            if {'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'{TRAIN_COLORS[train["id"] % len(TRAIN_COLORS)]["name"]} arrived'} not in state['logs'][:5]:
                state['logs'] = [{'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'{TRAIN_COLORS[train["id"] % len(TRAIN_COLORS)]["name"]} arrived at destination'}] + state.get('logs', [])[:20]
            continue
        
        current_edge = route_edges[idx]
        edge_data = edge_paths.get(current_edge) or edge_paths.get((current_edge[1], current_edge[0]))
        
        if not edge_data:
            train['current_edge_index'] += 1
            train['distance_on_edge'] = 0
            continue
        
        edge_len = edge_data['total_length']
        dist += delta
        total_dist += delta
        
        if dist >= edge_len:
            train['current_edge_index'] += 1
            train['distance_on_edge'] = 0
            dest = current_edge[1]
            if dest not in train['visited_switches']:
                train['visited_switches'].append(dest)
            highlighted.append(dest)
            
            if train['current_edge_index'] >= len(route_edges):
                train['status'] = 'stopped'
                train['position'] = coordinates.get(route_edges[-1][1])
        else:
            train['distance_on_edge'] = dist
            x, y = get_position_on_edge(current_edge, dist, edge_paths)
            if x is None:
                x, y = get_position_on_edge((current_edge[1], current_edge[0]), dist, edge_paths)
            if x is not None:
                train['position'] = (x, y)
        
        highlighted.extend(train.get('visited_switches', []))
    
    # Build train list UI
    train_list_items = []
    for i, train in enumerate(state['trains']):
        color = TRAIN_COLORS[i % len(TRAIN_COLORS)]
        status_color = '#2ecc71' if train.get('status') == 'moving' else '#e74c3c'
        train_list_items.append(
            html.Div(style=STYLES['trainIndicator'], children=[
                html.Div(style={**STYLES['trainDot'], 'backgroundColor': color['color']}),
                html.Div(children=[
                    html.Div(color['name'], style={'fontWeight': '600', 'fontSize': '12px'}),
                    html.Div(f"{'Moving' if train.get('status') == 'moving' else 'Arrived'}", 
                            style={'fontSize': '10px', 'color': status_color})
                ])
            ])
        )
    
    # Build log entries
    log_entries = [
        html.Div(style=STYLES['logEntry'], children=[
            html.Div(log['time'], style=STYLES['logTime']),
            html.Div(log['msg'], style=STYLES['logMessage'])
        ]) for log in state.get('logs', [])[:10]
    ]
    
    # Status indicator
    status = [
        html.Span('●', style={'color': '#2ecc71' if active_count > 0 else '#e74c3c', 'marginRight': '8px'}),
        html.Span(f'{active_count} Running' if active_count > 0 else 'Idle', style={'color': '#95a5a6'})
    ]
    
    fig = create_figure(trains=state['trains'], highlighted_switches=list(set(highlighted)), all_route_edges=all_route_edges)
    
    return fig, state, train_list_items, str(active_count), f'{int(total_dist)}m', log_entries, status


if __name__ == '__main__':
    app.run(debug=True)
