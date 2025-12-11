import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import bisect
from datetime import datetime

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# --------------------------------------------------
# Data Structures - Horizontally expanded coordinates
# --------------------------------------------------

switches = [11, 16, 12, 13, 52, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 112, 114, 116, 125, 123, 121, 129, 131, 132, 133, 134, 135, 152, 154, 156, 160, 162, 142, 193, 191, 200, 164]

# Expanded coordinates - stretched horizontally by 1.8x for better visibility
SCALE_X = 1.8
coordinates = {
    11: (0.0 * SCALE_X, 1502.3), 16: (0.0 * SCALE_X, 0),
    12: (829.82 * SCALE_X, 751.15), 13: (1907.09 * SCALE_X, 751.15),
    52: (2081.00 * SCALE_X, 751.15), 90: (3223.09 * SCALE_X, 300),
    91: (1600 * SCALE_X, 650), 92: (3550 * SCALE_X, 751.5),
    93: (3620 * SCALE_X, 1400), 94: (3550 * SCALE_X, 5000.0),
    95: (3200 * SCALE_X, 4700), 96: (3125 * SCALE_X, 4700),
    97: (3050 * SCALE_X, 4700), 98: (2800 * SCALE_X, 4900),
    99: (2580 * SCALE_X, 4900), 100: (2966 * SCALE_X, 1300),
    101: (2966 * SCALE_X, 1200), 102: (2163.05 * SCALE_X, 751.15),
    112: (2520.80 * SCALE_X, 751.15), 114: (2650 * SCALE_X, 650),
    116: (2900 * SCALE_X, 500.0), 121: (3352.29 * SCALE_X, 751.15),
    123: (3223.09 * SCALE_X, 650), 125: (2966.45 * SCALE_X, 650),
    129: (3550 * SCALE_X, 1500.0), 131: (3450 * SCALE_X, 1750),
    132: (3550 * SCALE_X, 2000.0), 133: (3450 * SCALE_X, 2115),
    134: (3450 * SCALE_X, 2250.0), 135: (3550 * SCALE_X, 2400.0),
    152: (3550 * SCALE_X, 3100), 154: (3450 * SCALE_X, 3300.0),
    156: (3350 * SCALE_X, 3500), 160: (3200 * SCALE_X, 3900.0), 162: (3050 * SCALE_X, 4200), 164: (3125 * SCALE_X, 4100),
    142: (2941.72 * SCALE_X, 3650), 193: (3450 * SCALE_X, 5200.0),
    191: (3350 * SCALE_X, 5400.0)
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
    (13, 91): {'segments': 2, 'offset': -50 * SCALE_X},
    (52, 90): {'segments': 2, 'offset': 240 * SCALE_X},
    (114, 116): {'segments': 2, 'offset': 90 * SCALE_X},
    (116, 123): {'segments': 2, 'offset': 90 * SCALE_X}
}

curve_edges = {(129, 93): {'control_offset': (1 * SCALE_X, 30)}}

mixed_edges = {
    (102, 129): {'straight_length': 250 * SCALE_X, 'control_offset': (1.5 * SCALE_X, -400), 'angle': np.radians(35), 'curve_first': False},
    (134, 142): {'straight_length': 300 * SCALE_X, 'control_offset': (1.5 * SCALE_X, 400), 'angle': np.radians(90), 'curve_first': False},
    (131, 100): {'straight_length': 150 * SCALE_X, 'control_offset': (1.5 * SCALE_X, -200), 'angle': np.radians(225), 'curve_first': False},
    (131, 101): {'straight_length': 200 * SCALE_X, 'control_offset': (1.5 * SCALE_X, -200), 'angle': np.radians(270), 'curve_first': False},
    (142, 98): {'straight_length': 450 * SCALE_X, 'control_offset': (1.5 * SCALE_X, -300), 'angle': np.radians(90), 'curve_first': True},
    (142, 99): {'straight_length': 600 * SCALE_X, 'control_offset': (1.5 * SCALE_X, -300), 'angle': np.radians(90), 'curve_first': True},
}

# Train configurations with emoji icons
TRAIN_COLORS = [
    {'color': '#e74c3c', 'name': 'Red Express', 'icon': '🚂'},
    {'color': '#3498db', 'name': 'Blue Cargo', 'icon': '🚃'},
    {'color': '#2ecc71', 'name': 'Green Freight', 'icon': '🚋'},
    {'color': '#f39c12', 'name': 'Orange Line', 'icon': '🚆'},
    {'color': '#9b59b6', 'name': 'Purple Metro', 'icon': '🚇'},
]

TRAIN_MISSIONS = [
    {'entry': 11, 'dest': 98, 'desc': 'Coil Loading Track 7', 'load_time': 50},
    {'entry': 16, 'dest': 99, 'desc': 'Coil Loading Track 8', 'load_time': 50},
    {'entry': 11, 'dest': 100, 'desc': 'Storage Track 102', 'load_time': 30},
    {'entry': 16, 'dest': 101, 'desc': 'Storage Track 101', 'load_time': 30},
    {'entry': 11, 'dest': 93, 'desc': 'Loco Lineup', 'load_time': 20},
]

TRACK_NAMES = {
    98: 'Coil Loading #7', 99: 'Coil Loading #8',
    100: 'Storage Track 102', 101: 'Storage Track 101',
    93: 'Loco Lineup', 90: 'Maintenance Bay',
    91: 'Sky Track 13', 92: 'End Track',
    97: 'Future Track 103', 95: 'Track 5', 96: 'Track 6',
    191: 'South Yard', 94: 'North Yard'
}

# --------------------------------------------------
# Geometry Functions
# --------------------------------------------------

def create_polyline(x1, y1, x2, y2, segments, offset):
    xs, ys = [x1], [y1]
    for i in range(1, segments):
        t = i / segments
        mid_x, mid_y = (1 - t) * x1 + t * x2, (1 - t) * y1 + t * y2
        dx, dy = x2 - x1, y2 - y1
        length = (dx ** 2 + dy ** 2) ** 0.5
        mid_x += (-dy / length) * offset * (-1)**i
        mid_y += (dx / length) * offset * (-1)**i
        xs.append(mid_x)
        ys.append(mid_y)
    xs.append(x2)
    ys.append(y2)
    return xs, ys

def create_quadratic_bezier(x1, y1, x2, y2, control_offset, num_points=100):
    cx, cy = (x1 + x2) / 2 + control_offset[0], (y1 + y2) / 2 + control_offset[1]
    t = np.linspace(0, 1, num_points)
    return (1 - t)**2 * x1 + 2 * (1 - t) * t * cx + t**2 * x2, (1 - t)**2 * y1 + 2 * (1 - t) * t * cy + t**2 * y2

def create_mixed_edge(x1, y1, x2, y2, straight_length, control_offset, angle=None, curve_first=False):
    if curve_first:
        ix = x2 - straight_length * np.cos(angle) if angle else x1
        iy = y2 - straight_length * np.sin(angle) if angle else y1
        bx, by = create_quadratic_bezier(x1, y1, ix, iy, control_offset)
        return list(bx) + [bx[-1], x2], list(by) + [by[-1], y2]
    else:
        if angle is None:
            r = straight_length / np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            sx, sy = x1 + r * (x2 - x1), y1 + r * (y2 - y1)
        else:
            sx, sy = x1 + straight_length * np.cos(angle), y1 + straight_length * np.sin(angle)
        bx, by = create_quadratic_bezier(sx, sy, x2, y2, control_offset)
        return [x1, sx] + list(bx)[1:], [y1, sy] + list(by)[1:]

def get_edge_path_points(node1, node2):
    if node1 not in coordinates or node2 not in coordinates:
        return None, None
    x1, y1 = coordinates[node1]
    x2, y2 = coordinates[node2]
    
    if (node1, node2) in polyline_edges or (node2, node1) in polyline_edges:
        key = (node1, node2) if (node1, node2) in polyline_edges else (node2, node1)
        offset = polyline_edges[key]['offset'] * (-1 if key != (node1, node2) else 1)
        return list(create_polyline(x1, y1, x2, y2, polyline_edges[key]['segments'], offset)[0]), list(create_polyline(x1, y1, x2, y2, polyline_edges[key]['segments'], offset)[1])
    elif (node1, node2) in curve_edges or (node2, node1) in curve_edges:
        key = (node1, node2) if (node1, node2) in curve_edges else (node2, node1)
        bx, by = create_quadratic_bezier(x1, y1, x2, y2, curve_edges[key]['control_offset'])
        return list(bx), list(by)
    elif (node1, node2) in mixed_edges or (node2, node1) in mixed_edges:
        key = (node1, node2) if (node1, node2) in mixed_edges else (node2, node1)
        me = mixed_edges[key]
        if key != (node1, node2):
            ox1, oy1, ox2, oy2 = *coordinates[key[0]], *coordinates[key[1]]
            xs, ys = create_mixed_edge(ox1, oy1, ox2, oy2, me['straight_length'], me['control_offset'], me.get('angle'), me.get('curve_first', False))
            return list(reversed(xs)), list(reversed(ys))
        return create_mixed_edge(x1, y1, x2, y2, me['straight_length'], me['control_offset'], me.get('angle'), me.get('curve_first', False))
    return np.linspace(x1, x2, 50).tolist(), np.linspace(y1, y2, 50).tolist()

def compute_cumulative_distances(xs, ys):
    d = [0.0]
    for i in range(1, len(xs)):
        d.append(d[-1] + np.sqrt((xs[i] - xs[i-1])**2 + (ys[i] - ys[i-1])**2))
    return d

def precompute_edge_paths():
    paths = {}
    for n1, n2, length in edges_with_lengths:
        for a, b in [(n1, n2), (n2, n1)]:
            if (a, b) in paths:
                continue
            xs, ys = get_edge_path_points(a, b)
            if xs:
                cd = compute_cumulative_distances(xs, ys)
                paths[(a, b)] = {'points': list(zip(xs, ys)), 'cumulative_dist': cd, 'total_length': cd[-1], 'edge_length': length}
    return paths

def get_position_on_edge(key, dist, paths):
    if key not in paths:
        return None, None
    p = paths[key]
    pts, cd = p['points'], p['cumulative_dist']
    if not pts:
        return None, None
    dist = max(0, min(dist, cd[-1]))
    i = max(0, min(bisect.bisect_right(cd, dist) - 1, len(pts) - 2))
    t = (dist - cd[i]) / (cd[i + 1] - cd[i]) if cd[i + 1] - cd[i] > 1e-9 else 0
    return pts[i][0] + t * (pts[i + 1][0] - pts[i][0]), pts[i][1] + t * (pts[i + 1][1] - pts[i][1])

def create_railyard_graph():
    G = nx.Graph()
    for n in coordinates:
        G.add_node(n, pos=coordinates[n])
    for n1, n2, l in edges_with_lengths:
        G.add_edge(n1, n2, weight=l)
    return G

def compute_route(start, end, graph):
    try:
        path = nx.shortest_path(graph, source=start, target=end, weight='weight')
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)], path
    except:
        return None, None

edge_paths = precompute_edge_paths()
railyard_graph = create_railyard_graph()

# --------------------------------------------------
# Figure Creation with Train Icons
# --------------------------------------------------

def create_figure(trains=None, highlighted_switches=None, all_route_edges=None):
    fig = go.Figure()
    
    trains = trains or []
    highlighted_switches = highlighted_switches or []
    all_route_edges = all_route_edges or []
    
    route_set = set()
    for e in all_route_edges:
        route_set.add(tuple(e) if isinstance(e, list) else e)
        route_set.add((e[1], e[0]))
    
    # Draw edges
    for n1, n2, length in edges_with_lengths:
        if n1 not in coordinates or n2 not in coordinates:
            continue
        x1, y1, x2, y2 = *coordinates[n1], *coordinates[n2]
        is_route = (n1, n2) in route_set
        
        if (n1, n2) in polyline_edges or (n2, n1) in polyline_edges:
            key = (n1, n2) if (n1, n2) in polyline_edges else (n2, n1)
            xs, ys = create_polyline(x1, y1, x2, y2, polyline_edges[key]['segments'], polyline_edges[key]['offset'])
            color, style = ('#9b59b6' if is_route else '#4a90d9'), dict(width=4 if is_route else 2, dash='dash')
        elif (n1, n2) in curve_edges or (n2, n1) in curve_edges:
            key = (n1, n2) if (n1, n2) in curve_edges else (n2, n1)
            xs, ys = create_quadratic_bezier(x1, y1, x2, y2, curve_edges[key]['control_offset'])
            color, style = ('#9b59b6' if is_route else '#f5a623'), dict(width=4 if is_route else 2, dash='dot')
        elif (n1, n2) in mixed_edges or (n2, n1) in mixed_edges:
            key = (n1, n2) if (n1, n2) in mixed_edges else (n2, n1)
            me = mixed_edges[key]
            xs, ys = create_mixed_edge(x1, y1, x2, y2, me['straight_length'], me['control_offset'], me.get('angle'), me.get('curve_first', False))
            color, style = ('#9b59b6' if is_route else '#7ed321'), dict(width=4 if is_route else 2, dash='longdash')
        else:
            xs, ys = [x1, x2], [y1, y2]
            color, style = ('#9b59b6' if is_route else '#6c757d'), dict(width=3 if is_route else 1.5)
        
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line={**style, 'color': color}, hoverinfo='skip', showlegend=False))
    
    # Draw nodes
    for sw in switches:
        if sw not in coordinates:
            continue
        x, y = coordinates[sw]
        is_hl = sw in highlighted_switches
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(size=12 if is_hl else 8, color='#e74c3c' if is_hl else '#e8f4f8', line=dict(width=2, color='#3498db')),
            text=[str(sw)], textposition='bottom center', textfont=dict(color='#ecf0f1', size=8),
            hoverinfo='text', hovertext=f"Switch {sw}", showlegend=False
        ))
    
    # Draw trains with text icons
    for train in trains:
        if train.get('position'):
            tx, ty = train['position']
            color_info = TRAIN_COLORS[train['id'] % len(TRAIN_COLORS)]
            icon = '📦' if train.get('phase') == 'loading' else color_info['icon']
            fig.add_trace(go.Scatter(
                x=[tx], y=[ty], mode='text+markers',
                text=[icon], textfont=dict(size=20),
                marker=dict(size=30, color=color_info['color'], opacity=0.3),
                hoverinfo='text', hovertext=color_info['name'], showlegend=False
            ))
    
    fig.update_layout(
        paper_bgcolor='#1a1a2e', plot_bgcolor='#1a1a2e',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y', scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=5, r=5, t=5, b=5), uirevision='constant', showlegend=False, height=950
    )
    return fig

# --------------------------------------------------
# Styles
# --------------------------------------------------

STYLES = {
    'container': {'display': 'flex', 'minHeight': '100vh', 'backgroundColor': '#0f0f1a', 'fontFamily': "'Segoe UI', sans-serif", 'color': '#ecf0f1'},
    'leftPanel': {'width': '280px', 'backgroundColor': '#16213e', 'padding': '15px', 'borderRight': '1px solid #2c3e50', 'overflowY': 'auto'},
    'centerPanel': {'flex': '1', 'backgroundColor': '#1a1a2e', 'display': 'flex', 'flexDirection': 'column', 'minWidth': '0'},
    'rightPanel': {'width': '300px', 'backgroundColor': '#16213e', 'padding': '15px', 'borderLeft': '1px solid #2c3e50', 'overflowY': 'auto'},
    'panelTitle': {'fontSize': '12px', 'fontWeight': '600', 'color': '#3498db', 'textTransform': 'uppercase', 'letterSpacing': '1px', 'marginBottom': '12px', 'paddingBottom': '8px', 'borderBottom': '1px solid #2c3e50'},
    'controlGroup': {'marginBottom': '15px'},
    'label': {'fontSize': '11px', 'color': '#95a5a6', 'marginBottom': '6px', 'display': 'block'},
    'button': {'width': '100%', 'padding': '10px', 'border': 'none', 'borderRadius': '6px', 'cursor': 'pointer', 'fontSize': '13px', 'fontWeight': '600', 'marginBottom': '8px'},
    'primaryBtn': {'background': 'linear-gradient(135deg, #2ecc71, #27ae60)', 'color': 'white'},
    'dangerBtn': {'background': 'linear-gradient(135deg, #e74c3c, #c0392b)', 'color': 'white'},
    'statsCard': {'backgroundColor': '#1a1a2e', 'borderRadius': '6px', 'padding': '10px', 'marginBottom': '8px', 'border': '1px solid #2c3e50', 'textAlign': 'center'},
    'statValue': {'fontSize': '18px', 'fontWeight': '700', 'color': '#3498db'},
    'statLabel': {'fontSize': '9px', 'color': '#7f8c8d', 'textTransform': 'uppercase'},
    'trainCard': {'backgroundColor': '#1a1a2e', 'borderRadius': '6px', 'padding': '10px', 'marginBottom': '8px', 'borderLeft': '3px solid #3498db'},
    'logEntry': {'padding': '6px 8px', 'backgroundColor': '#1a1a2e', 'borderRadius': '4px', 'marginBottom': '4px', 'borderLeft': '2px solid #3498db', 'fontSize': '10px'},
    'phaseTag': {'display': 'inline-block', 'padding': '2px 6px', 'borderRadius': '8px', 'fontSize': '9px', 'fontWeight': '600'},
}

PHASE_COLORS = {
    'waiting': '#95a5a6', 'departing': '#3498db', 'loading': '#9b59b6',
    'returning': '#1abc9c', 'completed': '#2ecc71',
}

# --------------------------------------------------
# Layout - No Header
# --------------------------------------------------

app.layout = html.Div(style=STYLES['container'], children=[
    dcc.Store(id='simulation-state', data={'trains': [], 'logs': [], 'running': False, 'tick': 0, 'total_distance': 0, 'sim_speed': 1}),
    dcc.Interval(id='animation-interval', interval=50, n_intervals=0, disabled=True),
    
    # Left Panel
    html.Div(style=STYLES['leftPanel'], children=[
        html.Div(style=STYLES['panelTitle'], children='Controls'),
        
        html.Div(style=STYLES['controlGroup'], children=[
            html.Label('Trains', style=STYLES['label']),
            dcc.Slider(id='train-count-slider', min=1, max=5, step=1, value=3,
                      marks={i: {'label': str(i), 'style': {'color': '#95a5a6', 'fontSize': '10px'}} for i in range(1, 6)})
        ]),
        
        html.Div(style=STYLES['controlGroup'], children=[
            html.Label('Train Speed (km/h)', style=STYLES['label']),
            dcc.Slider(id='speed-slider', min=20, max=200, step=10, value=100,
                      marks={i: {'label': str(i), 'style': {'color': '#95a5a6', 'fontSize': '10px'}} for i in [20, 100, 200]})
        ]),
        
        html.Div(style=STYLES['controlGroup'], children=[
            html.Label('Simulation Speed', style=STYLES['label']),
            dcc.Slider(id='sim-speed-slider', min=1, max=5, step=1, value=1,
                      marks={1: {'label': '1x', 'style': {'color': '#95a5a6', 'fontSize': '10px'}},
                             2: {'label': '2x', 'style': {'color': '#95a5a6', 'fontSize': '10px'}},
                             3: {'label': '3x', 'style': {'color': '#95a5a6', 'fontSize': '10px'}},
                             4: {'label': '4x', 'style': {'color': '#95a5a6', 'fontSize': '10px'}},
                             5: {'label': '5x', 'style': {'color': '#95a5a6', 'fontSize': '10px'}}})
        ]),
        
        html.Div(style={'marginTop': '15px'}, children=[
            html.Button('▶ Run', id='run-btn', n_clicks=0, style={**STYLES['button'], **STYLES['primaryBtn']}),
            html.Button('■ Stop', id='stop-btn', n_clicks=0, style={**STYLES['button'], **STYLES['dangerBtn']}),
        ]),
        
        html.Div(style={'marginTop': '20px'}, children=[
            html.Div(style=STYLES['panelTitle'], children='Train Status'),
            html.Div(id='train-list')
        ])
    ]),
    
    # Center Panel
    html.Div(style=STYLES['centerPanel'], children=[
        html.Div(id='sim-status', style={'padding': '8px 15px', 'backgroundColor': '#16213e', 'borderBottom': '1px solid #2c3e50', 'display': 'flex', 'alignItems': 'center', 'gap': '15px'}, children=[
            html.Span('●', style={'color': '#e74c3c', 'fontSize': '14px'}),
            html.Span('Idle', style={'color': '#95a5a6', 'fontSize': '12px'}),
        ]),
        dcc.Graph(id='railyard-graph', figure=create_figure(), style={'flex': '1'}, config={'displayModeBar': False, 'scrollZoom': True})
    ]),
    
    # Right Panel
    html.Div(style=STYLES['rightPanel'], children=[
        html.Div(style=STYLES['panelTitle'], children='Statistics'),
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '6px', 'marginBottom': '12px'}, children=[
            html.Div(style=STYLES['statsCard'], children=[html.Div(id='active-count', style=STYLES['statValue'], children='0'), html.Div('Active', style=STYLES['statLabel'])]),
            html.Div(style=STYLES['statsCard'], children=[html.Div(id='completed-count', style=STYLES['statValue'], children='0'), html.Div('Done', style=STYLES['statLabel'])]),
            html.Div(style=STYLES['statsCard'], children=[html.Div(id='loading-count', style=STYLES['statValue'], children='0'), html.Div('Loading', style=STYLES['statLabel'])]),
            html.Div(style=STYLES['statsCard'], children=[html.Div(id='distance-count', style=STYLES['statValue'], children='0'), html.Div('Meters', style=STYLES['statLabel'])]),
        ]),
        html.Div(style=STYLES['panelTitle'], children='Event Log'),
        html.Div(id='log-feed', style={'maxHeight': '500px', 'overflowY': 'auto'})
    ])
])

# --------------------------------------------------
# Callbacks
# --------------------------------------------------

@app.callback(
    [Output('simulation-state', 'data'), Output('animation-interval', 'disabled'), Output('animation-interval', 'interval')],
    [Input('run-btn', 'n_clicks'), Input('stop-btn', 'n_clicks')],
    [State('train-count-slider', 'value'), State('speed-slider', 'value'), State('sim-speed-slider', 'value'), State('simulation-state', 'data')],
    prevent_initial_call=True
)
def control_sim(run_c, stop_c, train_count, speed, sim_speed, state):
    from dash import ctx
    
    if ctx.triggered_id == 'stop-btn':
        state['trains'], state['running'], state['tick'], state['total_distance'] = [], False, 0, 0
        state['logs'] = [{'time': datetime.now().strftime('%H:%M:%S'), 'msg': '🛑 Stopped', 'type': 'system'}] + state.get('logs', [])[:30]
        return state, True, 50
    
    if ctx.triggered_id == 'run-btn':
        state['trains'], state['logs'], state['tick'], state['total_distance'] = [], [], 0, 0
        state['sim_speed'] = sim_speed
        
        for i, m in enumerate(TRAIN_MISSIONS[:train_count]):
            edges, nodes = compute_route(m['entry'], m['dest'], railyard_graph)
            if edges:
                total_len = sum(edge_paths.get(e, edge_paths.get((e[1], e[0]), {})).get('edge_length', 0) for e in edges)
                state['trains'].append({
                    'id': i, 'name': TRAIN_COLORS[i]['name'],
                    'route_edges': [list(e) for e in edges], 'route_nodes': nodes,
                    'current_edge_index': 0, 'distance_on_edge': 0.0, 'speed': speed,
                    'phase': 'waiting', 'start_delay': i * 20, 'load_timer': 0, 'load_time': m['load_time'],
                    'visited_switches': [], 'total_length': total_len, 'position': None,
                    'entry_node': m['entry'], 'dest_node': m['dest'], 'dest_name': m['desc'], 'progress': 0,
                })
        
        state['running'] = True
        state['logs'] = [{'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'🚀 Started {train_count} trains', 'type': 'system'}]
        interval = max(10, 50 // sim_speed)
        return state, False, interval
    
    return state, not state.get('running', False), 50


@app.callback(
    [Output('railyard-graph', 'figure'), Output('simulation-state', 'data', allow_duplicate=True),
     Output('train-list', 'children'), Output('active-count', 'children'), Output('completed-count', 'children'),
     Output('loading-count', 'children'), Output('distance-count', 'children'), Output('log-feed', 'children'), Output('sim-status', 'children')],
    Input('animation-interval', 'n_intervals'),
    [State('simulation-state', 'data'), State('speed-slider', 'value'), State('sim-speed-slider', 'value')],
    prevent_initial_call=True
)
def update_anim(n, state, speed, sim_speed):
    if not state or not state.get('trains'):
        return create_figure(), state, [], '0', '0', '0', '0', [], [html.Span('●', style={'color': '#e74c3c'}), html.Span(' Idle', style={'color': '#95a5a6', 'fontSize': '12px'})]
    
    state['tick'] += sim_speed
    all_routes, highlighted = [], []
    active, completed, loading = 0, 0, 0
    
    for train in state['trains']:
        train['speed'] = speed
        color = TRAIN_COLORS[train['id'] % len(TRAIN_COLORS)]
        
        if train['phase'] == 'waiting':
            if state['tick'] >= train['start_delay']:
                train['phase'] = 'departing'
                train['position'] = coordinates.get(train['entry_node'])
                train['visited_switches'] = [train['entry_node']]
                state['logs'] = [{'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'{color["icon"]} {color["name"]} departing', 'type': 'depart'}] + state.get('logs', [])[:25]
            continue
        
        if train['phase'] == 'completed':
            completed += 1
            continue
        
        if train['phase'] == 'loading':
            loading += 1
            train['load_timer'] += sim_speed
            if train['load_timer'] >= train['load_time']:
                ret_edges, ret_nodes = compute_route(train['dest_node'], train['entry_node'], railyard_graph)
                if ret_edges:
                    train['route_edges'], train['route_nodes'] = [list(e) for e in ret_edges], ret_nodes
                    train['current_edge_index'], train['distance_on_edge'] = 0, 0
                    train['phase'], train['visited_switches'] = 'returning', [train['dest_node']]
                    state['logs'] = [{'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'📦 {color["name"]} returning', 'type': 'return'}] + state.get('logs', [])[:25]
            continue
        
        active += 1
        edges = [tuple(e) for e in train['route_edges']]
        all_routes.extend(edges)
        idx, dist = train['current_edge_index'], train['distance_on_edge']
        delta = (speed / 3.6) * 0.05 * 25 * sim_speed
        
        if idx >= len(edges):
            if train['phase'] == 'departing':
                train['phase'], train['load_timer'], train['progress'] = 'loading', 0, 50
                train['position'] = coordinates.get(train['dest_node'])
                state['logs'] = [{'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'📍 {color["name"]} at {train["dest_name"]}', 'type': 'arrive'}] + state.get('logs', [])[:25]
            elif train['phase'] == 'returning':
                train['phase'], train['progress'] = 'completed', 100
                train['position'] = coordinates.get(train['entry_node'])
                state['logs'] = [{'time': datetime.now().strftime('%H:%M:%S'), 'msg': f'✅ {color["name"]} done', 'type': 'complete'}] + state.get('logs', [])[:25]
            continue
        
        edge = edges[idx]
        ed = edge_paths.get(edge) or edge_paths.get((edge[1], edge[0]))
        if not ed:
            train['current_edge_index'] += 1
            continue
        
        dist += delta
        state['total_distance'] = state.get('total_distance', 0) + delta
        
        if dist >= ed['total_length']:
            train['current_edge_index'] += 1
            train['distance_on_edge'] = 0
            dest = edge[1]
            if dest not in train['visited_switches']:
                train['visited_switches'].append(dest)
            highlighted.append(dest)
            train['progress'] = min(100, (train['current_edge_index'] / len(edges)) * (50 if train['phase'] == 'departing' else 100) + (50 if train['phase'] == 'returning' else 0))
        else:
            train['distance_on_edge'] = dist
            x, y = get_position_on_edge(edge, dist, edge_paths)
            if x is None:
                x, y = get_position_on_edge((edge[1], edge[0]), dist, edge_paths)
            if x:
                train['position'] = (x, y)
        
        highlighted.extend(train.get('visited_switches', []))
    
    # Build UI
    train_cards = []
    for t in state['trains']:
        c = TRAIN_COLORS[t['id'] % len(TRAIN_COLORS)]
        ph = t.get('phase', 'waiting')
        train_cards.append(html.Div(style={**STYLES['trainCard'], 'borderLeftColor': c['color']}, children=[
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}, children=[
                html.Span(f"{c['icon']} {c['name']}", style={'fontSize': '11px', 'color': c['color']}),
                html.Span(ph.upper(), style={**STYLES['phaseTag'], 'backgroundColor': PHASE_COLORS.get(ph, '#95a5a6'), 'color': 'white'})
            ]),
            html.Div(style={'backgroundColor': '#2c3e50', 'borderRadius': '3px', 'height': '4px', 'marginTop': '6px', 'overflow': 'hidden'},
                    children=[html.Div(style={'width': f'{t.get("progress", 0)}%', 'height': '100%', 'backgroundColor': c['color']})])
        ]))
    
    logs = [html.Div(style={**STYLES['logEntry'], 'borderLeftColor': {'depart': '#3498db', 'arrive': '#f39c12', 'return': '#1abc9c', 'complete': '#2ecc71'}.get(l.get('type'), '#3498db')},
                    children=[html.Span(l['time'], style={'color': '#7f8c8d', 'fontSize': '9px', 'marginRight': '6px'}), html.Span(l['msg'])]) for l in state.get('logs', [])[:12]]
    
    status_color = '#2ecc71' if active > 0 else ('#f39c12' if loading > 0 else '#e74c3c')
    status_text = f'{active} moving' if active > 0 else (f'{loading} loading' if loading > 0 else 'Idle')
    status = [html.Span('●', style={'color': status_color, 'fontSize': '14px'}), html.Span(f' {status_text} | {sim_speed}x', style={'color': '#95a5a6', 'fontSize': '12px'})]
    
    return create_figure(state['trains'], list(set(highlighted)), all_routes), state, train_cards, str(active), str(completed), str(loading), str(int(state.get('total_distance', 0))), logs, status


if __name__ == '__main__':
    app.run(debug=True)
