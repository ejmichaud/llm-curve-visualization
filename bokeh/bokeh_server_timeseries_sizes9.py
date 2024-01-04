
import os
import numpy as np
import pickle

from bokeh.models import ColumnDataSource, Div
from bokeh.plotting import figure, curdoc
from bokeh.layouts import row, column

# embeddings = np.load("files/umap_embedding_spaced_20k.npy")
# curves = np.load("files/curves_spaced_20k.npy")
# txts = torch.load("files/txts_spaced_20k.pt")

models = [
    "pythia-14m",
    "pythia-31m",
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
]

# indices = 
# indicess = []
# for model in models:
#     indicess.append(np.load(f"../files/{model}-timeseries-slice-indices.npy"))

# means = []
# for model in models:

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(SCRIPT_DIR, "../files/projection-9models/")

# means = {}
# for model in models:
#     means[model] = np.load(os.path.join(SCRIPT_DIR, f"../files/{model}-mean-curve.npy"))

steps = [0] + [2**i for i in range(10)] + list(range(1000, 144000, 1000))
curves = np.load(os.path.join(FILES_DIR, "curves3.npy"))
embeddings = np.load(os.path.join(FILES_DIR, "projection3.npy"))
with open(os.path.join(FILES_DIR, "tokens3.pkl"), "rb") as f:
    tokens = pickle.load(f)

N = curves.shape[0]
# max_mean_curve = max([max(mean_curve) for mean_curve in means.values()])

x_scatter = embeddings[:, 0]
y_scatter = embeddings[:, 1]

# Assuming data1 is your scatter plot data and data2 is your line plot data
data1 = {'x_values': x_scatter, 'y_values': y_scatter, 'indices': list(range(N))}
data2 = {
    'x14m_values': [], 'y14m_values': [],
    'x31m_values': [], 'y31m_values': [],
    'x70m_values': [], 'y70m_values': [],
    'x160m_values': [], 'y160m_values': [],
    'x410m_values': [], 'y410m_values': [],
    'x1b_values': [], 'y1b_values': [],
    'x1.4b_values': [], 'y1.4b_values': [],
    'x2.8b_values': [], 'y2.8b_values': [],
    'x6.9b_values': [], 'y6.9b_values': [],
}

source1 = ColumnDataSource(data=data1)
source2 = ColumnDataSource(data=data2)

p1 = figure(width=400, height=300, tools="tap,pan,box_zoom,reset", title="Space of training curves (click on a point to see its curve)")
p1.circle('x_values', 'y_values', size=5, source=source1,
          color='red',
          nonselection_fill_color='grey',
          nonselection_fill_alpha=0.5,
          nonselection_line_color='grey',
          selection_fill_color='red',
          selection_fill_alpha=1.0,
          selection_line_color='red')

colors = ['#440154', '#482475', '#414487', '#355f8d', '#2a788e', '#21918c', '#22a884', '#44bf70', '#7ad151']
p2 = figure(width=800, height=400, title="Selected training curve", x_axis_type='log', y_range=(-0.1, 18))
p2.line('x14m_values', 'y14m_values', source=source2, line_width=3, color=colors[0], legend_label="14m")
p2.line('x31m_values', 'y31m_values', source=source2, line_width=3, color=colors[1], legend_label="31m")
p2.line('x70m_values', 'y70m_values', source=source2, line_width=3, color=colors[2], legend_label="70m")
p2.line('x160m_values', 'y160m_values', source=source2, line_width=3, color=colors[3], legend_label="160m")
p2.line('x410m_values', 'y410m_values', source=source2, line_width=3, color=colors[4], legend_label="410m")
p2.line('x1b_values', 'y1b_values', source=source2, line_width=3, color=colors[5], legend_label="1b")
p2.line('x1.4b_values', 'y1.4b_values', source=source2, line_width=3, color=colors[6], legend_label="1.4b")
p2.line('x2.8b_values', 'y2.8b_values', source=source2, line_width=3, color=colors[7], legend_label="2.8b")
p2.line('x6.9b_values', 'y6.9b_values', source=source2, line_width=3, color=colors[8], legend_label="6.9b")

# p2.line(steps, means["pythia-70m"], line_width=3, color="", legend_label="pythia-70m mean")
# plot mean loss with dashed line
# p2.line(steps, means["pythia-70m"], line_width=3, color=colors[0], line_dash="dashed", legend_label="pythia-70m mean")
# p2.line(steps, means["pythia-160m"], line_width=3, color=colors[1], line_dash="dashed", legend_label="pythia-160m mean")
# p2.line(steps, means["pythia-410m"], line_width=3, color=colors[2], line_dash="dashed", legend_label="pythia-410m mean")
p2.xaxis.axis_label = "steps"
p2.yaxis.axis_label = "cross-entropy loss (nats)"

# p2.legend.location = "top_right"  # You can change to any standard location string e.g., 'top_right'
p2.legend.location = "top_center"
p2.legend.orientation = "horizontal"
# p2.legend.title = 'Legend'
p2.legend.background_fill_alpha = 0.5  # Set the transparency of legend's background
# make legend smaller
p2.legend.label_text_font_size = "8pt"

# Create a widget for text display with a box around it
text = Div(text="Sample with context and next token (token to be predicted). Answer token highlighted in red.", 
           width=400, height=300,
        #    render_as_text=True,
           styles={'border': '2px solid black',
                   'background-color': 'white',
                   'font-size': '75%',
                   'font-family': 'monospace',
                   }
           )

newline_tokens = ['\n', '\r', '\r\n', '\v', '\f']
def tokens_to_html(tokens, max_len=150):
    html = ""
    txt = ""
    if len(tokens) > max_len:
        html += '<span>...</span>'
    tokens = tokens[-max_len:]
    for i, token in enumerate(tokens):
        background_color = "white" if i != len(tokens) - 1 else "#FF9999"
        txt += token
        if all([c in newline_tokens for c in token]):
            # replace all instances with ⏎
            token_rep = len(token) * "⏎"
            brs = "<br>" * len(token)
            html += f'<span style="border: 1px solid #DDD; background-color: {background_color}; white-space: pre-wrap;">{token_rep}</span>{brs}'
        else:
            html += f'<span style="border: 1px solid #DDD; background-color: {background_color}; white-space: pre-wrap;">{token}</span>'
    if "</" in txt:
        return "CONTEXT NOT LOADED FOR SECURITY REASONS SINCE IT CONTAINS HTML CODE (could contain javascript)."
    else:
        return html

def callback(attr, old, new):
    if new:
        selected_index = source1.data['indices'][new[0]]
        source2.data = {
            'x14m_values': steps,
            'y14m_values': curves[selected_index][0:154],
            'x31m_values': steps,
            'y31m_values': curves[selected_index][154:154*2],
            'x70m_values': steps,
            'y70m_values': curves[selected_index][154*2:154*3],
            'x160m_values': steps,
            'y160m_values': curves[selected_index][154*3:154*4],
            'x410m_values': steps,
            'y410m_values': curves[selected_index][154*4:154*5],
            'x1b_values': steps,
            'y1b_values': curves[selected_index][154*5:154*6],
            'x1.4b_values': steps,
            'y1.4b_values': curves[selected_index][154*6:154*7],
            'x2.8b_values': steps,
            'y2.8b_values': curves[selected_index][154*7:154*8],
            'x6.9b_values': steps,
            'y6.9b_values': curves[selected_index][154*8:],
        }
        max_token_curve = max(curves[selected_index])
        # p2.y_range.end = max(max_token_curve, max_mean_curve) * 1.2
        p2.y_range.end = max_token_curve * 1.2
        text.text = tokens_to_html(tokens[selected_index])

source1.selected.on_change('indices', callback)

# set up layout
lo = column(row(p1, text), p2)

curdoc().add_root(lo)
