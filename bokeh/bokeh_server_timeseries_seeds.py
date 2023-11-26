
import os
import numpy as np
import pickle

from bokeh.models import ColumnDataSource, Div
from bokeh.plotting import figure, curdoc
from bokeh.layouts import row, column

# embeddings = np.load("files/umap_embedding_spaced_20k.npy")
# curves = np.load("files/curves_spaced_20k.npy")
# txts = torch.load("files/txts_spaced_20k.pt")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(SCRIPT_DIR, "../files/projection6/")
MEAN_CURVE_PATH = os.path.join(SCRIPT_DIR, "../files/pythia-160m-mean-curve.npy")

steps = [0] + [2**i for i in range(10)] + list(range(1000, 144000, 1000))
embeddings = np.load(os.path.join(FILES_DIR, "projection.npy"))
curves = np.load(os.path.join(FILES_DIR, "curves.npy"))
with open(os.path.join(FILES_DIR, "tokens.pkl"), "rb") as f:
    tokens = pickle.load(f)
N = curves.shape[0]
mean_curve = np.load(MEAN_CURVE_PATH)
max_mean_curve = max(mean_curve)

x_scatter = embeddings[:, 0]
y_scatter = embeddings[:, 1]

# Assuming data1 is your scatter plot data and data2 is your line plot data
data1 = {'x_values': x_scatter, 'y_values': y_scatter, 'indices': list(range(N))}
data2 = {'x1234_values': [], 'y1234_values': [],
        'x1_values': [], 'y1_values': [],
        'x2_values': [], 'y2_values': [],
        'x3_values': [], 'y3_values': [],}

source1 = ColumnDataSource(data=data1)
source2 = ColumnDataSource(data=data2)

p1 = figure(width=400, height=400, tools="tap,pan,box_zoom,reset", title="Space of training curves (click on a point to see its curve)")
p1.circle('x_values', 'y_values', size=5, source=source1,
          color='red',
          nonselection_fill_color='grey',
          nonselection_fill_alpha=0.5,
          nonselection_line_color='grey',
          selection_fill_color='red',
          selection_fill_alpha=1.0,
          selection_line_color='red')

colors = ['#440154', '#31688e', '#35b779', '#fde725']
p2 = figure(width=400, height=400, title="Selected training curve", x_axis_type='log', y_range=(-0.1, 18))
p2.line('x1234_values', 'y1234_values', source=source2, line_width=3, color=colors[0], legend_label="token loss (seed 1234)")
p2.line('x1_values', 'y1_values', source=source2, line_width=3, color=colors[1], legend_label="token loss (seed 1)")
p2.line('x2_values', 'y2_values', source=source2, line_width=3, color=colors[2], legend_label="token loss (seed 2)")
p2.line('x3_values', 'y3_values', source=source2, line_width=3, color=colors[3], legend_label="token loss (seed 3)")
p2.line(steps, mean_curve, line_width=3, color="grey", line_dash="dashed", legend_label="token mean (seed 1234)")
p2.xaxis.axis_label = "steps"
p2.yaxis.axis_label = "cross-entropy loss (nats)"

p2.legend.location = "top_right"  # You can change to any standard location string e.g., 'top_right'
# p2.legend.title = 'Legend'
p2.legend.background_fill_alpha = 0.5  # Set the transparency of legend's background
# make legend smaller
p2.legend.label_text_font_size = "8pt"


# Create a widget for text display with a box around it
text = Div(text="Sample with context and next token (token to be predicted). Answer token highlighted in red.", 
           width=800, height=600,
        #    render_as_text=True,
           styles={'border': '2px solid black',
                   'background-color': 'white',
                   'font-size': '100%',
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
            'x1234_values': steps,
            'y1234_values': curves[selected_index][0:154],
            'x1_values': steps,
            'y1_values': curves[selected_index][154:154*2],
            'x2_values': steps,
            'y2_values': curves[selected_index][154*2:154*3],
            'x3_values': steps,
            'y3_values': curves[selected_index][154*3:],
        }
        max_token_curve = max(curves[selected_index])
        p2.y_range.end = max(max_token_curve, max_mean_curve) * 1.2
        text.text = tokens_to_html(tokens[selected_index])

source1.selected.on_change('indices', callback)

# set up layout
lo = column(row(p1, p2), text)

curdoc().add_root(lo)
