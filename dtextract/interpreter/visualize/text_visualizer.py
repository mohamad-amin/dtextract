import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.offline import init_notebook_mode


def init_notebook_visualization():
    init_notebook_mode(connected=True)


def visualize_normal(feature_names, bias, prediction, explanation, class_names, feature_count=10):

    print(' '.join(feature_names[np.where(explanation.data != 0)]))
    print(''.join(['#'] * 50))

    lack_fig, have_fig, result_fig = visualize_without_text(bias, prediction, explanation, class_names, feature_count)

    if lack_fig is not None:
        lack_fig.update_layout(autosize=False, height=300)
        lack_fig.show()
    if have_fig is not None:
        have_fig.update_layout(autosize=False, height=300)
        have_fig.show()

    result_fig.show()


def _get_important_features(contributions_df, feature_count):

    nonzero_contributions = (contributions_df[1] != 0.0) & (contributions_df[2] != 0.0)
    
    lacking_cols = (contributions_df[0] == 0.0) & nonzero_contributions
    lacking_features = contributions_df[lacking_cols].iloc[:min(feature_count, len(lacking_cols)), :]
    lacking_features = lacking_features.ix[lacking_features[1].abs().sort_values(ascending=False).index]

    having_cols = (contributions_df[0] != 0.0) & nonzero_contributions
    having_features = contributions_df[having_cols].iloc[:min(feature_count, len(having_cols)), :]
    having_features = having_features.ix[having_features[1].abs().sort_values(ascending=False).index]

    return lacking_features, having_features


# Experimental, only two classes!
def visualize_without_text(bias, prediction, explanation, class_names, feature_count=10):

    lacking_features, having_features = _get_important_features(explanation.contrib_df, feature_count)

    lacking_figure = None
    if len(lacking_features) == 0:
        print('No contribution from non-existing words!')
    else:
        lacking_figure = _get_feature_contribution_figure(
            'Non-existing words contributions', lacking_features.index[::-1], lacking_features[1][::-1], class_names)

    having_figure = None
    if len(having_features) == 0:
        print('No contribution from existing words!')
    else:
        having_figure = _get_feature_contribution_figure(
            'Existing words contributions', having_features.index[::-1], having_features[1][::-1], class_names)

    contribution = lacking_features[1].append(having_features[1])
    result_figure = _get_result_figure(
        'Final result (Prediction = Bias + Contribution)', bias, contribution, prediction, class_names)

    return lacking_figure, having_figure, result_figure


def _get_feature_contribution_figure(plot_name, index, contributions, class_names):

    if len(index) == 1:
        index = ['', ' '] + index.tolist()
        contributions = pd.Series([0] * 2 + contributions.tolist(), index=index)
    elif len(index) == 2:
        index = [''] + index.tolist()
        contributions = pd.Series([0] + contributions.tolist(), index=index)

    colors = [('teal' if d >= 0 else 'orange') for d in contributions]
    texts = [str(index[i]) + ': ' + '{:0.4f}'.format(contributions[i]) for i in range(len(index))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=index, x=contributions, base=0,
        text=texts, textposition='auto',
        orientation='h', marker_color=colors,
        name='Contributions'
    ))

    fig.update_layout(annotations=[
        go.layout.Annotation(
            x=.5 * max(contributions), y=len(index) + 1,
            text=class_names[0], font=dict(size=18, color='teal'),
            showarrow=False
        ),
        go.layout.Annotation(
            x=.5 * min(contributions), y=len(index) + 1,
            text=class_names[1], font=dict(size=18, color='orange'),
            showarrow=False
        )
    ], shapes=[
        go.layout.Shape(
            type="line",
            x0=0, y0=-.4, x1=0, y1=len(index) - .6,
            line=dict(
                color="Black",
                width=2
            )
        )
    ], plot_bgcolor='rgba(0,0,0,0)', title='<b>' + plot_name + '<b/>', titlefont=dict(size=16))

    fig.update_yaxes(showticklabels=False)
    fig.update_xaxes(showticklabels=False)

    return fig


def _get_result_figure(plot_name, bias, contributions, prediction, class_names):
    zero_class = np.array([bias[0], float(contributions[contributions > 0].sum()), prediction[0]][::-1])
    first_class = -1 * np.array([bias[1], float(contributions[contributions < 0].abs().sum()), prediction[1]][::-1])
    zero_text = list(map(lambda x: '{:.4f}'.format(abs(x)), zero_class))
    first_text = list(map(lambda x: '{:.4f}'.format(abs(x)), first_class))

    names = ['Bias', 'Contribution', 'Prediction'][::-1]

    fig = go.Figure(data=[
        go.Bar(
            y=names, x=zero_class, base=0,
            orientation='h', marker=dict(color='teal'),
            text=zero_text, textposition='auto'
        ),
        go.Bar(
            y=names, x=first_class, base=0,
            orientation='h', marker=dict(color='orange'),
            text=first_text, textposition='auto'
        )
    ], layout=dict(autosize=False, width=500, height=350))

    fig.update_layout(annotations=[
        go.layout.Annotation(
            x=.3, y=3,
            text=class_names[0], font=dict(size=18, color='teal'),
            showarrow=False
        ),
        go.layout.Annotation(
            x=-.3, y=3,
            text=class_names[1], font=dict(size=18, color='orange'),
            showarrow=False
        )
    ], shapes=[
        go.layout.Shape(
            type="line",
            x0=0, y0=-.4, x1=0, y1=3 -.6,
            line=dict(
                color="Black",
                width=2
            )
        )
    ], plot_bgcolor='rgba(0,0,0,0)', showlegend=False, title='<b>' + plot_name + '<b/>', titlefont=dict(size=16))

    fig.update_yaxes(tickfont=dict(size=16))
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(barmode='stack')

    return fig
