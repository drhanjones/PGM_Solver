#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import plotly.express as px
import plotly.io as pio

# show plots in html
pio.renderers.default = "browser"


filepath2 = '<pathto avevaluespgm.csv>'

averaged_data = pd.read_csv(filepath2)

averaged_data=averaged_data.dropna()

avefig = px.scatter(averaged_data, y="node_size", x="ave time_taken", color='node_size', 
                                    labels={
                      'ave time_taken': "Average Time (s)",
                      'node_size': "Network Size (nodes)",
                      'color': "Number of Nodes"
                  },
)
avefig.update_layout(yaxis_range=[0,25])
avefig.update_traces(marker=dict(size=12,
                              line=dict(width=0.5,
                                        color='Black')),
                  selector=dict(mode='markers'))

avefig.show()


evidencevstime = px.scatter(averaged_data, y="evidence_size", x="ave time_taken", size='evidence_size', 
                                    labels={
                      'ave time_taken': "Average Time (s)",
                      'evidence_size': "Evidence Size (variables)",
                      'color': "Ordering heuristic"
                  },
)
evidencevstime.update_layout(yaxis_range=[0,25])
evidencevstime.update_traces(marker=dict(size=12,
                              line=dict(width=0.5,
                                        color='Black')),
                  selector=dict(mode='markers'))

evidencevstime.show()

evidencevstime = px.scatter(averaged_data, y="evidence_type", x="ave time_taken", size='evidence_size', color='evidence_size',color_continuous_scale='viridis',
                                    labels={
                      'ave time_taken': "Average Time (s)",
                      'evidence_type': "Evidence Size (variables)",
                      'color': "Ordering heuristic"
                  },
)


averaged_netsize25 = averaged_data.drop(averaged_data[averaged_data.node_size != 25 ].index)
averaged_netsize25.to_csv('averaged_netsize25.csv', index=False)

minfill = averaged_data.drop(averaged_data[averaged_data.ordering_heuristic != 'min_fill' ].index)
minfillplot = px.scatter(minfill, x='ave time_taken', y='evidence_type', color='ordering_heuristic',
                         labels={
                      'ave time_taken': "Average Time (s)",
                      'evidence_type': "Evidence Size (variables)",
                      'color': "Ordering heuristic"
                  },
)


minfillplot.show()
minfillplot.update_layout(xaxis_range=[0,6])

mindegree = averaged_data.drop(averaged_data[averaged_data.ordering_heuristic != 'min_degree' ].index)
mindegreeplot = px.scatter(mindegree, x='ave time_taken', y='evidence_type', color='ordering_heuristic',
                         labels={
                      'ave time_taken': "Average Time (s)",
                      'evidence_type': "Evidence Size (variables)",
                      'color': "Ordering heuristic"
                  },
)


mindegreeplot.show()
mindegreeplot.update_layout(xaxis_range=[0,6])

allorderingplot = px.scatter(averaged_data, x='ave time_taken', y='evidence_type', color='ordering_heuristic',
                         labels={
                      'ave time_taken': "Average Time (s)",
                      'evidence_type': "Evidence Size (variables)",
                      'color': "Ordering heuristic"
                  },
)


allorderingplot.show()
allorderingplot.update_layout(yaxis_range=[0,6])



orderingvssize = px.line(averaged_netsize25, y='ave time_taken', x='evidence_type', color='ordering_heuristic',
                         labels={
                      'ave time_taken': "Average Time (s)",
                      'evidence_type': "Evidence Size (variables)",
                      'color': "Ordering heuristic"
                  },
)


orderingvssize.show()

ordering = px.scatter(averaged_netsize25, y="evidence_type", x="ave time_taken", size='evidence_size', color='evidence_size',color_continuous_scale='viridis',
                                    labels={
                      'ave time_taken': "Average Time (s)",
                      'evidence_type': "Evidence Size (variables)",
                      'color': "Ordering heuristic"
                  },
)


####### working on non-averaged data

filepath = '<pathto-PGM_Results.csv>'

data = pd.read_csv (filepath)


fig = px.scatter(data, y="node_size", x="time_taken", color='node_size', 
                                    labels={
                      'time_taken': "Computation Time",
                      'node_size': "Graph Size (number of nodes)",
                      'color': "Node Size"
                  },
)
fig.update_layout(yaxis_range=[0,12])
fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='Black')),
                  selector=dict(mode='markers'))

fig.show()
# marginal_x="histogram"


netsize25 = data.drop(data[data.node_size != 25 ].index)


fig = px.scatter(netsize25, y="time_taken", x="evidence_type", color='evidence_size',
                                    labels={
                      'time_taken': "Computation Time",
                      'evidence_type': "Size of evidence",
                      'color': "Node Size"
                  },
)
fig.update_traces(marker=dict(size=12,
                              line=dict(width=0.5,
                                        color='Black')),
                  selector=dict(mode='markers'))

fig.show()


###### evidence size vs MPE time
netsize25.loc[netsize25.evidence_type == 'Lowest Evidence', 'evidence_type'] = 1
netsize25.loc[netsize25.evidence_type == 'Maximum Evidence', 'evidence_type'] = 100
netsize25.loc[netsize25.evidence_type == '25% Evidence', 'evidence_type'] = 25
netsize25.loc[netsize25.evidence_type == '50% Evidence', 'evidence_type'] = 50
netsize25.loc[netsize25.evidence_type == '75% Evidence', 'evidence_type'] = 75

newfigtry = px.scatter(netsize25, y="time_taken", x="evidence_type", color='evidence_size',trendline="lowess",
                                    labels={
                      'time_taken': "Computation Time",
                      'evidence_type': "Size of evidence",
                      'color': "Node Size"
                  },
)
newfigtry.update_traces(marker=dict(size=12,
                              line=dict(width=0.5,
                                        color='Black')),
                  selector=dict(mode='markers'))

newfigtry.show()
newfigtry.update_layout(yaxis_range=[0,12])

newfigtry = px.scatter(netsize25, x="evidence_type", y="time_taken", trendline="lowess",
                                    labels={
                      'time_taken': "Computation Time",
                      'evidence_type': "Size of evidence",
                      'color': "Node Size"
                  },
)
newfigtry.update_layout(yaxis_range=[0,12])
newfigtry.update_traces(marker=dict(size=4,
                              line=dict(width=0.5,
                                        color='LightGrey')),
                  selector=dict(mode='markers'))

newfigtry.show()

# reset netsize
netsize25 = data.drop(data[data.node_size != 25 ].index)

fig = px.scatter(netsize25, y="time_taken", x="evidence_type", color='evidence_size',
                                    labels={
                      'time_taken': "Computation Time",
                      'evidence_type': "Size of evidence",
                      'color': "Node Size"
                  },
)


netsize25_highevidence = netsize25.drop(netsize25[netsize25.evidence_type != 'Maximum Evidence' ].index)

netsize25_highevidence.to_csv('netsize25_highevidence.csv', index=False)

netsize25_lowevidence = netsize25.drop(netsize25[netsize25.evidence_type != 'Lowest Evidence' ].index)

fig = px.scatter(netsize25_lowevidence, x="ordering_heuristic", y="time_taken", trendline="lowess")
fig.show()

netsize25_lowevidence.to_csv('netsize25_lowevidence.csv', index=False)

fig = px.scatter(netsize25_lowevidence, x="time_taken", y="ordering_heuristic", color='evidence_size',marginal_x="histogram",trendline="ols",
                                    labels={
                      'time_taken': "Computation Time",
                      'ordering_heuristic': "Ordering Heuristic",
                  },
)
fig.update_traces(marker=dict(size=12,
                              line=dict(width=0.5,
                                        color='Black')),
                  selector=dict(mode='markers'))
fig.update_layout(yaxis_range=[0,12])


import plotly.graph_objects as go

fig = go.Figure(data=go.Scatter(x=netsize25["evidence_type"], y=netsize25["time_taken"]))
fig.show()

lineplt_netsize25 = px.line(netsize25, x="evidence_type", y="time_taken")
lineplt_netsize25.show()
lineplt_netsize25.add_trace(go.Scatter(x=netsize25["evidence_type"], y=netsize25["time_taken"], name="spline",
                    text=["tweak line smoothness<br>with 'smoothing' in line object"],
                    hoverinfo='text+name',
                    line_shape='spline'))



############ for only low evidence MPE ############

onlylowevidence = data.drop(data[data.evidence_type != 'Lowest Evidence' ].index)

###### for min degree ordering

onlylowevidence_and_mindegree = onlylowevidence.drop(onlylowevidence[onlylowevidence.ordering_heuristic != 'min_degree' ].index)

fig = px.scatter(onlylowevidence_and_mindegree, x="node_size", y="time_taken", color='ordering_heuristic', symbol='ordering_heuristic',
                                    labels={
                      'time_taken': "Computation Time",
                      'node_size': "Graph Size (number of nodes)",
                      'color': "Node Size"
                  },
)
fig.update_layout(yaxis_range=[0,12])
fig.update_traces(marker=dict(size=12,
                              line=dict(width=1,
                                        color='Grey')),
                  selector=dict(mode='markers'))

fig.show()

###### for min fill ordering

onlylowevidence_and_minfill = onlylowevidence.drop(onlylowevidence[onlylowevidence.ordering_heuristic != 'min_fill' ].index)
fig = px.scatter(onlylowevidence_and_minfill, x="node_size", y="time_taken", color='ordering_heuristic', symbol='ordering_heuristic',
                                    labels={
                      'time_taken': "Computation Time",
                      'node_size': "Graph Size (number of nodes)",
                      'color': "Node Size"
                  },
)
fig.update_layout(yaxis_range=[0,12])
fig.update_traces(marker=dict(size=12,
                              symbol="diamond",
                              color='Green',
                              line=dict(width=1,
                                        color='Grey')),
                  selector=dict(mode='markers'))

fig.show()

###### for random ordering

onlylowevidence_and_random = onlylowevidence.drop(onlylowevidence[onlylowevidence.ordering_heuristic != 'random' ].index)
fig = px.scatter(onlylowevidence_and_random, x="node_size", y="time_taken", color='ordering_heuristic', symbol='ordering_heuristic',
                                    labels={
                      'time_taken': "Computation Time",
                      'node_size': "Graph Size (number of nodes)",
                      'color': "Node Size"
                  },
)
fig.update_layout(yaxis_range=[0,12])
fig.update_traces(marker=dict(size=12,
                              symbol="square",
                              color='Yellow',
                              line=dict(width=1,
                                        color='Grey')),
                  selector=dict(mode='markers'))

fig.show()


############ for only high evidence MPE ############

onlyhighevidence = data.drop(data[data.evidence_type != 'Maximum Evidence' ].index)

###### for min degree ordering

onlyhighevidence_and_mindegree = onlyhighevidence.drop(onlyhighevidence[onlyhighevidence.ordering_heuristic != 'min_degree' ].index)

fig = px.scatter(onlyhighevidence_and_mindegree, x="node_size", y="time_taken", color='ordering_heuristic', symbol='ordering_heuristic',
                                    labels={
                      'time_taken': "Computation Time",
                      'node_size': "Graph Size (number of nodes)",
                      'color': "Node Size"
                  },
)
fig.update_layout(yaxis_range=[0,12])
fig.update_traces(marker=dict(size=12,
                              line=dict(width=1,
                                        color='Grey')),
                  selector=dict(mode='markers'))

fig.show()

########## Time vs Evidence #####


size= data['node_size']*4

fig = px.scatter(data, x="evidence_type", y="time_taken",  symbol='node_size', size=size, opacity=0.1,
                                    labels={
                      'time_taken': "Computation Time",
                      'evidence_type': "Size of Evidence Given",
                  },
)


opacity_entry= 1/(25/data['node_size'])

fig = px.scatter(data, x="evidence_type", y="time_taken", color='node_size', symbol='node_size', size=size, opacity=0.5,
                                    labels={
                      'time_taken': "Computation Time",
                      'evidence_type': "Size of Evidence Given",
                  },
)

fig.update_traces(marker=dict(
                    size=size,
                    sizemode='area',
                    sizeref=2.*max(size)/(40.**2),
                    sizemin=4
                ),
selector=dict(mode='markers'))

fig.update_layout(yaxis_range=[0,12])
fig.update_traces(marker=dict(size=size,
                              line=dict(width=1,
                                        color='Grey')),
                  selector=dict(mode='markers'))

fig.show()

