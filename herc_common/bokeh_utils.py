import networkx as nx
import numpy as np
import pandas as pd

from bokeh.io import export_svgs, show
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          MultiLine, Plot, Range1d, ResetTool,CustomJS,WheelZoomTool,PanTool,MultiSelect)
from bokeh.models.tools import HoverTool
from bokeh.palettes import Spectral4
from bokeh.plotting import figure, from_networkx, ColumnDataSource
from bokeh.layouts import column


def build_graph_plot(G, title=""):
    """ Return a Bokeh plot of the given networkx graph

    Parameters
    ----------
    G: :obj:`networkx.Graph`
        Networkx graph instance to be plotted.
    title: str
        Title of the final plot
    
    Returns
    -------
    :obj:`bokeh.models.plot`
        Bokeh plot of the graph.
    """

    plot = Plot(plot_width=600, plot_height=450,
                x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
    plot.title.text = title
    
    node_attrs = {}
    for node in G.nodes(data=True):
        node_color = Spectral4[node[1]['n']]
        node_attrs[node[0]] = node_color
    nx.set_node_attributes(G, node_attrs, "node_color")

    node_hover_tool = HoverTool(tooltips=[("Label", "@label"), ("n", "@n")])
    wheelZoom = WheelZoomTool()
    plot.add_tools(node_hover_tool,PanTool(),wheelZoom, ResetTool())
    plot.toolbar.active_scroll = wheelZoom

    graph_renderer = from_networkx(G,nx.spring_layout, k=0.3,iterations=200,scale=1, center=(0, 0))
    graph_renderer.node_renderer.glyph = Circle(size=15,fill_color="node_color")
    graph_renderer.edge_renderer.glyph = MultiLine(line_alpha=0.8, line_width=1)
    
    plot.renderers.append(graph_renderer)
    
    selectCallback = CustomJS(args = dict(graph_renderer=graph_renderer), code =
            """
            let new_data_nodes = Object.assign({},graph_renderer.node_renderer.data_source.data);
            new_data_nodes['node_color'] = {};
            let colors = ['#2b83ba','#ABDDA4','#fdae61'];
            let ns = cb_obj.value.reduce((acc,v)=>{
                if(v=='fullGraph'){
                   acc.push(0);
                   acc.push(1);
                   acc.push(2);
                }
               if(v=='n0')acc.push(0);
               if(v=='n1')acc.push(1);
               if(v=='n2')acc.push(2);
               return acc;
            },[])


            Object.keys(graph_renderer.node_renderer.data_source.data['node_color']).map((n,i)=>{
                new_data_nodes['node_color'][i]='transparent';
            })


             ns.map(n=>{
                Object.keys(graph_renderer.node_renderer.data_source.data['node_color']).map((g,i)=>{
                    if(graph_renderer.node_renderer.data_source.data['n'][i]==n){
                        new_data_nodes['node_color'][i]=colors[n];
                    }
                })
            })

            graph_renderer.node_renderer.data_source.data = new_data_nodes

            """)
    

    multi_select = MultiSelect(title="Option:", options=[("fullGraph", "Full Graph"),("n0", "Seed Nodes"), ("n1", "N1"), ("n2", "N2")])
    multi_select.js_on_change('value', selectCallback)

    return column(plot,multi_select)


class BokehHistogram():
    def __init__(self, color_fill, color_hover, fill_alpha=0.7,
                 height=600, width=600, bins=20):
        self.color_fill = color_fill
        self.color_hover = color_hover
        self.fill_alpha = fill_alpha
        self.height = height
        self.width = width
        self.bins = bins
        self.plot = None
    
    def load_plot(self, df, column, title, x_label, y_label, notebook_handle=False):
        hist, edges = np.histogram(df[column], bins=self.bins)
        hist_df = pd.DataFrame({column: hist,
                                "left": edges[:-1],
                                "right": edges[1:]})
        hist_df["interval"] = ["%d to %d" % (left, right) for left, 
                                right in zip(hist_df["left"], hist_df["right"])]
        self.plot = figure(plot_height=self.height, plot_width=self.width,
                      title=title, x_axis_label=x_label, y_axis_label=y_label)
        
        data_src = ColumnDataSource(hist_df)
        self.plot.quad(bottom=0, top=column, left="left", 
            right="right", source=data_src, fill_color=self.color_fill, 
            line_color="black", fill_alpha=self.fill_alpha,
            hover_fill_alpha=1.0, hover_fill_color=self.color_hover)

        hover = HoverTool(tooltips=[('Interval', '@interval'),
                                    ('Count', str("@" + column))])
        self.plot.add_tools(hover)
        show(self.plot, notebook_handle=notebook_handle)
    
    def save_plot(self, file_name):
        if self.plot is None:
            print("There is nothing to save. You must load a plot first...")
            return
        try:
            self.plot.output_backend = "svg"
            export_svgs(self.plot, filename=file_name)
        except Exception as e:
            print("There was an error exporting the plot. Please verify that both " 
                  + f"Selenium and Geckodriver are installed: {e}")
