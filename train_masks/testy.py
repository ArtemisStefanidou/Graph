# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:28:47 2022

    print("--------------np.array-----------\n", a)
    print("--------------np.array-----------\n", type(a))
    print("--------------graph edges---------", graph)
    print("--------------graph type---------", type(graph))
    print("--------------graph edges type---------", type(graph.edges()))
    print("--------------graph edges data---------", graph.edges.data())
    print("--------------ps---------", graph)
    print("--------------graph---------", graph[s][e])
    print("--------------ps---------", ps)
    print("--------------ps[0] type---------", type(ps[0]))
    print("--ps list 1--", ps[:, 1])
    print("--ps list 0--", ps[:, 0])
    print("Total number of nodes: ", int(graph.number_of_nodes()))
    print("Total number of edges: ", int(graph.number_of_edges()))
    print("List of all nodes: ", list(graph.nodes()))
    print("List of all edges: ", list(graph.edges(data = True)))
    print("Degree for all nodes: ", dict(graph.degree()))
    print("Total number of self-loops: ", int(graph.number_of_selfloops()))
    print("List of all nodes with self-loops: ", list(graph.nodes_with_selfloops()))
    print("wkt_list : ", wkt_list)
    print("--------------nodes---------", ps)
    print("--------------nodes len---------", len(ps))
    print("----------------------------HERE-------------------", edges_npArray)
    print("----------------------------HERE-------------------", list_edges)
    print("----------------------------Gout.edges.data()-------------------", Gout.edges.data())
    print("node_loc_dic : ", node_loc_dic)
    print("edge_dic : ", edge_dic)
"""
def main():
    import libraryGraph as lib
    import networkx as nx
    from shapely import wkt

    # open and skeletonize
    #pathImage'SN3_roads_train_AOI_3_Paris_PS-MS_img148.tif'
    # pathImage = 'SN3_roads_train_AOI_2_Vegas_PS-MS_img2.tif'
    pathImage = "../SN3_roads_train_AOI_2_Vegas_PS-MS_img2.tif"
    img = lib.image(pathImage)
    graph = lib.skeletonizeImage(img)

    # ========#=========================================== 04 =================================================================== #
    nx.write_gpickle(graph, "graphtest.gpickle")
    pickled_graph = nx.read_gpickle("graphtest.gpickle")
    wkt_list = lib.G_to_wkt(pickled_graph, add_small=True, verbose=False, super_verbose=False)
    print("wkt_list : ", wkt_list)
    lib.plot_Graph(graph,img)


    # ====================================================== 05 =================================================================== #


    simplify_graph = True

    #G_projected = ox.project_graph(Gout)


    Gout = lib.wkt_to_graph(wkt_list)


    # ox.plot_graph(Gout)
    # lib.plot_simplify_Graph(Gout,img)
    lib.plot_Graph(Gout,img)

# ====================================================== 08 =================================================================== #
    # dar tutorial settings
    save_only_route_png = False  # True
    fig_height = 12
    fig_width = 12
    node_color = '#66ccff'  # light blue
    node_size = 0.4
    node_alpha = 0.6
    edge_color = '#bfefff'  # lightblue1
    edge_linewidth = 0.5
    edge_alpha = 0.6
    edge_color_key = 'inferred_speed_mph'
    orig_dest_node_size = 8 * node_size
    max_plots = 3
    shuffle = True
    invert_xaxis = False
    invert_yaxis = False

    node = list(Gout.nodes())[-1]
    print("Gout.nodes[node]['lat']",Gout.nodes[node]['lat'])
    if Gout.nodes[node]['lat'] < 0:
                print("Negative latitude, inverting yaxis for plotting")
                invert_yaxis = True

    for u, v, key, data in Gout.edges(keys=True, data=True):
        for attr_key in data:
            if (attr_key == 'geometry') and (type(data[attr_key]) == str):
                # print("update geometry...")
                data[attr_key] = wkt.loads(data[attr_key])
            elif (attr_key == 'geometry_pix') and (type(data[attr_key]) == str):
                data[attr_key] = wkt.loads(data[attr_key])
            else:
                continue
    # max out dpi at 3500
    dpi = int(np.min([3500, desired_dpi]))
    print("plot dpi:", dpi)

    plot_graph_pix(Gout, pathImage, fig_height=fig_height, fig_width=fig_width,
                   node_size=node_size, node_alpha=node_alpha, node_color=node_color,
                   edge_linewidth=edge_linewidth, edge_alpha=edge_alpha, edge_color=edge_color,
                   default_dpi=dpi,
                   edge_color_key=None,
                   show=False, save=True,
                   invert_yaxis=invert_yaxis,
                   invert_xaxis=invert_xaxis)
if __name__ == "__main__":
    main()


