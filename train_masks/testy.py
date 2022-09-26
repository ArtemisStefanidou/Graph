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
    import numpy as np
    import cv2
    import skimage.io

    # open and skeletonize

    pathImage = 'SN3_roads_train_AOI_2_Vegas_PS-MS_img2.tif'
    # pathImage = "../SN3_roads_train_AOI_2_Vegas_PS-MS_img2.tif"
    img = lib.image(pathImage)
    graph = lib.skeletonizeImage(img)

    # ========#=========================================== 04 =================================================================== #

    wkt_list = lib.G_to_wkt(graph, add_small=True, verbose=False, super_verbose=False)
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
    lib.plot_Graph_with_pixels(pathImage, Gout)
if __name__ == "__main__":
    main()


