# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:28:47 2022

@author: USER
"""

from skimage.morphology import skeletonize
from skimage import data
import sknw
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, filters
import numpy as np
from skimage import io
from skimage.morphology import closing
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import networkx as nx
from itertools import tee
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict #, defaultdict

import shapely.wkt
import shapely.ops
from shapely.geometry import mapping, Point, LineString


linestring = "LINESTRING {}"

def line_points_dist(line1, pts):
    return np.cross(line1[1] - line1[0], pts - line1[0]) / np.linalg.norm(line1[1] - line1[0])

def get_angle(p0, p1=np.array([0, 0]), p2=None):
    """ compute angle (in degrees) for p0p1p2 corner
    Inputs:
        p0,p1,p2 - points in the form of [x,y]
    """
    if p2 is None:
        p2 = p1 + np.array([1, 0])
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def add_small_segments(G, terminal_points, terminal_lines,
                       dist1=24, dist2=80,
                       angle1=30, angle2=150,
                       verbose=False):
    '''Connect small, missing segments
    terminal points are the end of edges.  This function tries to pair small
    gaps in roads.  It will not try to connect a missed T-junction, as the
    crossroad will not have a terminal point'''

    print("Running add_small_segments()")
    try:
        node = G.node
    except:
        node = G.nodes
    # if verbose:
    #   print("node:", node)

    term = [node[t]['o'] for t in terminal_points]
    # print("term:", term)
    dists = squareform(pdist(term))
    possible = np.argwhere((dists > 0) & (dists < dist1))
    good_pairs = []
    for s, e in possible:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]

        if G.has_edge(s, e):
            continue
        good_pairs.append((s, e))

    possible2 = np.argwhere((dists > dist1) & (dists < dist2))
    for s, e in possible2:
        if s > e:
            continue
        s, e = terminal_points[s], terminal_points[e]
        if G.has_edge(s, e):
            continue
        l1 = terminal_lines[s]
        l2 = terminal_lines[e]
        d = line_points_dist(l1, l2[0])

        if abs(d) > dist1:
            continue
        angle = get_angle(l1[1] - l1[0], np.array((0, 0)), l2[1] - l2[0])
        if (-1 * angle1 < angle < angle1) or (angle < -1 * angle2) or (angle > angle2):
            good_pairs.append((s, e))

    if verbose:
        print("  good_pairs:", good_pairs)

    dists = {}
    for s, e in good_pairs:
        s_d, e_d = [G.nodes[s]['o'], G.nodes[e]['o']]
        dists[(s, e)] = np.linalg.norm(s_d - e_d)

    dists = OrderedDict(sorted(dists.items(), key=lambda x: x[1]))

    wkt = []
    added = set()
    good_coords = []
    for s, e in dists.keys():
        if s not in added and e not in added:
            added.add(s)
            added.add(e)
            s_d, e_d = G.nodes[s]['o'].astype(np.int32), G.nodes[e]['o'].astype(np.int32)
            line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in [s_d, e_d]]
            line = '(' + ", ".join(line_strings) + ')'
            wkt.append(linestring.format(line))
            good_coords.append((tuple(s_d), tuple(e_d)))
    return wkt, good_pairs, good_coords

def remove_sequential_duplicates(seq):
    # todo
    res = [seq[0]]
    for elem in seq[1:]:
        if elem == res[-1]:
            continue
        res.append(elem)
    return res

def remove_duplicate_segments(seq):
    seq = remove_sequential_duplicates(seq)
    segments = set()
    split_seg = []
    res = []
    for idx, (s, e) in enumerate(pairwise(seq)):
        if (s, e) not in segments and (e, s) not in segments:
            segments.add((s, e))
            segments.add((e, s))
        else:
            split_seg.append(idx + 1)
    for idx, v in enumerate(split_seg):
        if idx == 0:
            res.append(seq[:v])
        if idx == len(split_seg) - 1:
            res.append(seq[v:])
        else:
            s = seq[split_seg[idx - 1]:v]
            if len(s) > 1:
                res.append(s)
    if not len(split_seg):
        res.append(seq)
    return res


def add_direction_change_nodes(pts, s, e, s_coord, e_coord):
    # print("```````````pts````````````", pts)
    if len(pts) > 3:
        ps = pts.reshape(pts.shape[0], 1, 2).astype(np.int32)
        approx = 2
        ps = cv2.approxPolyDP(ps, approx, False)
        ps = np.squeeze(ps, 1)
        st_dist = np.linalg.norm(ps[0] - s_coord)
        en_dist = np.linalg.norm(ps[-1] - s_coord)
        if st_dist > en_dist:
            s, e = e, s
            s_coord, e_coord = e_coord, s_coord
        ps[0] = s_coord
        ps[-1] = e_coord
    else:
        ps = np.array([s_coord, e_coord], dtype=np.int32)
    return ps


def graph2lines(G):
    node_lines = []
    edges = list(G.edges())
    if len(edges) < 1:
        return []
    prev_e = edges[0][1]
    current_line = list(edges[0])
    added_edges = {edges[0]}
    for s, e in edges[1:]:
        if (s, e) in added_edges:
            continue
        if s == prev_e:
            current_line.append(e)
        else:
            node_lines.append(current_line)
            current_line = [s, e]
        added_edges.add((s, e))
        prev_e = e
    if current_line:
        node_lines.append(current_line)
    return node_lines


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def flatten(l):
    return [item for sublist in l for item in sublist]


def G_to_wkt(G, add_small=True, connect_crossroads=True,
             img_copy=None, debug=False, verbose=False, super_verbose=False):
    """Transform G to wkt"""

    # print("G:", G)
    if G == [linestring.format("EMPTY")] or type(G) == str:
        return [linestring.format("EMPTY")]

    node_lines = graph2lines(G)
    # if verbose:
    #    print("node_lines:", node_lines)

    if not node_lines:
        return [linestring.format("EMPTY")]
    try:
        node = G.node
    except:
        node = G.nodes
    # print("node:", node)
    deg = dict(G.degree())
    wkt = []
    terminal_points = [i for i, d in deg.items() if d == 1]

    # refine wkt
    if verbose:
        print("Refine wkt...")
    terminal_lines = {}
    vertices = []
    for i, w in enumerate(node_lines):
        if ((i % 10000) == 0) and (i > 0) and verbose:
            print("  ", i, "/", len(node_lines))
        coord_list = []
        additional_paths = []
        for s, e in pairwise(w):
            # print("----------------G[s][e][pts]-------------", G[s][e]['pts'])
            # print("type", type(G[s][e]))

            vals = flatten([[v] for v in G[s][e].values()])
            # print(type(vals))
            # print("----------------vals---------------", vals)

            for ix, val in enumerate(vals):

                # print("--------vals----------", vals)

                s_coord, e_coord = node[s]['o'], node[e]['o']
                # print("s_coord:", s_coord, "e_coord:", e_coord)
                # print(type(val))
                # print("-----------------------------val------------------", val)

                #######################
                pts = G[s][e]['pts']
                # print("--------G[s][e]['pts']----------", G[s][e]['pts'])
                #######################

                if s in terminal_points:
                    terminal_lines[s] = (s_coord, e_coord)
                if e in terminal_points:
                    terminal_lines[e] = (e_coord, s_coord)

                ps = add_direction_change_nodes(pts, s, e, s_coord, e_coord)

                if len(ps.shape) < 2 or len(ps) < 2:
                    continue

                if len(ps) == 2 and np.all(ps[0] == ps[1]):
                    continue

                line_strings = ["{1:.1f} {0:.1f}".format(*c.tolist()) for c in ps]
                if ix == 0:
                    coord_list.extend(line_strings)
                else:
                    additional_paths.append(line_strings)

                vertices.append(ps)

        if not len(coord_list):
            continue
        segments = remove_duplicate_segments(coord_list)
        # print("segments:", segments)
        # return

        for coord_list in segments:
            if len(coord_list) > 1:
                line = '(' + ", ".join(coord_list) + ')'
                wkt.append(linestring.format(line))
        for line_strings in additional_paths:
            line = ", ".join(line_strings)
            line_rev = ", ".join(reversed(line_strings))
            for s in wkt:
                if line in s or line_rev in s:
                    break
            else:
                wkt.append(linestring.format('(' + line + ')'))

    if add_small and len(terminal_points) > 1:
        small_segs, good_pairs, good_coords = add_small_segments(
            G, terminal_points, terminal_lines, verbose=verbose)
        # print("small_segs", small_segs)
        wkt.extend(small_segs)

    if debug:
        vertices = flatten(vertices)
        visualize(img_copy, G, vertices)

    if not wkt:
        return [linestring.format("EMPTY")]

    # return cross_segs
    return wkt


# open and skeletonize
pathImage = 'SN3_roads_train_AOI_3_Paris_PS-MS_img148.tif'
# pathImage = 'SN3_roads_train_AOI_3_Paris_PS-MS_img148_result.tif'
img = cv2.imread(pathImage, cv2.IMREAD_GRAYSCALE)
binary = img > filters.threshold_otsu(img)

ske = skeletonize(binary).astype(np.uint16)

graph = sknw.build_sknw(ske, ('iso' == False))

nx.write_gpickle(graph, "graphtest.gpickle")
pickled_graph = nx.read_gpickle("graphtest.gpickle")
wkt_list = G_to_wkt(pickled_graph, add_small=True, verbose=False, super_verbose=False)

plt.imshow(img, cmap='gray')

for (s, e) in graph.edges():
    # print("--------------ps---------", graph)
    ps = graph[s][e]['pts']

    plt.plot(ps[:, 1], ps[:, 0], 'green')

nodes = graph.nodes()

ps = np.array([nodes[i]['o'] for i in nodes])

plt.plot(ps[:, 1], ps[:, 0], 'r.')

# print("wkt_list : ", wkt_list)

# # title and show
plt.title('Build Graph')
plt.show()

# ====================================================== 05 =================================================================== #

def wkt_list_to_nodes_edges(wkt_list, node_iter=10000, edge_iter=10000):
    '''Convert wkt list to nodes and edges
    Make an edge between each node in linestring. Since one linestring
    may contain multiple edges, this is the safest approach'''
    
    node_loc_set = set()    # set of edge locations
    node_loc_dic = {}       # key = node idx, val = location
    node_loc_dic_rev = {}   # key = location, val = node idx
    edge_loc_set = set()    # set of edge locations
    edge_dic = {}           # edge properties
    
    for i,lstring in enumerate(wkt_list):
        # get lstring properties
        # print("lstring:", lstring)
        shape = shapely.wkt.loads(lstring)
        # print("shape:", shape)
        xs, ys = shape.coords.xy
        length_orig = shape.length
        
        # iterate through coords in line to create edges between every point
        for j,(x,y) in enumerate(zip(xs, ys)):
            loc = (x,y)
            # for first item just make node, not edge
            if j == 0:
                # if not yet seen, create new node
                if loc not in node_loc_set:
                    node_loc_set.add(loc)
                    node_loc_dic[node_iter] = loc
                    node_loc_dic_rev[loc] = node_iter
                    node = node_iter
                    node_iter += 1
                    
            # if not first node in edge, retrieve previous node and build edge
            else:
                prev_loc = (xs[j-1], ys[j-1])
                #print ("prev_loc:", prev_loc)
                prev_node = node_loc_dic_rev[prev_loc]

                # if new, create new node
                if loc not in node_loc_set:
                    node_loc_set.add(loc)
                    node_loc_dic[node_iter] = loc
                    node_loc_dic_rev[loc] = node_iter
                    node = node_iter
                    node_iter += 1
                # if seen before, retrieve node properties
                else:
                    node = node_loc_dic_rev[loc]

                # add edge, which is start_node to end_node
                edge_loc = (loc, prev_loc)
                edge_loc_rev = (prev_loc, loc)
                # shouldn't be duplicate edges, so break if we see one
                if (edge_loc in edge_loc_set) or (edge_loc_rev in edge_loc_set):
                    # print ("Oops, edge already seen, returning:", edge_loc)
                    return
                
                # get distance to prev_loc and current loc
                proj_prev = shape.project(Point(prev_loc))
                proj = shape.project(Point(loc))
                # edge length is the diffence of the two projected lengths
                #   along the linestring
                edge_length = abs(proj - proj_prev)
                # make linestring
                line_out = LineString([prev_loc, loc])
                line_out_wkt = line_out.wkt
                
                edge_props = {'start': prev_node,
                              'start_loc_pix': prev_loc,
                              'end': node,
                              'end_loc_pix': loc,
                              'length_pix': edge_length,
                              'wkt_pix': line_out_wkt,
                              'geometry_pix': line_out,
                              'osmid': i}
                #print ("edge_props", edge_props)
                
                edge_loc_set.add(edge_loc)
                edge_dic[edge_iter] = edge_props
                edge_iter += 1

    return node_loc_dic, edge_dic

def nodes_edges_to_G(node_loc_dic, edge_dic, name='glurp'):
    '''Take output of wkt_list_to_nodes_edges(wkt_list) and create networkx 
    graph'''
    
    G = nx.MultiDiGraph()
    # set graph crs and name
    G.graph = {'name': name,
               'crs': {'init': 'epsg:4326'}
               }
    
    # add nodes
    #for key,val in node_loc_dic.iteritems():
    for key in node_loc_dic.keys():
        val = node_loc_dic[key]
        attr_dict = {'osmid': key,
                     'x_pix': val[0],
                     'y_pix': val[1]}
        G.add_node(key, **attr_dict)
    
    # add edges
    #for key,val in edge_dic.iteritems():
    for key in edge_dic.keys():
        val = edge_dic[key]
        attr_dict = val
        u = attr_dict['start']
        v = attr_dict['end']
        #attr_dict['osmid'] = str(i)
        
        #print ("nodes_edges_to_G:", u, v, "attr_dict:", attr_dict)
        if type(attr_dict['start_loc_pix']) == list:
            return
        
        G.add_edge(u, v, **attr_dict)
            
    G2 = G.to_undirected()
    
    return G2

def clean_sub_graphs(G_, min_length=300, max_nodes_to_skip=20,
                      weight='length_pix', verbose=True,
                      super_verbose=False):
    '''Remove subgraphs with a max path length less than min_length,
    if the subgraph has more than max_noxes_to_skip, don't check length 
        (this step great improves processing time)'''
    
    if len(G_.nodes()) == 0:
        return G_
    
    if verbose:
         print("Running clean_sub_graphs...")
    try:
        sub_graphs = list(nx.connected_component_subgraphs(G_))
    except:
        sub_graph_nodes = nx.connected_components(G_)
        sub_graphs = [G_.subgraph(c).copy() for c in sub_graph_nodes]
    
    if verbose:
        print("  sub_graph node count:", [len(z.nodes) for z in sub_graphs])
        #print("  sub_graphs:", [z.nodes for z in sub_graphs])
        
    bad_nodes = []
    if verbose:
        print("  len(G_.nodes()):", len(G_.nodes()) )
        # print("  len(G_.edges()):", len(G_.edges()) )
    if super_verbose:
        # print("G_.nodes:", G_.nodes())
        edge_tmp = G_.edges()[np.random.randint(len(G_.edges()))]
        print(edge_tmp, "G.edge props:", G_.edge[edge_tmp[0]][edge_tmp[1]])

    for G_sub in sub_graphs:
        # don't check length if too many nodes in subgraph
        if len(G_sub.nodes()) > max_nodes_to_skip:
            continue
        
        else:
            all_lengths = dict(nx.all_pairs_dijkstra_path_length(G_sub, weight=weight))
            if super_verbose:
                print("  \nGs.nodes:", G_sub.nodes() )
                # print("  all_lengths:", all_lengths )
            # get all lenghts
            lens = []
            #for u,v in all_lengths.iteritems():
            for u in all_lengths.keys():
                v = all_lengths[u]
                #for uprime, vprime in v.iteritems():
                for uprime in v.keys():
                    vprime = v[uprime]
                    lens.append(vprime)
                    if super_verbose:
                        print("  u, v", u,v )
                        # print("    uprime, vprime:", uprime, vprime )
            max_len = np.max(lens)
            if super_verbose:
                print("  Max length of path:", max_len)
            if max_len < min_length:
                bad_nodes.extend(G_sub.nodes())
                if super_verbose:
                    print(" appending to bad_nodes:", G_sub.nodes())

    # remove bad_nodes
    G_.remove_nodes_from(bad_nodes)
    if verbose:
        print(" num bad_nodes:", len(bad_nodes))
        # #print("bad_nodes:", bad_nodes)
        # print(" len(G'.nodes()):", len(G_.nodes()))
        # print(" len(G'.edges()):", len(G_.edges()))
    if super_verbose:
        print("  G_.nodes:", G_.nodes())
        
    return G_

node_loc_dic, edge_dic = wkt_list_to_nodes_edges(wkt_list, node_iter=10000,edge_iter=10000)

# print("----------------------node_loc_dic : ---------------------", node_loc_dic)
# print("--------------------------------edge_dic : ----------------------------", edge_dic.values())
# print("--------------------------------type edge_dic : ----------------------------", type(edge_dic))


G0 = nodes_edges_to_G(node_loc_dic, edge_dic)  

G1 = clean_sub_graphs(G0, min_length=20, weight='length_pix', verbose=True, super_verbose=False)

Gout = G0

plt.imshow(img, cmap='gray')

for (s, e) in Gout.edges():

    ps = Gout[s][e]

    print("--------------ps---------", ps)
    plt.plot(ps[:, 1], ps[:, 0], 'green')

nodes = Gout.nodes()

ps = np.array([nodes[i]['o'] for i in nodes])

plt.plot(ps[:, 1], ps[:, 0], 'r.')

# print("wkt_list : ", wkt_list)

# # title and show
plt.title('Build Graph')
plt.show()
#
# print("------------Gout---------\n", Gout)
# simplify_graph = True
#
# for value in edge_dic.values():
#     print("--------------value---------", value['wkt_pix'])
#
# '----------------------------------------------------------------------------------------'
# '----------------------------------------------------------------------------------------'
# '----------------------------------------------------------------------------------------'
# '----------------------------------------------------------------------------------------'
#
#
# def wkt_to_G(params):
#     '''Execute all functions'''
#
#     n_threads_max = 12
#
#     wkt_list, im_file, min_subgraph_length_pix, \
#     node_iter, edge_iter, \
#     min_spur_length_m, simplify_graph, \
#     rdp_epsilon, \
#     manually_reproject_nodes, \
#     out_file, graph_dir, n_threads, verbose \
#         = params
#
#     print("im_file:", im_file)
#     # print("wkt_list:", wkt_list)
#     pickle_protocol = 4
#
#     t0 = time.time()
#     if verbose:
#         print("Running wkt_list_to_nodes_edges()...")
#     node_loc_dic, edge_dic = wkt_list_to_nodes_edges(wkt_list,
#                                                      node_iter=node_iter,
#                                                      edge_iter=edge_iter)
#     t1 = time.time()
#     if verbose:
#         print("Time to run wkt_list_to_nodes_egdes():", t1 - t0, "seconds")
#
#     # print ("node_loc_dic:", node_loc_dic)
#     # print ("edge_dic:", edge_dic)
#
#     if verbose:
#         print("Creating G...")
#     G0 = nodes_edges_to_G(node_loc_dic, edge_dic)
#     if verbose:
#         print("  len(G.nodes():", len(G0.nodes()))
#         print("  len(G.edges():", len(G0.edges()))
#     # for edge_tmp in G0.edges():
#     #    print ("\n 0 wtk_to_G():", edge_tmp, G0.edge[edge_tmp[0]][edge_tmp[1]])
#
#     t2 = time.time()
#     if verbose:
#         print("Time to run nodes_edges_to_G():", t2 - t1, "seconds")
#
#     # This graph will have a unique edge for each line segment, meaning that
#     #  many nodes will have degree 2 and be in the middle of a long edge.
#
#     # run clean_sub_graph() in 04_skeletonize.py?  - Nope, do it here
#     # so that adding small terminals works better...
#     if verbose:
#         print("Clean out short subgraphs")
#     G1 = clean_sub_graphs(G0, min_length=min_subgraph_length_pix,
#                           weight='length_pix', verbose=verbose,
#                           super_verbose=False)
#     t3 = time.time()
#     if verbose:
#         print("Time to run clean_sub_graphs():", t3 - t2, "seconds")
#     t3 = time.time()
#     # G1 = G0
#
#     if len(G1) == 0:
#         return G1
#
#     # print ("Simplifying graph")
#     # G0 = ox.simplify_graph(G0.to_directed())
#     # G0 = G0.to_undirected()
#     # #G0 = ox.project_graph(G0)
#     # #G_p_init = create_edge_linestrings(G_p_init, remove_redundant=True, verbose=False)
#     # t3 = time.time()
#     # print ("  len(G.nodes():", len(G0.nodes()))
#     # print ("  len(G.edges():", len(G0.edges()))
#     # print ("Time to run simplify graph:", t30 - t3, "seconds")
#
#     # for edge_tmp in G0.edges():
#     #    print ("\n 1 wtk_to_G():", edge_tmp, G0.edge[edge_tmp[0]][edge_tmp[1]])
#
#     # edge_tmp = G0.edges()[5]
#     # print (edge_tmp, "G0.edge props:", G0.edge[edge_tmp[0]][edge_tmp[1]])
#
#     # geo coords
#     if im_file:
#         if verbose:
#             print("Running get_node_geo_coords()...")
#         # let's not over multi-thread a multi-thread
#         if n_threads > 1:
#             n_threads_tmp = 1
#         else:
#             n_threads_tmp = n_threads_max
#         G1 = get_node_geo_coords(G1, im_file, n_threads=n_threads_tmp,
#                                  verbose=verbose)
#         t4 = time.time()
#         if verbose:
#             print("Time to run get_node_geo_coords():", t4 - t3, "seconds")
#
#         if verbose:
#             print("Running get_edge_geo_coords()...")
#         # let's not over multi-thread a multi-thread
#         if n_threads > 1:
#             n_threads_tmp = 1
#         else:
#             n_threads_tmp = n_threads_max
#         G1 = get_edge_geo_coords(G1, im_file, n_threads=n_threads_tmp,
#                                  verbose=verbose)
#         t5 = time.time()
#         if verbose:
#             print("Time to run get_edge_geo_coords():", t5 - t4, "seconds")
#
#         if verbose:
#             print("pre projection...")
#         node = list(G1.nodes())[-1]
#         if verbose:
#             print(node, "random node props:", G1.nodes[node])
#             # print an edge
#             edge_tmp = list(G1.edges())[-1]
#             print(edge_tmp, "random edge props:", G1.get_edge_data(edge_tmp[0], edge_tmp[1]))
#
#         if verbose:
#             print("projecting graph...")
#         G_projected = ox.project_graph(G1)
#
#         # get geom wkt (for printing/viewing purposes)
#         for i, (u, v, attr_dict) in enumerate(G_projected.edges(data=True)):
#             attr_dict['geometry_wkt'] = attr_dict['geometry'].wkt
#
#         if verbose:
#             print("post projection...")
#             node = list(G_projected.nodes())[-1]
#             print(node, "random node props:", G_projected.nodes[node])
#             # print an edge
#             edge_tmp = list(G_projected.edges())[-1]
#             print(edge_tmp, "random edge props:", G_projected.get_edge_data(edge_tmp[0], edge_tmp[1]))
#
#         t6 = time.time()
#         if verbose:
#             print("Time to project graph:", t6 - t5, "seconds")
#
#         # simplify
#         # G_simp = ox.simplify_graph(G_projected.to_directed())
#         # ox.plot_graph(G_projected)
#         # G1.edge[19][22]
#
#         Gout = G_projected  # G_simp
#
#
#     else:
#         Gout = G0
#
#     # ###########################################################################
#     # # remove short edges?
#     # # this is done in 04_skeletonize.remove_small_terminal()
#     # t31 = time.time()
#     # Gout = remove_short_edges(Gout, min_spur_length_m=min_spur_length_m)
#     # t32 = time.time()
#     # print("Time to remove_short_edges():", t32 - t31, "seconds")
#     # ###########################################################################
#
#     if simplify_graph:
#         if verbose:
#             print("Simplifying graph")
#         t7 = time.time()
#         # 'geometry' tag breaks simplify, so maket it a wkt
#         for i, (u, v, attr_dict) in enumerate(G_projected.edges(data=True)):
#             if 'geometry' in attr_dict.keys():
#                 attr_dict['geometry'] = attr_dict['geometry'].wkt
#
#         G0 = ox.simplify_graph(Gout.to_directed())
#         G0 = G0.to_undirected()
#         # print("G0")
#         # node = list(G0.nodes())[-1]
#         # print(node, "random node props:", G0.nodes[node])
#
#         # Gout = G0
#         # reprojecting graph screws up lat lon, so convert to string?
#         Gout = ox.project_graph(G0)
#
#         # print("Gout")
#         # node = list(Gout.nodes())[-1]
#         # print(node, "random node props:", Gout.nodes[node])
#
#         if verbose:
#             print("post simplify...")
#             node = list(Gout.nodes())[-1]
#             print(node, "random node props:", Gout.nodes[node])
#             # print an edge
#             edge_tmp = list(Gout.edges())[-1]
#             print(edge_tmp, "random edge props:", Gout.get_edge_data(edge_tmp[0], edge_tmp[1]))
#
#         t8 = time.time()
#         if verbose:
#             print("Time to run simplify graph:", t8 - t7, "seconds")
#         # When the simplify funciton combines edges, it concats multiple
#         #  edge properties into a list.  This means that 'geometry_pix' is now
#         #  a list of geoms.  Convert this to a linestring with
#         #   shaply.ops.linemergeconcats
#
#         # BUG, GOOF, ERROR IN OSMNX PROJECT, SO NEED TO MANUALLY SET X, Y FOR NODES!!??
#         if manually_reproject_nodes:
#             # make sure geometry is utm for nodes?
#             for i, (n, attr_dict) in enumerate(Gout.nodes(data=True)):
#                 attr_dict['x'] = attr_dict['utm_east']
#                 attr_dict['y'] = attr_dict['utm_north']
#
#                 # if simplify_graph:
#         #     print ("Simplifying graph")
#         #     t7 = time.time()
#         #     G0 = ox.simplify_graph(Gout.to_directed())
#         #     Gout = G0.to_undirected()
#         #     #Gout = ox.project_graph(G0)
#
#         #     t8 = time.time()
#         #     print ("Time to run simplify graph:", t8-t7, "seconds")
#         #     # When the simplify funciton combines edges, it concats multiple
#         #     #  edge properties into a list.  This means that 'geometry_pix' is now
#         #     #  a list of geoms.  Convert this to a linestring with
#         #     #   shaply.ops.linemergeconcats
#
#         if verbose:
#             print("Merge 'geometry' linestrings...")
#         keys_tmp = ['geometry_wkt', 'geometry_pix', 'geometry_latlon_wkt',
#                     'geometry_utm_wkt']
#         for key_tmp in keys_tmp:
#             if verbose:
#                 print("Merge", key_tmp, "...")
#             for i, (u, v, attr_dict) in enumerate(Gout.edges(data=True)):
#                 if key_tmp not in attr_dict.keys():
#                     continue
#
#                 if (i % 10000) == 0:
#                     print(i, u, v)
#                 geom = attr_dict[key_tmp]
#                 # print (i, u, v, "geom:", geom)
#                 # print ("  type(geom):", type(geom))
#
#                 if type(geom) == list:
#                     # check if the list items are wkt strings, if so, create
#                     #   linestrigs
#                     if (type(geom[0]) == str):  # or (type(geom_pix[0]) == unicode):
#                         geom = [shapely.wkt.loads(ztmp) for ztmp in geom]
#                     # merge geoms
#                     # geom = shapely.ops.linemerge(geom)
#                     # attr_dict[key_tmp] =  geom
#                     geom_out = shapely.ops.linemerge(geom)
#                     # attr_dict[key_tmp] = shapely.ops.linemerge(geom)
#                 elif type(geom) == str:
#                     geom_out = shapely.wkt.loads(geom)
#                     # attr_dict[key_tmp] = shapely.wkt.loads(geom)
#                 else:
#                     geom_out = geom
#
#                 # now straighten edge with rdp
#                 if rdp_epsilon > 0:
#                     if verbose and ((i % 10000) == 0):
#                         print("  Applying rdp...")
#                     coords = list(geom_out.coords)
#                     new_coords = rdp.rdp(coords, epsilon=rdp_epsilon)
#                     geom_out_rdp = LineString(new_coords)
#                     geom_out_final = geom_out_rdp
#                 else:
#                     geom_out_final = geom_out
#
#                 len_out = geom_out_final.length
#
#                 # updata edge properties
#                 attr_dict[key_tmp] = geom_out_final
#
#                 # update length
#                 if key_tmp == 'geometry_pix':
#                     attr_dict['length_pix'] = len_out
#                 if key_tmp == 'geometry_utm_wkt':
#                     attr_dict['length_utm'] = len_out
#
#                     # assign 'geometry' tag to geometry_wkt
#         # !! assign 'geometry' tag to geometry_utm_wkt
#         key_tmp = 'geometry_wkt'  # 'geometry_utm_wkt'
#         for i, (u, v, attr_dict) in enumerate(Gout.edges(data=True)):
#             if verbose and ((i % 10000) == 0):
#                 print("Create 'geometry' field in edges...")
#             line = attr_dict['geometry_utm_wkt']
#             if type(line) == str:  # or type(line) == unicode:
#                 attr_dict['geometry'] = shapely.wkt.loads(line)
#             else:
#                 attr_dict['geometry'] = attr_dict[key_tmp]
#             attr_dict['geometry_wkt'] = attr_dict['geometry'].wkt
#
#             # set length
#             attr_dict['length'] = attr_dict['geometry'].length
#
#             # update wkt_pix?
#             # print ("attr_dict['geometry_pix':", attr_dict['geometry_pix'])
#             attr_dict['wkt_pix'] = attr_dict['geometry_pix'].wkt
#
#             # update 'length_pix'
#             attr_dict['length_pix'] = np.sum([attr_dict['length_pix']])
#
#         # Gout = ox.project_graph(Gout)
#
#     # print a random node and edge
#     if verbose:
#         node_tmp = list(Gout.nodes())[-1]
#         print(node_tmp, "random node props:", Gout.nodes[node_tmp])
#         # print an edge
#         edge_tmp = list(Gout.edges())[-1]
#         print("random edge props for edge:", edge_tmp, " = ",
#               Gout.edges[edge_tmp[0], edge_tmp[1], 0])
#
#     # get a few stats (and set to graph properties)
#     if verbose:
#         logger1.info("Number of nodes: {}".format(len(Gout.nodes())))
#         logger1.info("Number of edges: {}".format(len(Gout.edges())))
#     # print ("Number of nodes:", len(Gout.nodes()))
#     # print ("Number of edges:", len(Gout.edges()))
#     Gout.graph['N_nodes'] = len(Gout.nodes())
#     Gout.graph['N_edges'] = len(Gout.edges())
#
#     # get total length of edges
#     tot_meters = 0
#     for i, (u, v, attr_dict) in enumerate(Gout.edges(data=True)):
#         tot_meters += attr_dict['length']
#     if verbose:
#         print("Length of edges (km):", tot_meters / 1000)
#     Gout.graph['Tot_edge_km'] = tot_meters / 1000
#
#     if verbose:
#         print("G.graph:", Gout.graph)
#
#     # save
#     if len(Gout.nodes()) == 0:
#         nx.write_gpickle(Gout, out_file, protocol=pickle_protocol)
#         return
#
#     # # print a node
#     # node = list(Gout.nodes())[-1]
#     # print (node, "random node props:", Gout.nodes[node])
#     # # print an edge
#     # edge_tmp = list(Gout.edges())[-1]
#     # #print (edge_tmp, "random edge props:", G.edges([edge_tmp[0], edge_tmp[1]])) #G.edge[edge_tmp[0]][edge_tmp[1]])
#     # print (edge_tmp, "random edge props:", Gout.get_edge_data(edge_tmp[0], edge_tmp[1]))
#
#     # save graph
#     if verbose:
#         logger1.info("Saving graph to directory: {}".format(graph_dir))
#     # print ("Saving graph to directory:", graph_dir)
#     nx.write_gpickle(Gout, out_file, protocol=pickle_protocol)
#
#     # # save shapefile as well?
#     # if save_shapefiles:
#     #     logger1.info("Saving shapefile to directory: {}".format(graph_dir))
#     #     try:
#     #         ox.save_graph_shapefile(G, filename=image_id.split('.')[0] , folder=graph_dir, encoding='utf-8')
#     #     except:
#     #         print("Cannot save shapefile...")
#     #     #out_file2 = os.path.join(graph_dir, image_id.split('.')[0] + '.graphml')
#     #     #ox.save_graphml(G, image_id.split('.')[0] + '.graphml', folder=graph_dir)
#     #
#     # # plot, if desired
#     # if make_plots:
#     #     print ("Plotting graph...")
#     #     outfile_plot = os.path.join(graph_dir, image_id)
#     #     print ("outfile_plot:", outfile_plot)
#     #     ox.plot_graph(G, fig_height=9, fig_width=9,
#     #                   #save=True, filename=outfile_plot, margin=0.01)
#     #                   )
#     #     #plt.tight_layout()
#     #     plt.savefig(outfile_plot, dpi=400)
#
#     t7 = time.time()
#     if verbose:
#         print("Total time to run wkt_to_G():", t7 - t0, "seconds")
#
#     return  # Gout
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# #
# # nodes = Gout.nodes()
# #
# # ps = np.array([nodes[i]['o'] for i in nodes])
# #
# # plt.plot(ps[:, 1], ps[:, 0], 'r.')
# #
# # # # title and show
# # plt.title('Build Gout')
# # plt.show()