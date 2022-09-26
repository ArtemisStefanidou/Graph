from skimage.morphology import skeletonize
import numpy as np
from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict #, defaultdict
from itertools import tee
import cv2
import sknw
from skimage import filters
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import Point, LineString
import shapely.wkt
from multiprocessing.pool import Pool
import osmnx as ox
import utm
from osgeo import gdal, ogr, osr
import rdp as rdp
import geopandas as gpd
from matplotlib.collections import LineCollection
import osmnx.settings as ox_settings
import os
import skimage.io
import argparse as argparse
import json as json
'''----------To do----------'''
#from configs.config import Config




# ====================================================== 04 =================================================================== #
'''---line_points_dist---'''
def line_points_dist(line1, pts):
    return np.cross(line1[1] - line1[0], pts - line1[0]) / np.linalg.norm(line1[1] - line1[0])

'''---get_angle---'''
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


linestring = "LINESTRING {}"

'''---add_small_segments---'''
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


'''---remove_sequential_duplicates---'''
def remove_sequential_duplicates(seq):

    res = [seq[0]]
    for elem in seq[1:]:
        if elem == res[-1]:
            continue
        res.append(elem)
    return res

'''---pairwise---'''
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

'''---remove_duplicate_segments---'''
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

'''---add_direction_change_nodes---'''
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


'''---graph2lines---'''
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


'''flatten'''
def flatten(l):
    return [item for sublist in l for item in sublist]


'''G_to_wkt'''
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
        print("small_segs", small_segs)
        wkt.extend(small_segs)

    if debug:
        vertices = flatten(vertices)

    if not wkt:
        return [linestring.format("EMPTY")]

    # return cross_segs
    return wkt

def image(pathImage):
     # pathImage = 'SN3_roads_train_AOI_3_Paris_PS-MS_img148_result.tif'
    img = cv2.imread(pathImage, cv2.IMREAD_GRAYSCALE)
    return img
    
def skeletonizeImage(img):

    binary = img > filters.threshold_otsu(img)

    ske = skeletonize(binary).astype(np.uint16)

    graph = sknw.build_sknw(ske, ('iso' == False))
    
    return graph


    
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
                    print ("Oops, edge already seen, returning:", edge_loc)
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
        print("  len(G_.edges()):", len(G_.edges()) )
    if super_verbose:
        print("G_.nodes:", G_.nodes())
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
                print("  all_lengths:", all_lengths )
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
                        print("    uprime, vprime:", uprime, vprime )
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
        print("bad_nodes:", bad_nodes)
        print(" len(G'.nodes()):", len(G_.nodes()))
        print(" len(G'.edges()):", len(G_.edges()))
    if super_verbose:
        print("  G_.nodes:", G_.nodes())
        
    return G_

def wkt_to_graph(wkt_list):

    '''To do'''
    # parser = argparse.ArgumentParser()
    # parser.add_argument('config_path')
    # args = parser.parse_args()
    # with open(args.config_path, 'r') as f:
    #     cfg = json.load(f)
    #     config = Config(**cfg)

    #----params----
    n_threads_max = 12
    n_threads = 12
    verbose = False
    im_file = 'SN3_roads_train_AOI_2_Vegas_PS-MS_img2.tif'
    pickle_protocol = 4
    simplify_graph = True
    manually_reproject_nodes = True
    rdp_epsilon = 1
    out_file = "./SN3_roads_train_AOI_2_Vegas_PS-MS_img2.gpickle"
    #rdp_epsilon = config.rdp_epsilon


    node_loc_dic, edge_dic = wkt_list_to_nodes_edges(wkt_list, node_iter=10000,edge_iter=10000)
    G0 = nodes_edges_to_G(node_loc_dic, edge_dic)
    G1 = clean_sub_graphs(G0, min_length=20, weight='length_pix', verbose=verbose, super_verbose=False)
    if len(G1) == 0:
        return G1

    if im_file:
        if n_threads > 1:
            n_threads_tmp = 1
        else:
            n_threads_tmp = n_threads_max
        G1 = get_node_geo_coords(G1, im_file, n_threads=n_threads_tmp,verbose=verbose)
        G1 = get_edge_geo_coords(G1, im_file, n_threads=n_threads_tmp,verbose=verbose)
        try:
            G_projected = ox.project_graph(G1)
        except:
            # make sure points have a geom
            for i, (n, attr_dict) in enumerate(G1.nodes(data=True)):
                # lon, lat = coords_dict[n]
                node_geom = Point(attr_dict['x'], attr_dict['y'])
                attr_dict['geometry'] = node_geom
            G_projected = ox.project_graph(G1)

        # get geom wkt (for printing/viewing purposes)
        for i, (u, v, attr_dict) in enumerate(G_projected.edges(data=True)):
            if 'geometry' in attr_dict.keys():
                attr_dict['geometry_wkt'] = attr_dict['geometry'].wkt
        Gout = G_projected

    else:
        Gout = G0

    if simplify_graph:
        # 'geometry' tag breaks simplify, so maket it a wkt
        for i, (u, v, attr_dict) in enumerate(G_projected.edges(data=True)):
            if 'geometry' in attr_dict.keys():
                attr_dict['geometry'] = attr_dict['geometry'].wkt

        G0 = ox.simplify_graph(Gout.to_directed())
        G0 = G0.to_undirected()
        # Gout = ox.project_graph(G0)

        # BUG, GOOF, ERROR IN OSMNX PROJECT, SO NEED TO MANUALLY SET X, Y FOR NODES!!??
        if manually_reproject_nodes:
            # make sure geometry is utm for nodes?
            for i, (n, attr_dict) in enumerate(Gout.nodes(data=True)):
                attr_dict['x'] = attr_dict['utm_east']
                attr_dict['y'] = attr_dict['utm_north']


        keys_tmp = ['geometry_wkt', 'geometry_pix', 'geometry_latlon_wkt','geometry_utm_wkt']
        for key_tmp in keys_tmp:
            for i, (u, v, attr_dict) in enumerate(Gout.edges(data=True)):
                if key_tmp not in attr_dict.keys():
                    continue

                if (i % 10000) == 0:
                    print(i, u, v)
                geom = attr_dict[key_tmp]

                if type(geom) == list:
                    # check if the list items are wkt strings, if so, create
                    #   linestrigs
                    if (type(geom[0]) == str):  # or (type(geom_pix[0]) == unicode):
                        geom = [shapely.wkt.loads(ztmp) for ztmp in geom]
                    # merge geoms
                    # geom = shapely.ops.linemerge(geom)
                    # attr_dict[key_tmp] =  geom
                    geom_out = shapely.ops.linemerge(geom)
                    # attr_dict[key_tmp] = shapely.ops.linemerge(geom)
                elif type(geom) == str:
                    geom_out = shapely.wkt.loads(geom)
                    # attr_dict[key_tmp] = shapely.wkt.loads(geom)
                else:
                    geom_out = geom

                if rdp_epsilon > 0:
                    coords = list(geom_out.coords)
                    new_coords = rdp.rdp(coords, epsilon=rdp_epsilon)
                    geom_out_rdp = LineString(new_coords)
                    geom_out_final = geom_out_rdp
                else:
                    geom_out_final = geom_out

                len_out = geom_out_final.length

                # updata edge properties
                attr_dict[key_tmp] = geom_out_final

                # update length
                if key_tmp == 'geometry_pix':
                    attr_dict['length_pix'] = len_out
                if key_tmp == 'geometry_utm_wkt':
                    attr_dict['length_utm'] = len_out
        # assign 'geometry' tag to geometry_wkt
        # !! assign 'geometry' tag to geometry_utm_wkt
        '''----------here------------'''
        key_tmp = 'geometry_utm_wkt'  # 'geometry_utm_wkt'
        for i, (u, v, attr_dict) in enumerate(Gout.edges(data=True)):
            line = attr_dict['geometry_utm_wkt']
            if type(line) == str:  # or type(line) == unicode:
                attr_dict['geometry'] = shapely.wkt.loads(line)
            else:
                attr_dict['geometry'] = attr_dict[key_tmp]
                attr_dict['geometry_wkt'] = attr_dict['geometry'].wkt

                # set length
                attr_dict['length'] = attr_dict['geometry'].length

                # update wkt_pix?
                # print ("attr_dict['geometry_pix':", attr_dict['geometry_pix'])
                attr_dict['wkt_pix'] = attr_dict['geometry_pix'].wkt

                # update 'length_pix'
                attr_dict['length_pix'] = np.sum([attr_dict['length_pix']])
        # get total length of edges
        tot_meters = 0
        for i, (u, v, attr_dict) in enumerate(Gout.edges(data=True)):
            tot_meters += attr_dict['length']
        Gout.graph['Tot_edge_km'] = tot_meters / 1000

        # save
        if len(Gout.nodes()) == 0:
            nx.write_gpickle(Gout, out_file, protocol=pickle_protocol)
            return
        nx.write_gpickle(Gout, out_file, protocol=pickle_protocol)
        print('the end')
        # G_epsg3857 = ox.project_graph(Gout, to_crs='epsg:3857')
        # print("out_file", out_file)
        # p0_tmp, p1_tmp, p2_tmp = out_file.split('.')
        # out_file_tmp = p0_tmp + p1_tmp + '_3857.' + p2_tmp
        # nx.write_gpickle(G_epsg3857, out_file_tmp, protocol=pickle_protocol)
        # print(type(Gout))
        return Gout

  
    
# ====================================================== Plot Graphs =================================================================== #
def plot_Graph(graph,img):
    
    plt.imshow(img, cmap='gray')
    
    if str(type(graph)) == "<class 'networkx.classes.graph.Graph'>" :
        
        #graph = ox.simplify_graph(graph.to_directed())
        '---plot edges---'
        for (s, e) in graph.edges():
    
            ps = graph[s][e]['pts']
            plt.plot(ps[:, 1], ps[:, 0], 'green')
        
        '---plot nodes---'
        nodes = graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        plt.plot(ps[:, 1], ps[:, 0], 'r.')
        
        # # title and show
        plt.title('Build Graph')
        plt.show()
    
    elif str(type(graph)) == "<class 'networkx.classes.multigraph.MultiGraph'>" or str(type(graph)) == "<class 'networkx.classes.multidigraph.MultiDiGraph'>":
        '---plot edges---'
        for (u,v,attrib_dict) in list(graph.edges.data()):
    
            x_pix = []
            y_pix = []
            
            x_pix.append(attrib_dict['start_loc_pix'][0])
            x_pix.append(attrib_dict['end_loc_pix'][0])
            y_pix.append(attrib_dict['start_loc_pix'][1])
            y_pix.append(attrib_dict['end_loc_pix'][1])
            plt.plot(x_pix, y_pix, 'green')

        '---plot nodes---'
        nodes_list = []
        for (u,attrib_dict) in list(graph.nodes.data()):
            # weight.append(attrib_dict['start'])
            plt.plot(attrib_dict['x_pix'], attrib_dict['y_pix'], 'r.')
            nodes_list.append((attrib_dict['x_pix'], attrib_dict['y_pix']))
        
        # # title and show
        plt.title('Build Graph')
        plt.show()


def get_node_geo_coords(G, im_file, fix_utm_zone=True, n_threads=12,
                        verbose=False):

    # get pixel params
    params = []
    nn = len(G.nodes())

    for i, (n, attr_dict) in enumerate(G.nodes(data=True)):
        # if verbose and ((i % 1000) == 0):
        #     print (i, "/", nn, "node:", n)
        x_pix, y_pix = attr_dict['x_pix'], attr_dict['y_pix']
        params.append((n, x_pix, y_pix, im_file))

    if verbose:
        print("node params[:5]:", params[:5])

    n_threads = min(n_threads, nn)
    # execute
    n_threads = 12
    print("Computing geo coords for nodes (" + str(n_threads) + " threads)...")

    if n_threads > 1:
        pool = Pool(n_threads)
        coords_dict_list = pool.map(pixelToGeoCoord, params)
    else:
        coords_dict_list = pixelToGeoCoord(params[0])

    # combine the disparate dicts
    coords_dict = {}

    for d in coords_dict_list:
        #the update wants iterable and type(d) = int
        coords_dict.update(d)
    if verbose:
        print("  nodes: list(coords_dict)[:5]:", list(coords_dict)[:5])

    # update data
    print("Updating data properties")
    utm_letter = 'Oooops'
    for i, (n, attr_dict) in enumerate(G.nodes(data=True)):
        if verbose and ((i % 5000) == 0):
            print(i, "/", nn, "node:", n)

        lon, lat = coords_dict[n]

        # fix zone
        if i == 0 or fix_utm_zone == False:
            [utm_east, utm_north, utm_zone, utm_letter] = \
                utm.from_latlon(lat, lon)
            if verbose and (i == 0):
                print("utm_letter:", utm_letter)
                print("utm_zone:", utm_zone)
        else:
            [utm_east, utm_north, _, _] = utm.from_latlon(lat, lon,
                                                          force_zone_number=utm_zone, force_zone_letter=utm_letter)

        if lat > 90:
            print("lat > 90, returning:", n, attr_dict)
            return
        attr_dict['lon'] = lon
        attr_dict['lat'] = lat
        attr_dict['utm_east'] = utm_east
        attr_dict['utm_zone'] = utm_zone
        attr_dict['utm_letter'] = utm_letter
        attr_dict['utm_north'] = utm_north
        attr_dict['x'] = lon
        attr_dict['y'] = lat

    return G

def pixelToGeoCoord(params):
    '''from spacenet geotools'''
    # lon, lat = pixelToGeoCoord(x_pix, y_pix, im_file, targetSR=targetSR)
    #         params.append((x_pix, y_pix, im_file, targetSR))

    sourceSR = ''
    geomTransform = ''
    targetSR = osr.SpatialReference()
    targetSR.ImportFromEPSG(4326)

    identifier, xPix, yPix, inputRaster = params

    if targetSR =='':
        performReprojection=False
        targetSR = osr.SpatialReference()
        targetSR.ImportFromEPSG(4326)
    else:
        performReprojection=True

    if geomTransform=='':
        srcRaster = gdal.Open(inputRaster)
        geomTransform = srcRaster.GetGeoTransform()

        source_sr = osr.SpatialReference()
        source_sr.ImportFromWkt(srcRaster.GetProjectionRef())

    geom = ogr.Geometry(ogr.wkbPoint)
    xOrigin = geomTransform[0]
    yOrigin = geomTransform[3]
    pixelWidth = geomTransform[1]
    pixelHeight = geomTransform[5]

    xCoord = (xPix * pixelWidth) + xOrigin
    yCoord = (yPix * pixelHeight) + yOrigin
    geom.AddPoint(xCoord, yCoord)

    if performReprojection:
        if sourceSR=='':
            srcRaster = gdal.Open(inputRaster)
            sourceSR = osr.SpatialReference()
            sourceSR.ImportFromWkt(srcRaster.GetProjectionRef())
        coord_trans = osr.CoordinateTransformation(sourceSR, targetSR)
        geom.Transform(coord_trans)

    return {identifier: (geom.GetX(), geom.GetY())}


def get_edge_geo_coords(G, im_file, remove_pix_geom=True, fix_utm_zone=True,
                        n_threads=12, verbose=False, super_verbose=False):
    '''Get geo coords of all edges'''

    # first, get utm letter and zone of first node in graph
    for i, (n, attr_dict) in enumerate(G.nodes(data=True)):
        x_pix, y_pix = attr_dict['x_pix'], attr_dict['y_pix']
        if i > 0:
            break
    params_tmp = ('tmp', x_pix, y_pix, im_file)

    tmp_dict = pixelToGeoCoord(params_tmp)

    (lon, lat) = list(tmp_dict.values())[0]
    [utm_east, utm_north, utm_zone, utm_letter] = utm.from_latlon(lat, lon)

    # now get edge params
    params = []
    ne = len(list(G.edges()))
    for i, (u, v, attr_dict) in enumerate(G.edges(data=True)):
        # if verbose and ((i % 1000) == 0):
        #     print (i, "/", ne, "edge:", u,v)
        #     print ("  attr_dict_init:", attr_dict)

        geom_pix = attr_dict['geometry_pix']

        # if i == 0 :
        #     # identifier, geom_pix_wkt, im_file, utm_zone, utm_letter, verbose = params
        #     params_tmp = (identifier, geom_pix_wkt, im_file, None, None, verbose)
        #     dict_tmp = convert_pix_lstring_to_geo(params_tmp)
        #     (lstring_latlon, lstring_utm, utm_zone, utm_letter) = list(dict_tmp.values())[0]
        #     # lstring_latlon, lstring_utm, utm_zone, utm_letter \
        #     #        = convert_pix_lstring_to_geo_raw(geom_pix, im_file)
        #     params.append(((u,v), geom_pix.wkt, im_file,
        #                    utm_zone, utm_letter, verbose))

        # identifier, geom_pix_wkt, im_file, utm_zone, utm_letter, verbose = params
        if fix_utm_zone == False:
            params.append(((u, v), geom_pix.wkt, im_file,
                           None, None, super_verbose))
        else:
            params.append(((u, v), geom_pix.wkt, im_file,
                           utm_zone, utm_letter, super_verbose))

    if verbose:
        print("edge params[:5]:", params[:5])

    n_threads = min(n_threads, ne)
    n_threads = 12
    # execute
    print("Computing geo coords for edges (" + str(n_threads) + " threads)...")

    if n_threads > 1:
        pool = Pool(n_threads)
        coords_dict_list = pool.map(convert_pix_lstring_to_geo, params)
    else:
        coords_dict_list = convert_pix_lstring_to_geo(params[0])

    # combine the disparate dicts
    coords_dict = {}
    for d in coords_dict_list:
        coords_dict.update(d)
    if verbose:
        print("  edges: list(coords_dict)[:5]:", list(coords_dict)[:5])

    print("Updating edge data properties")
    for i, (u, v, attr_dict) in enumerate(G.edges(data=True)):

        # if verbose and ((i % 1000) == 0):
        #     print (i, "/", ne, "edge:", u,v)
        #     print ("  attr_dict_init:", attr_dict)
        geom_pix = attr_dict['geometry_pix']

        lstring_latlon, lstring_utm, utm_zone, utm_letter = coords_dict[(u, v)]

        attr_dict['geometry_latlon_wkt'] = lstring_latlon.wkt
        attr_dict['geometry_utm_wkt'] = lstring_utm.wkt
        attr_dict['length_latlon'] = lstring_latlon.length
        attr_dict['length_utm'] = lstring_utm.length
        attr_dict['length'] = lstring_utm.length
        attr_dict['utm_zone'] = utm_zone
        attr_dict['utm_letter'] = utm_letter
        if verbose and ((i % 1000) == 0):
            print("   attr_dict_final:", attr_dict)

        # geometry screws up osmnx.simplify function
        if remove_pix_geom:
            # attr_dict['geometry_wkt'] = lstring_latlon.wkt
            attr_dict['geometry_pix'] = geom_pix.wkt

        # try actual geometry, not just linestring, this seems necessary for
        #  projections
        attr_dict['geometry'] = lstring_latlon

        # ensure utm length isn't excessive
        if lstring_utm.length > 5000:
            print(u, v, "edge length too long:", attr_dict, "returning!")
            return

    return G

def convert_pix_lstring_to_geo(params):

    '''Convert linestring in pixel coords to geo coords
    If zone or letter changes inthe middle of line, it's all screwed up, so
    force zone and letter based on first point
    (latitude, longitude, force_zone_number=None, force_zone_letter=None)
    Or just force utm zone and letter explicitly
        '''

    identifier, geom_pix_wkt, im_file, utm_zone, utm_letter, verbose = params
    shape = shapely.wkt.loads(geom_pix_wkt)
    x_pixs, y_pixs = shape.coords.xy
    coords_latlon = []
    coords_utm = []
    for i,(x,y) in enumerate(zip(x_pixs, y_pixs)):
        params_tmp = ('tmp', x, y, im_file)
        tmp_dict = pixelToGeoCoord(params_tmp)
        (lon, lat) = list(tmp_dict.values())[0]
        # targetSR = osr.SpatialReference()
        # targetSR.ImportFromEPSG(4326)
        # lon, lat = pixelToGeoCoord_raw(x, y, im_file, targetSR=targetSR)

        if utm_zone and utm_letter:
            [utm_east, utm_north, _, _] = utm.from_latlon(lat, lon,
                force_zone_number=utm_zone, force_zone_letter=utm_letter)
        else:
            [utm_east, utm_north, utm_zone, utm_letter] = utm.from_latlon(lat, lon)

#        # If zone or letter changes in the middle of line, it's all screwed up, so
#        # force zone and letter based on first point?
#        if i == 0:
#            [utm_east, utm_north, utm_zone, utm_letter] = utm.from_latlon(lat, lon)
#        else:
#            [utm_east, utm_north, _, _] = utm.from_latlon(lat, lon,
#                force_zone_number=utm_zone, force_zone_letter=utm_letter)
        if verbose:
            print("lat lon, utm_east, utm_north, utm_zone, utm_letter]",
                [lat, lon, utm_east, utm_north, utm_zone, utm_letter])
        coords_utm.append([utm_east, utm_north])
        coords_latlon.append([lon, lat])

    lstring_latlon = LineString([Point(z) for z in coords_latlon])
    lstring_utm = LineString([Point(z) for z in coords_utm])

    return {identifier: (lstring_latlon, lstring_utm, utm_zone, utm_letter)}


# ====================================================== 08 =================================================================== #
def graph_to_gdfs_pix(G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True):
    """
    Convert a graph into node and/or edge GeoDataFrames
    Parameters
    ----------
    G : networkx multidigraph
    nodes : bool
        if True, convert graph nodes to a GeoDataFrame and return it
    edges : bool
        if True, convert graph edges to a GeoDataFrame and return it
    node_geometry : bool
        if True, create a geometry column from node x and y data
    fill_edge_geometry : bool
        if True, fill in missing edge geometry fields using origin and
        destination nodes
    Returns
    -------
    GeoDataFrame or tuple
        gdf_nodes or gdf_edges or both as a tuple
    """

    if not (nodes or edges):
        raise ValueError('You must request nodes or edges, or both.')

    to_return = []

    if nodes:

        #start_time = time.time()

        nodes = {node:data for node, data in G.nodes(data=True)}
        gdf_nodes =gpd.GeoDataFrame(nodes).T

        if node_geometry:
            #gdf_nodes['geometry'] = gdf_nodes.apply(lambda row: Point(row['x'], row['y']), axis=1)
            gdf_nodes['geometry_pix'] = gdf_nodes.apply(lambda row: Point(row['x_pix'], row['y_pix']), axis=1)

        gdf_nodes.crs = G.graph['crs']
        gdf_nodes.gdf_name = '{}_nodes'.format(G.graph['name'])
        gdf_nodes['osmid'] = gdf_nodes['osmid'].astype(np.int64).map(make_str)

        to_return.append(gdf_nodes)

    if edges:

        #start_time = time.time()

        # create a list to hold our edges, then loop through each edge in the
        # graph
        edges = []
        for u, v, key, data in G.edges(keys=True, data=True):

            # for each edge, add key and all attributes in data dict to the
            # edge_details
            edge_details = {'u':u, 'v':v, 'key':key}
            for attr_key in data:
                edge_details[attr_key] = data[attr_key]

             # if edge doesn't already have a geometry attribute, create one now
            # if fill_edge_geometry==True
            if 'geometry_pix' not in data:
                if fill_edge_geometry:
                    point_u = Point((G.nodes[u]['x_pix'], G.nodes[u]['y_pix']))
                    point_v = Point((G.nodes[v]['x_pix'], G.nodes[v]['y_pix']))
                    edge_details['geometry_pix'] = LineString([point_u, point_v])
                else:
                    edge_details['geometry_pix'] = np.nan

            # # if edge doesn't already have a geometry attribute, create one now
            # # if fill_edge_geometry==True
            # if 'geometry' not in data:
            #     if fill_edge_geometry:
            #         point_u = Point((G.nodes[u]['x'], G.nodes[u]['y']))
            #         point_v = Point((G.nodes[v]['x'], G.nodes[v]['y']))
            #         edge_details['geometry'] = LineString([point_u, point_v])
            #     else:
            #         edge_details['geometry'] = np.nan

            edges.append(edge_details)

        # create a GeoDataFrame from the list of edges and set the CRS
        gdf_edges = gpd.GeoDataFrame(edges)
        gdf_edges.crs = G.graph['crs']
        gdf_edges.gdf_name = '{}_edges'.format(G.graph['name'])

        to_return.append(gdf_edges)

    if len(to_return) > 1:
        return tuple(to_return)
    else:
        return to_return[0]

def color_func(speed):
    if speed < 15:
        color = '#ffffb2'
    elif speed >= 15 and speed < 25:
        color = '#ffe281'
#     if speed < 17.5:
#         color = '#ffffb2'
#     elif speed >= 17.5 and speed < 25:
#         color = '#ffe281'
    elif speed >= 25 and speed < 35:
        color = '#fec357'
    elif speed >= 35 and speed < 45:
        color = '#fe9f45'
    elif speed >= 45 and speed < 55:
        color = '#fa7634'
    elif speed >= 55 and speed < 65:
        color = '#f24624'
    elif speed >= 65 and speed < 75:
        color = '#da2122'
    elif speed >= 75:
        color = '#bd0026'
    return color

def make_color_dict_list(max_speed=80, verbose=False):
    color_dict = {}
    color_list = []
    for speed in range(max_speed):
        c = color_func(speed)
        color_dict[speed] = c
        color_list.append(c)
    if verbose:
        print("color_dict:", color_dict)
        print("color_list:", color_list)

    return color_dict, color_list

def save_and_show(fig, ax, save, show, close, filename, file_format, dpi,
                  axis_off, tight_layout=False,
                  invert_xaxis=False, invert_yaxis=True,
                  verbose=False):
    """
    Save a figure to disk and show it, as specified.
    Assume filename holds entire path to file

    Parameters
    ----------
    fig : figure
    ax : axis
    save : bool
        whether to save the figure to disk or not
    show : bool
        whether to display the figure or not
    close : bool
        close the figure (only if show equals False) to prevent display
    filename : string
        the name of the file to save
    file_format : string
        the format of the file to save (e.g., 'jpg', 'png', 'svg')
    dpi : int
        the resolution of the image file if saving
    axis_off : bool
        if True matplotlib axis was turned off by plot_graph so constrain the
        saved figure's extent to the interior of the axis
    Returns
    -------
    fig, ax : tuple
    """

    if invert_yaxis:
        ax.invert_yaxis()
    if invert_xaxis:
        ax.invert_xaxis()

    # save the figure if specified
    if save:
        # start_time = time.time()

        # create the save folder if it doesn't already exist
        if not os.path.exists(os.path.dirname("./Save")):
            os.mkdir("./Save")
            os.chmod(os.path.join("./", "Save"), 0o777)

        path_filename = "./Save"  # os.path.join(settings.imgs_folder, os.extsep.join([filename, file_format]))

        if file_format == 'svg':
            # if the file_format is svg, prep the fig/ax a bit for saving
            ax.axis('off')
            ax.set_position([0, 0, 1, 1])
            ax.patch.set_alpha(0.)
            fig.patch.set_alpha(0.)
            fig.savefig(path_filename, bbox_inches=0, format=file_format, facecolor=fig.get_facecolor(),
                        transparent=True)
        else:
            if axis_off:
                # if axis is turned off, constrain the saved figure's extent to
                # the interior of the axis
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            else:
                extent = 'tight'

            if tight_layout:
                # extent = 'tight'
                fig.gca().set_axis_off()
                fig.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                    hspace=0, wspace=0)
                plt.margins(0, 0)
                # fig.gca().xaxis.set_major_locator(NullLocator())
                # fig.gca().yaxis.set_major_locator(NullLocator())
                # fig.savefig(path_filename, dpi=dpi, bbox_inches=extent,
                #             format=file_format, facecolor=fig.get_facecolor(),
                #             transparent=True, pad_inches=0)
            else:
                print("To do save fig")

                # fig.savefig(path_filename, dpi=dpi, bbox_inches=extent,
                #             format=file_format, facecolor=fig.get_facecolor(),
                #             transparent=True)

        # if verbose:
        #     print('Saved the figure to disk in {:,.2f} seconds'.format(time.time() - start_time))

    # show the figure if specified
    print("show", show)
    show = True
    if show:
        # start_time = time.time()
        plt.show()
        # if verbose:
        #     print('Showed the plot in {:,.2f} seconds'.format(time.time() - start_time))
    # if show=False, close the figure if close=True to prevent display
    elif close:
        plt.close()

    return fig, ax

def plot_graph_pix(G, im=None, bbox=None, fig_height=6, fig_width=None, margin=0.02,
                   axis_off=True, equal_aspect=False, bgcolor='w', show=True,
                   save=False, close=True, file_format='png', filename='temp',
                   default_dpi=300, annotate=False, node_color='#66ccff', node_size=15,
                   node_alpha=1, node_edgecolor='none', node_zorder=1,
                   edge_color='#999999', edge_linewidth=1, edge_alpha=1,
                   edge_color_key='speed_mph', color_dict={},
                   edge_width_key='speed_mph',
                   edge_width_mult=1. / 25,
                   use_geom=True,
                   invert_xaxis=False, invert_yaxis=False,
                   fig=None, ax=None):
    """
    Plot a networkx spatial graph.
    Parameters
    ----------
    G : networkx multidigraph
    bbox : tuple
        bounding box as north,south,east,west - if None will calculate from
        spatial extents of data. if passing a bbox, you probably also want to
        pass margin=0 to constrain it.
    fig_height : int
        matplotlib figure height in inches
    fig_width : int
        matplotlib figure width in inches
    margin : float
        relative margin around the figure
    axis_off : bool
        if True turn off the matplotlib axis
    equal_aspect : bool
        if True set the axis aspect ratio equal
    bgcolor : string
        the background color of the figure and axis
    show : bool
        if True, show the figure
    save : bool
        if True, save the figure as an image file to disk
    close : bool
        close the figure (only if show equals False) to prevent display
    file_format : string
        the format of the file to save (e.g., 'jpg', 'png', 'svg')
    filename : string
        the name of the file if saving
    default_dpi : int
        the resolution of the image file if saving (may get altered for
        large images)
    annotate : bool
        if True, annotate the nodes in the figure
    node_color : string
        the color of the nodes
    node_size : int
        the size of the nodes
    node_alpha : float
        the opacity of the nodes
    node_edgecolor : string
        the color of the node's marker's border
    node_zorder : int
        zorder to plot nodes, edges are always 2, so make node_zorder 1 to plot
        nodes beneath them or 3 to plot nodes atop them
    edge_color : string
        the color of the edges' lines
    edge_linewidth : float
        the width of the edges' lines
    edge_alpha : float
        the opacity of the edges' lines
    edge_width_key : str
        optional: key in edge propwerties to determine edge width,
        supercedes edge_linewidth, default to "speed_mph"
    edge_width_mult : float
        factor to rescale width for plotting, default to 1./25, which gives
        a line width of 1 for 25 mph speed limit.
    use_geom : bool
        if True, use the spatial geometry attribute of the edges to draw
        geographically accurate edges, rather than just lines straight from node
        to node
    Returns
    -------
    fig, ax : tuple
    """

    print('Begin plotting the graph...')
    node_Xs = [float(x) for _, x in G.nodes(data='x_pix')]
    node_Ys = [float(y) for _, y in G.nodes(data='y_pix')]
    # node_Xs = [float(x) for _, x in G.nodes(data='x')]
    # node_Ys = [float(y) for _, y in G.nodes(data='y')]

    # get north, south, east, west values either from bbox parameter or from the
    # spatial extent of the edges' geometries
    if bbox is None:

        edges = graph_to_gdfs_pix(G, nodes=False, fill_edge_geometry=True)
        # print("plot_graph_pix():, edges.columns:", edges.columns)
        # print("type edges['geometry_pix'].:", type(edges['geometry_pix']))
        # print("type gpd.GeoSeries(edges['geometry_pix']):", type(gpd.GeoSeries(edges['geometry_pix'])))
        # print("type gpd.GeoSeries(edges['geometry_pix'][0]):", type(gpd.GeoSeries(edges['geometry_pix']).iloc[0]))

        west, south, east, north = gpd.GeoSeries(edges['geometry_pix']).total_bounds
        # west, south, east, north = edges.total_bounds
    else:
        north, south, east, west = bbox

    bbox_aspect_ratio = (north - south) / (east - west)
    if fig_width is None:
        fig_width = fig_height / bbox_aspect_ratio

    if im is not None:
        if fig == None and ax == None:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        # ax.imshow(im)

    else:
        if fig == None and ax == None:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=bgcolor)
        ax.set_facecolor(bgcolor)

    #start_time = time.time()
    lines = []
    widths = []
    edge_colors = []
    for u, v, data in G.edges(keys=False, data=True):
        if 'geometry_pix' in data and use_geom:

            xs, ys = data['geometry_pix'].xy
            lines.append(list(zip(xs, ys)))
        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x_pix']
            y1 = G.nodes[u]['y_pix']
            x2 = G.nodes[v]['x_pix']
            y2 = G.nodes[v]['y_pix']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)

        # get widths
        if edge_width_key in data.keys():
            width = int(np.rint(data[edge_width_key] * edge_width_mult))
        else:
            width = edge_linewidth
        widths.append(width)

        if edge_color_key and color_dict:
            print("data edge_color_key", data)
            color_key_val = 45
            edge_colors.append(color_dict[color_key_val])
        else:
            edge_colors.append(edge_color)

    lc = LineCollection(lines, colors=edge_colors,
                        linewidths=widths,
                        alpha=edge_alpha, zorder=2)
    ax.add_collection(lc)

    # scatter plot the nodes
    ax.scatter(node_Xs, node_Ys, s=node_size, c=node_color, alpha=node_alpha,
               edgecolor=node_edgecolor, zorder=node_zorder)

    # set the extent of the figure
    margin_ns = (north - south) * margin
    margin_ew = (east - west) * margin
    ax.set_ylim((south - margin_ns, north + margin_ns))
    ax.set_xlim((west - margin_ew, east + margin_ew))

    # configure axis appearance
    xaxis = ax.get_xaxis()
    yaxis = ax.get_yaxis()

    xaxis.get_major_formatter().set_useOffset(False)
    yaxis.get_major_formatter().set_useOffset(False)

    # the ticks in so there's no space around the plot
    if axis_off:
        ax.axis('off')
        ax.margins(0)
        ax.tick_params(which='both', direction='in')
        xaxis.set_visible(False)
        yaxis.set_visible(False)
        fig.canvas.draw()

    if equal_aspect:
        # make everything square
        ax.set_aspect('equal')
        fig.canvas.draw()
    else:
        # if the graph is not projected, conform the aspect ratio to not stretch the plot
        if G.graph['crs'] == ox_settings.default_crs:
            coslat = np.cos((min(node_Ys) + max(node_Ys)) / 2. / 180. * np.pi)
            ax.set_aspect(1. / coslat)
            fig.canvas.draw()

    # annotate the axis with node IDs if annotate=True
    if annotate:
        for node, data in G.nodes(data=True):
            ax.annotate(node, xy=(data['x_pix'], data['y_pix']))

    # # update dpi, if image
    # if im is not None:
    #
    #     max_dpi = int(23000 / max(fig_height, fig_width))
    #     h, w = im.shape[:2]
    #     # try to set dpi to native resolution of imagery
    #     desired_dpi = max(default_dpi, 1.0 * h / fig_height)
    #     # desired_dpi = max(default_dpi, int( np.max(im.shape) / max(fig_height, fig_width) ) )
    #     dpi = int(np.min([max_dpi, desired_dpi]))
    # else:
    #     dpi = default_dpi
    dpi = default_dpi
    # save and show the figure as specified
    fig, ax = save_and_show(fig, ax, save, show, close, filename,
                            file_format, dpi, axis_off,
                            invert_xaxis=invert_xaxis,
                            invert_yaxis=invert_yaxis)
    return fig, ax


###############################################################################
def plot_graph_route_pix(G, route, im=None, bbox=None, fig_height=6, fig_width=None,
                     margin=0.02, bgcolor='w', axis_off=True, show=True,
                     save=False, close=True, file_format='png', filename='temp',
                     default_dpi=300, annotate=False, node_color='#999999',
                     node_size=15, node_alpha=1, node_edgecolor='none',
                     node_zorder=1,
                     edge_color='#999999', edge_linewidth=1,
                     edge_alpha=1,
                     edge_color_key='speed_mph', color_dict={},
                     edge_width_key='speed_mph',
                     edge_width_mult=1./25,
                     use_geom=True, origin_point=None,
                     destination_point=None, route_color='r', route_linewidth=4,
                     route_alpha=0.5, orig_dest_node_alpha=0.5,
                     orig_dest_node_size=100,
                     orig_dest_node_color='r',
                     invert_xaxis=False, invert_yaxis=True,
                     fig=None, ax=None):
    """
    Plot a route along a networkx spatial graph.
    Parameters
    ----------
    G : networkx multidigraph
    route : list
        the route as a list of nodes
    bbox : tuple
        bounding box as north,south,east,west - if None will calculate from
        spatial extents of data. if passing a bbox, you probably also want to
        pass margin=0 to constrain it.
    fig_height : int
        matplotlib figure height in inches
    fig_width : int
        matplotlib figure width in inches
    margin : float
        relative margin around the figure
    axis_off : bool
        if True turn off the matplotlib axis
    bgcolor : string
        the background color of the figure and axis
    show : bool
        if True, show the figure
    save : bool
        if True, save the figure as an image file to disk
    close : bool
        close the figure (only if show equals False) to prevent display
    file_format : string
        the format of the file to save (e.g., 'jpg', 'png', 'svg')
    filename : string
        the name of the file if saving
    default_dpi : int
        the resolution of the image file if saving
    annotate : bool
        if True, annotate the nodes in the figure
    node_color : string
        the color of the nodes
    node_size : int
        the size of the nodes
    node_alpha : float
        the opacity of the nodes
    node_edgecolor : string
        the color of the node's marker's border
    node_zorder : int
        zorder to plot nodes, edges are always 2, so make node_zorder 1 to plot
        nodes beneath them or 3 to plot nodes atop them
    edge_color : string
        the color of the edges' lines
    edge_linewidth : float
        the width of the edges' lines
    edge_alpha : float
        the opacity of the edges' lines
    use_geom : bool
        if True, use the spatial geometry attribute of the edges to draw
        geographically accurate edges, rather than just lines straight from node
        to node
    origin_point : tuple
        optional, an origin (lat, lon) point to plot instead of the origin node
    destination_point : tuple
        optional, a destination (lat, lon) point to plot instead of the
        destination node
    route_color : string
        the color of the route
    route_linewidth : int
        the width of the route line
    route_alpha : float
        the opacity of the route line
    orig_dest_node_alpha : float
        the opacity of the origin and destination nodes
    orig_dest_node_size : int
        the size of the origin and destination nodes
    orig_dest_node_color : string
        the color of the origin and destination nodes
        (can be a string or list with (origin_color, dest_color))
        of nodes
    Returns
    -------
    fig, ax : tuple
    """

    # plot the graph but not the route
    fig, ax = plot_graph_pix(G, im=im, bbox=bbox, fig_height=fig_height, fig_width=fig_width,
                         margin=margin, axis_off=axis_off, bgcolor=bgcolor,
                         show=False, save=False, close=False, filename=filename,
                         default_dpi=default_dpi, annotate=annotate, node_color=node_color,
                         node_size=node_size, node_alpha=node_alpha,
                         node_edgecolor=node_edgecolor, node_zorder=node_zorder,
                         edge_color_key=edge_color_key, color_dict=color_dict,
                         edge_color=edge_color, edge_linewidth=edge_linewidth,
                         edge_alpha=edge_alpha, edge_width_key=edge_width_key,
                         edge_width_mult=edge_width_mult,
                         use_geom=use_geom,
                         fig=fig, ax=ax)

    # the origin and destination nodes are the first and last nodes in the route
    origin_node = route[0]
    destination_node = route[-1]

    if origin_point is None or destination_point is None:
        # if caller didn't pass points, use the first and last node in route as
        # origin/destination
        origin_destination_ys = (G.nodes[origin_node]['y_pix'],
                                 G.nodes[destination_node]['y_pix'])
        origin_destination_xs = (G.nodes[origin_node]['x_pix'],
                                 G.nodes[destination_node]['x_pix'])
    else:
        # otherwise, use the passed points as origin/destination
        origin_destination_xs = (origin_point[0], destination_point[0])
        origin_destination_ys = (origin_point[1], destination_point[1])

    # scatter the origin and destination points
    ax.scatter(origin_destination_xs, origin_destination_ys,
               s=orig_dest_node_size,
               c=orig_dest_node_color,
               alpha=orig_dest_node_alpha, edgecolor=node_edgecolor, zorder=4)

    # plot the route lines
    edge_nodes = list(zip(route[:-1], route[1:]))
    lines = []
    for u, v in edge_nodes:
        # if there are parallel edges, select the shortest in length
        data = min(G.get_edge_data(u, v).values(), key=lambda x: x['length'])

        # if it has a geometry attribute (ie, a list of line segments)
        if 'geometry_pix' in data and use_geom:
            # add them to the list of lines to plot
            xs, ys = data['geometry_pix'].xy
            lines.append(list(zip(xs, ys)))
        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x_pix']
            y1 = G.nodes[u]['y_pix']
            x2 = G.nodes[v]['x_pix']
            y2 = G.nodes[v]['y_pix']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)

    # add the lines to the axis as a linecollection
    lc = LineCollection(lines, colors=route_color, linewidths=route_linewidth, alpha=route_alpha, zorder=3)
    ax.add_collection(lc)

    # update dpi, if image
    if im is not None:
        #   mpl can handle a max of 2^29 pixels, or 23170 on a side
        # recompute max_dpi
        max_dpi = int(23000 / max(fig_height, fig_width))
        h, w = im.shape[:2]
        # try to set dpi to native resolution of imagery
        desired_dpi = max(default_dpi, 1.0 * h / fig_height)
        #desired_dpi = max(default_dpi, int( np.max(im.shape) / max(fig_height, fig_width) ) )
        dpi = int(np.min([max_dpi, desired_dpi ]))


    # save and show the figure as specified
    fig, ax = save_and_show(fig, ax, save, show, close, filename,
                            file_format, dpi, axis_off,
                            invert_yaxis=invert_yaxis,
                            invert_xaxis=invert_xaxis)

    return fig, ax


def plot_Graph_with_pixels(pathImage, Gout):
    # dar tutorial settings
    im_file = pathImage
    save_only_route_png = False  # True
    fig_height = 12
    fig_width = 12
    node_color = '#66ccff'  # light blue
    node_size = 0.4
    node_alpha = 0.6
    edge_color = '#000'  # lightblue1
    edge_linewidth = 0.5
    edge_alpha = 0.6
    edge_color_key = 'inferred_speed_mph'
    orig_dest_node_size = 8 * node_size
    max_plots = 3
    shuffle = True
    invert_xaxis = False
    invert_yaxis = False
    route_color = 'blue'
    orig_dest_node_color = 'blue'
    route_linewidth = 4 * edge_linewidth

    node = list(Gout.nodes())[-1]
    print("Gout.nodes[node]['lat']", Gout.nodes[node]['lat'])
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

    try:
        # convert to rgb (cv2 reads in bgr)
        img_cv2 = cv2.imread(im_file, 1)
        print("img_cv2.shape:", img_cv2.shape)
        im = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    except:
        im = skimage.io.imread(im_file).astype(np.uint8)  # [::-1]
        # im = skimage.io.imread(im_file, as_grey=False).astype(np.uint8)#[::-1]

    # set dpi to approximate native resolution
    desired_dpi = int(np.max(im.shape) / np.max([fig_height, fig_width]))
    # max out dpi at 3500
    dpi = int(np.min([3500, desired_dpi]))
    print("plot dpi:", dpi)

    plot_graph_pix(Gout, im_file, fig_height=fig_height, fig_width=fig_width,
                       node_size=node_size, node_alpha=node_alpha, node_color=node_color,
                       edge_linewidth=edge_linewidth, edge_alpha=edge_alpha, edge_color=edge_color,
                       default_dpi=dpi,
                       edge_color_key=None,
                       show=False, save=True,
                       invert_yaxis=invert_yaxis,
                       invert_xaxis=invert_xaxis)

    # plot with speed
    out_file_plot_speed = "speed.tif"
    print("outfile_plot_speed:", out_file_plot_speed)
    # width_key = 'inferred_speed_mph'
    color_dict, color_list = make_color_dict_list()
    plot_graph_pix(Gout, im, fig_height=fig_height, fig_width=fig_width,
                       node_size=node_size, node_alpha=node_alpha, node_color=node_color,
                       edge_linewidth=edge_linewidth, edge_alpha=edge_alpha, edge_color=edge_color,
                       filename=out_file_plot_speed, default_dpi=dpi,
                       show=False, save=True,
                       invert_yaxis=invert_yaxis,
                       invert_xaxis=invert_xaxis,
                       edge_color_key=edge_color_key, color_dict=color_dict)

    ################
    # plot graph route
    print("\nPlot a random route on the graph...")
    # t0 = time.time()
    # set source
    source_idx = np.random.randint(0, len(Gout.nodes()))
    source = list(Gout.nodes())[source_idx]
    # get routes
    lengths, paths = nx.single_source_dijkstra(Gout, source=source, weight='length')
    # random target
    targ_idx = np.random.randint(0, len(list(lengths.keys())))
    target = list(lengths.keys())[targ_idx]
    # specific route
    route = paths[target]
    print("source:", source)
    print("target:", target)
    print("route:", route)
    # plot route
    out_file_route = '_ox_route_r0_length.tif'
    print("outfile_route:", out_file_route)
    plot_graph_route_pix(Gout, route, im=im, fig_height=fig_height, fig_width=fig_width,
                             node_size=node_size, node_alpha=node_alpha, node_color=node_color,
                             edge_linewidth=edge_linewidth, edge_alpha=edge_alpha, edge_color=edge_color,
                             orig_dest_node_size=orig_dest_node_size,
                             route_color=route_color,
                             orig_dest_node_color=orig_dest_node_color,
                             route_linewidth=route_linewidth,
                             filename=out_file_route, default_dpi=dpi,
                             show=False, save=True,
                             invert_yaxis=invert_yaxis,
                             invert_xaxis=invert_xaxis,
                             edge_color_key=None)
    # t1 = time.time()
    # print("Time to run plot_graph_route_pix():", t1-t0, "seconds")

    ################
    # plot graph route (speed)
    print("\nPlot a random route on the graph...")
    # t0 = time.time()
    # get routes
    lengths, paths = nx.single_source_dijkstra(Gout, source=source, weight='Travel Time (h)')
    # specific route
    route = paths[target]

    # plot route
    out_file_route = '_ox_route_r0_speed.tif'
    print("outfile_route:", out_file_route)
    plot_graph_route_pix(Gout, route, im=im, fig_height=fig_height, fig_width=fig_width,
                             node_size=node_size, node_alpha=node_alpha, node_color=node_color,
                             edge_linewidth=edge_linewidth, edge_alpha=edge_alpha, edge_color=edge_color,
                             orig_dest_node_size=orig_dest_node_size,
                             route_color=route_color,
                             orig_dest_node_color=orig_dest_node_color,
                             route_linewidth=route_linewidth,
                             filename=out_file_route, default_dpi=dpi,
                             show=False, save=True,
                             invert_yaxis=invert_yaxis,
                             invert_xaxis=invert_xaxis,
                             edge_color_key=edge_color_key, color_dict=color_dict)
    # t1 = time.time()
    # print("Time to run plot_graph_route_pix():", t1-t0, "seconds")