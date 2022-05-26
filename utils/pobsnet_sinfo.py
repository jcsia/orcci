"""
Analysis of partially observed network with side information

Author: Jayson Co Sia (http://www-scf.usc.edu/~jsia/)

References:

Versions
0.1         File creation

TODO:
    *

"""

from random import sample
from communityDetection.orcci import *  # TODO: update latest ricciCurvature package


def sinfo_orc(g_obs, percent_sinfo, block_label="block", verbose="ERROR"):
    """
    Method 1
    :param g_obs: (nx.Graph class)
    :param percent_sinfo: (float) percent of nodes with known community labels
    :param method: (str) TODO Add method options.
    :param block_label: (str) block label to consider as ground truth
    :param verbose: (bool) info and/or debug flag
    :return orc_sinfo: (OllivierRicci class) graph output with nodes labelled based on
        orcci
    ----------
    TODO:
        * to update to a general multi-block network
    """

    # # Randomly choose % of nodes as known side info
    # g_obs_size = len(g_obs)  # number of nodes
    # nodes_obs = g_obs.nodes()  # list of all node ids
    # known_subset_size = round(g_obs_size * percent_sinfo)

    # Randomly choose % of nodes as known side info
    # list of all nodes with non-'unknown' label
    sinfo_list = []
    nodes_obs = g_obs.nodes()  # list of all node ids
    for node in nodes_obs:
        if g_obs.nodes[node][block_label] != 'unknown':  # TODO make 'unknown' configurable
            sinfo_list.append(node)

    known_subset_size = round(len(sinfo_list) * percent_sinfo)

    # Get subgraph based on randomly sampled nodes (without replacement)
    nodes_known = sample(sinfo_list, k=known_subset_size)
    # nodes_subset.sort()
    # print("known nodes:", nodes_known)

    # label known nodes as '1', unknown nodes as '0'
    for node in nodes_obs:
        known = 0
        if node in nodes_known:
            known = 1
        g_obs.nodes[node]['known'] = known

    # TODO: to update to a general multi-block network
    # check if known nodes belonging to the same community have connection
    known_partition = dict()  # {0: [], 1: [], 2: []}

    edges_known = []  # list of all edges known (both existing and artificially created)
    for i in nodes_known:
        block = g_obs.nodes[i][block_label]
        insert_to_dict_list(known_partition, block, i)
        # known_partition[block].append(i)

    # print(known_partition)

    # artificially create edge links between known nodes that belong to the same community
    for i in known_partition:  # range(len(known_partition)) # block
        for j in range(len(known_partition[i])):  # range(len(known_partition[i])) # node
            node_s = known_partition[i][j]
            for k in range(j + 1, len(known_partition[i])):
                node_d = known_partition[i][k]
                # Method 1: "STRONG" -- add edges
                # if not g_obs.has_edge(node_s, node_d):
                #     g_obs.add_edge(node_s, node_d, weight=1.0, ricciCurvature=1.0)
                # edges_known.append((node_s, node_d))

                # Method 2: "Default"
                if g_obs.has_edge(node_s, node_d):
                    edges_known.append((node_s, node_d))

    if verbose == "TRACE":
        print("known edges:", edges_known)

    g_orc_obs, size_list = orcci(g_obs, edges_known=edges_known, block_label="blockRicciSinfo",
                                 verbose=verbose)

    return g_orc_obs


def sbm_sinfo_orc(g_obs, percent_sinfo, block_label="block", verbose="ERROR"):
    """
    TODO Backup for SBM test
    :param g_obs: (nx.Graph class)
    :param percent_sinfo: (float) percent of nodes with known community labels
    :param method: (str) TODO Add method options.
    :param block_label: (str) block label to consider as ground truth
    :param verbose: (bool) info and/or debug flag
    :return orc_sinfo: (OllivierRicci class) graph output with nodes labelled based on
        orcci
    ----------
    TODO:
        * to update to a general multi-block network
    """

    # Randomly choose % of nodes as known side info
    g_obs_size = len(g_obs)  # number of nodes
    nodes_obs = g_obs.nodes()  # list of all node ids
    known_subset_size = round(g_obs_size * percent_sinfo)

    # Get subgraph based on randomly sampled nodes (without replacement)
    nodes_known = sample(nodes_obs, k=known_subset_size)
    # nodes_subset.sort()
    # print("known nodes:", nodes_known)

    # label known nodes as '1', unknown nodes as '0'
    for node in nodes_obs:
        known = 0
        if node in nodes_known:
            known = 1
        g_obs.nodes[node]['known'] = known

    # TODO: to update to a general multi-block network
    # check if known nodes belonging to the same community have connection
    known_partition = dict()  # {0: [], 1: [], 2: []}

    edges_known = []  # list of all edges known (both existing and artificially created)
    for i in nodes_known:
        block = g_obs.nodes[i][block_label]
        insert_to_dict_list(known_partition, block, i)
        # known_partition[block].append(i)

    # print(known_partition)

    # artificially create edge links between known nodes that belong to the same community
    for i in known_partition:  # range(len(known_partition)) # block
        for j in range(len(known_partition[i])):  # range(len(known_partition[i])) # node
            node_s = known_partition[i][j]
            for k in range(j + 1, len(known_partition[i])):
                node_d = known_partition[i][k]
                # Method 1: "STRONG" -- add edges
                # if not g_obs.has_edge(node_s, node_d):
                #     g_obs.add_edge(node_s, node_d, weight=1.0, ricciCurvature=1.0)
                # edges_known.append((node_s, node_d))

                # Method 2: "Default"
                if g_obs.has_edge(node_s, node_d):
                    edges_known.append((node_s, node_d))

    if verbose == "TRACE":
        print("known edges:", edges_known)

    g_orc_obs, size_list = orcci(g_obs, edges_known=edges_known, block_label="blockRicciSinfo",
                                 verbose=verbose)

    return g_orc_obs


def sbm_rand_pobs(f_in, percent_obs_list, verbose="INFO"):
    g = nx.read_gml(f_in)
    if verbose == "INFO":
        print(nx.info(g) + "\n")

    # Randomly choose percent of nodes observed (rest of the nodes are hidden together with its associated edges)
    g_size = len(g)
    nodes_all = g.nodes()  # list of all node id's

    for percent_obs in percent_obs_list:
        g_obs_size = round(g_size * percent_obs)

        # Get subgraph based on randomly sampled nodes
        nodes_subset = sample(nodes_all, k=g_obs_size)
        g_subset = g.subgraph(nodes_subset).copy()

        print(nx.info(g_subset))

        g_orc_obs, partitions_list = orcci(g_subset, block_label="blockRicciPobs", verbose=verbose)

        # output .gml filename creation
        str_obs = str(percent_obs).replace('.', '')
        f_out = f_in[:-4] + "_obs" + str_obs[:2] + ".gml"

        print("written in: " + f_out + "\n")
        nx.write_gml(g_orc_obs.G, f_out, stringizer=str)


def sbm_pobs_sinfo(f_in, known_list, block_label="block", verbose="ERROR"):
    """
    Perform orcci with side info. Iterate through all values in known_list.

    :param f_in: (str) input file string
    :param known_list: (list) list of nodes with known community label
    :param block_label: (str) side information label
    :param verbose: (str) verbose flag
    :return: (None)
    """

    g_obs = nx.read_gml(f_in)

    for percent_sinfo in known_list:
        orc_sinfo = sinfo_orc(g_obs, percent_sinfo, block_label=block_label, verbose=verbose)  # TODO change for SBM

        # output .gml filename creation
        str_obs = str(percent_sinfo).replace('.', '')
        f_out = f_in[:-4] + "_sinfo" + str_obs[:2] + ".gml"

        print("written in", f_out)
        nx.write_gml(orc_sinfo.G, f_out, stringizer=str)


def genex_pathway_pobs_sinfo(f_in, known_list, verbose="INFO"):
    """
    Gene co-expression with pathway side information ORCCI analysis.

    :param f_in: GML network file input.
    :param known_list: List of tuples known side information e.g. [(node, label id), ...]
    :param verbose: Flag for different information printing for debugging and analysis.
    :return:
    """

    g_obs = nx.read_gml(f_in)

    for percent_sinfo in known_list:
        g_orc_seq_obs = sinfo_orc(g_obs, percent_sinfo, verbose)

        # output .gml filename creation
        str_obs = str(percent_sinfo).replace('.', '')
        f_out = f_in[:-4] + "_obs_" + str_obs[:2] + ".gml"

        print("written in", f_out)
        nx.write_gml(g_orc_seq_obs, f_out, stringizer=str)


def insert_to_dict_list(dict_list, key, val):
    if key in dict_list:
        dict_list[key].append(val)
    else:
        dict_list[key] = [val]
