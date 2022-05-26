"""
Community Detection using Ollivier-Ricci Curvature


Author:

    Jayson Co Sia
    http://www-scf.usc.edu/~jsia/

Reference:

"""

import time
import operator
import networkx as nx
import networkx.algorithms.community as nx_comm

# dendrogram implementation
import matplotlib.pyplot as plt
from itertools import chain, combinations
from scipy.cluster.hierarchy import dendrogram

from multiprocessing import cpu_count
# from GraphRicciCurvature_old.OllivierRicci import ricciCurvature
from GraphRicciCurvature.OllivierRicci import OllivierRicci

from copy import deepcopy


def label_community_truth(G, ground_truth_size_list):
    """
    Label the graph node attribute according to the community ground truth size list.

    :param G: A connected NetworkX graph.
    :param ground_truth_size_list: A list of ground truth community labels indexed by node number.
    :return:
    """
    ctr = 0
    community_truth = {}
    for idx, val in enumerate(ground_truth_size_list):
        for i in range(val):
            node_id = ctr + i
            community_truth[node_id] = idx
        ctr = ctr + val

    nx.set_node_attributes(G, community_truth, 'label_truth')


def label_communities(G, component_list, label):
    """"
    :param G: A NetworkX graph.
    :param component_list: A list of sets of connected components (communities).
    :param label: Node label string.
    :return: NULL
    """

    community_label_list = {}
    for idx, community in enumerate(component_list):
        for node in community:
            community_label_list[node] = idx

    nx.set_node_attributes(G, community_label_list, label)


def detection_accuracy(G, truth_label, detect_label, verbose=False):
    """

    :param G: A NetworkX graph with set 'label_truth' and 'label_detected' node parameters.
    :return:
    """

    # truth set list
    truth_labels = nx.get_node_attributes(G, truth_label)
    truth_label_set = set(truth_labels.values())
    num_labels_truth = len(truth_label_set)

    truth_dict = dict()
    ctr = 0
    for label in truth_label_set:
        truth_dict[label] = ctr
        ctr = ctr + 1

    # detected set list
    detect_labels = nx.get_node_attributes(G, detect_label)
    detect_label_set = set(detect_labels.values())
    num_labels_detected = len(detect_label_set)

    detect_dict = dict()
    ctr = 0
    for label in detect_label_set:
        detect_dict[label] = ctr
        ctr = ctr + 1

    truth_community_list = [set() for i in range(num_labels_truth)]
    detected_community_list = [set() for i in range(num_labels_detected)]

    for key, val in truth_labels.items():
        idx = truth_dict[val]
        truth_community_list[idx].add(key)

    for key, val in detect_labels.items():
        idx = detect_dict[val]
        detected_community_list[idx].add(key)

    # make list of frozensets
    truth_sets = set([frozenset(x) for x in truth_community_list])
    detected_sets = set([frozenset(x) for x in detected_community_list])

    if verbose:
        print(truth_label_set)
        print(detect_label_set)
        print("truth communities")
        for item in truth_sets:
            print(item)

        print()
        print("detected communities")
        for item in detected_sets:
            print(item)

    # do a set comparison to detect how many communities are correctly detection
    common_set = truth_sets.intersection(detected_sets)
    accuracy = len(common_set) / len(truth_sets)

    # return detection accuracy and list of correctly detected community
    return accuracy, common_set


def orcci(G, weight='weight', method='Sinkhorn', edges_known=[], block_label="blockRicci",
                max_cores=16, verbose="ERROR"):
    """
    Community detection using Ollivier-Ricci curvature with known side information.

    :param G: (NetworkX graph class)
        Input undirected network to perform the ORCCI community detection method.
    :param weight: (str)
        Whether to consider the edge weights. Possible values: 'weighted' or 'unweighted'.
    :param method: (str)
        Method type for (Wasserstein) distance calculation. Default: 'Sinkhorn' ot approximation. Other
        possible values: 'OTD' - exact optimal transport calculation via linear programming; "ATD" for Average
        Transportation Distance.
    :param edges_known: (list: edge tuple)
        Edge list of known side information
    :param block_label: (str)
        Label for the discovered "block" or community
    :param max_cores: (int)
        Number of allowed maximum cores.
    :param verbose: (str) = {"INFO", "TRACE","DEBUG","ERROR"}
        Verbose level. Verbose tags taken from graphRicciCurvature.
            - "INFO": show only iteration process log.
            - "TRACE": show detailed iteration process log.
            - "DEBUG": show all output logs.
            - "ERROR": only show log if error happened.

    :return: orc: (OllivierRicci class)
        The OllivierRicci class of given edge with nodes labelled according to its assigned community

             component_list: list of sets of nodes from detected communities
             comp_sizes: list detected community sizes
             partitions: list of tuples of partitions
             rm_edges: list of removed edges

    ---------------------------------------------
    TODO Append Community Detection as Node Properties
    TODO Batch version (greedy approach?)
    TODO Set number of threads explicitly
    TODO Unweighted (created edge weight attribute as 1.0)
    """
    t0 = time.time()

    MIN_TOL = -0.1  # Minimum tolerance for edge ORC
    MAX_CORES = max_cores  # maximum number of cores

    ALPHA = 0.00001

    # list of all partitions
    partitions_list = []

    cpu_cores = cpu_count()
    if cpu_cores > MAX_CORES:
        cpu_cores = MAX_CORES

    if True:
        print("Using number of cpu threads: = %d\n" % cpu_cores)

    # Setting of OllivierRicci object parameters
    orc = OllivierRicci(G, weight=weight, method=method, alpha=ALPHA, proc=cpu_cores,
                        verbose=verbose)
    G_prime = orc.G

    # Keep OllivierRicci Class object containing the original network
    orc_orig = OllivierRicci(G, weight=weight, method=method, alpha=ALPHA, proc=cpu_cores,
                             verbose=verbose)  # placeholder for the original calculation
    # G_orig = orc_orig.G

    iter_ctr = 0  # iteration counter

    edge_list = []

    # TODO Consider nodes_known instead of edges_known
    # 2020.08.02 - Force positive ricci curvature for known edges (side information)
    for k in edges_known:
        G_prime[k[0]][k[1]]['ricciCurvature'] = 1

    if verbose == "TRACE":
        print("Network summary")
        print(nx.info(G))
    elif verbose == "INFO":
        print("Partition splits")
        print("========================================================")

    while True:
        iter_ctr = iter_ctr + 1

        if verbose == "TRACE":
            print("========================================================")
            print("Iteration: %i" % (iter_ctr))

        # calculate ORC for all edges at the first iteration
        if not edge_list:
            edge_list = list(G_prime.edges())  # 2020.07.31 - JCS changed to list

        # 2020.08.03 - JCS [TODO: optimize - implement flag instead]
        # remove known edges in edge_list for Ricci calculation
        for i in edge_list:
            for k in edges_known:
                k_inv = (k[1], k[0])
                if k in edge_list:
                    edge_list.remove(k)
                elif k_inv in edge_list:
                    edge_list.remove(k_inv)

        # print("edge_list: ", edge_list)

        # 2020.08.29 - changed to OllivierRicci class (using latest graphRicciCurvature package)
        # TODO: check if orc_orig is properly copied
        orc_edges_dict = orc.compute_ricci_curvature_edges(edge_list=edge_list)
        # G_prime = orc.G
        nx.set_edge_attributes(G_prime, orc_edges_dict, 'ricciCurvature')
        # G_prime = ricciCurvature(G_prime, alpha=0, weight=None, proc=cpu_cores, edge_list=edge_list, verbose=False)

        if iter_ctr == 1:
            orc_orig.G = deepcopy(orc.G)
            G_orig = orc_orig.G

        edge_Ricci_curv = nx.get_edge_attributes(G_prime, 'ricciCurvature')

        # sort according to increasing curvature
        edge_Ricci_curv_sorted = sorted(edge_Ricci_curv.items(), key=operator.itemgetter(1))
        # print(edge_Ricci_curv_sorted)

        # TODO: Batch (greedy approach)
        # rm_edge = dict()        # {edge: curv}
        # for edge, curv in edge_Ricci_curv.items():
        #     if curv < MIN_TOL:
        #         rm_edge[edge] = curv

        # dictionary of edges to remove
        rm_edge = dict()  # {edge: ricciCurvature}

        # sequential removal
        least_curved_edge = edge_Ricci_curv_sorted[0]

        # for edge, curv in edge_Ricci_curv.items():
        if least_curved_edge[1] < MIN_TOL:  # remove only the most negative edge ORC
            rm_edge[least_curved_edge[0]] = least_curved_edge[1]

        # TODO change order (move after removal of edge)
        # List of component partitions
        component_list = list(nx.connected_components(G_prime))
        num_components = len(component_list)
        comp_sizes = [len(x) for x in component_list]
        largest_cc = max(comp_sizes)

        # 2020.09.14 - JCS (dendrogram)
        modularity = nx_comm.modularity(G_orig, component_list)

        partition = {"iteration": iter_ctr, "partition": component_list, "modularity": modularity}
        partition_set = set(frozenset(i) for i in component_list)

        if iter_ctr == 1:  # first iteration
            # prev_partition_set = {}
            partitions_list.append(partition)
        else:
            prev_partition = partitions_list[-1]["partition"]
            prev_partition_set = set(frozenset(i) for i in prev_partition)

            # check if same as previous iteration
            if prev_partition_set != partition_set:
                if verbose == "INFO":
                    print("iteration: %d,\t #components: %d,\t max_component_size: %d,\t modularity: %.4f" % (
                        partition["iteration"], len(partition["partition"]), largest_cc, partition["modularity"]))
                elif verbose == "TRACE":
                    print("PARTITION SPLIT: #components: %d,\t max_component_size: %d,\t modularity: %.4f" % (
                        len(partition["partition"]), largest_cc, partition["modularity"]))
                partitions_list.append(partition)

        # TODO: Batch
        # if rm_edge:
        #     minCurv = min(rm_edge.values())
        #     maxCurv = max(rm_edge.values())
        # else:
        #     minCurv = 0
        #     maxCurv = 0

        if verbose == "TRACE":
            print("\nNumber of components: %i" % num_components)
            print("Size of largest component: %i" % largest_cc)

            str1 = "List of component sizes: "
            str2 = ' '.join(str(x) for x in comp_sizes)
            print(str1 + '[' + str2 + ']')
            print("Number of deleted edges: %i" % (len(rm_edge)))
            print("Deleted edge: ")
            print(rm_edge)
            # print("Removed edge curvatures: %.4f (min), %.4f (max)" % (minCurv, maxCurv))

        # remove edges simultaneously (sequential and batch)
        G_prime.remove_edges_from(rm_edge.keys())

        # Only recalculate curvature for affected edges
        #   neighbors of u and v nodes where uv is the deleted edge
        nodes_affected = set(node for edge in rm_edge for node in edge)
        edges_affected = G_prime.edges(nodes_affected)

        # TODO add affected edges where there's a direct path between the neighbors of u and v

        edge_list = list(edges_affected)  # 2020.07.31 - JCS made list

        if verbose == "TRACE":
            print("Number of affected nodes/edges: %i, %i" % (len(nodes_affected), len(edges_affected)))

        # exit condition
        if not rm_edge:  # and not in known_list
            component_list = list(nx.connected_components(G_prime))
            num_components = len(component_list)
            comp_sizes = [len(x) for x in component_list]
            largest_cc = max(comp_sizes)

            idx_max, mod_max = find_max_modularity_partition(partitions_list)
            partition_maxmod = partitions_list[idx_max]['partition']
            comp_sizes_max = [len(x) for x in partition_maxmod]

            label_communities(G_orig, partition_maxmod, block_label + "MaxMod")
            label_communities(G_orig, component_list, block_label + "Final")

            t_end = time.time() - t0

            if True:
                print()
                print("FINAL RESULT")
                print("========================================================")
                print("Elapsed time: %.4f (s)" % t_end)
                print("Number of iterations: %i" % iter_ctr)
                print("Number of communities detected: %i" % num_components)
                print("Size of largest community: %i" % largest_cc)

                str1 = "List of component sizes (final): "
                str2 = ' '.join(str(x) for x in comp_sizes)
                print(str1 + '[' + str2 + ']')

                print("MAXIMUM MODULARITY PARTITION RESULT")
                print("========================================================")
                print("Modularity: %.4f" % mod_max)
                print("Number of communities detected: %i" % len(partition_maxmod))
                str1 = "List of component sizes (max mod): "
                str2 = ' '.join(str(x) for x in comp_sizes_max)
                print(str1 + '[' + str2 + ']')

                print()

            break

    return orc_orig, partitions_list  # return the original graph object with labeled nodes (G_orig)


def draw_dendrogram(partition_list, threshold=None, figsize=(8, 6), fontsize=16, verbose=False):

    TOL = 0.0001  # threshold tolerance

    communities = []
    for part in partition_list:
        communities.append(part["partition"])

    # building initial dict of node_id to each possible subset:
    node_id = 0
    init_node2community_dict = {}
    for comm in communities:
        for subset in list(comm):
            if subset not in init_node2community_dict.values():
                init_node2community_dict[node_id] = subset
                node_id += 1

    # turning this dictionary to the desired format in @mdml's answer
    node_id_to_children = {e: [] for e in init_node2community_dict.keys()}
    for node_id1, node_id2 in combinations(init_node2community_dict.keys(), 2):
        for node_id_parent, group in init_node2community_dict.items():
            if len(init_node2community_dict[node_id1].intersection(
                    init_node2community_dict[node_id2])) == 0 and group == init_node2community_dict[node_id1].union(
                    init_node2community_dict[node_id2]):
                node_id_to_children[node_id_parent].append(node_id1)
                node_id_to_children[node_id_parent].append(node_id2)

    # also recording node_labels dict for the correct label for dendrogram leaves
    node_labels = dict()
    for node_id, group in init_node2community_dict.items():
        if len(group) == 1:
            node_labels[node_id] = list(group)[0]
        else:
            node_labels[node_id] = node_id

    # also needing a subset to rank dict to later know within all k-length merges which came first
    subset_rank_dict = dict()
    rank = 0
    for e in communities[::-1]:
        for p in list(e):
            if tuple(p) not in subset_rank_dict:
                subset_rank_dict[tuple(sorted(p))] = rank
                rank += 1
    subset_rank_dict[tuple(sorted(chain.from_iterable(communities[-1])))] = rank

    # for key, value in subset_rank_dict.items():
    #     print(key, ' : ', value)

    # TODO: Alternative distance metric instead of splitting index.
    # # my function to get a merge height so that it is unique (probably not that efficient)
    # def get_merge_height(sub):
    #     sub_tuple = tuple(sorted([node_labels[i] for i in sub]))
    #     n = len(sub_tuple)
    #     other_same_len_merges = {k: v for k, v in subset_rank_dict.items() if len(k) == n}
    #     min_rank, max_rank = min(other_same_len_merges.values()), max(other_same_len_merges.values())
    #     range = (max_rank-min_rank) if max_rank > min_rank else 1
    #     print(len(sub))
    #     print((subset_rank_dict[sub_tuple] - min_rank))
    #     return float(len(sub)) + 0.8 * (subset_rank_dict[sub_tuple] - min_rank) / range

    # finally using @mdml's magic, slightly modified:
    G = nx.DiGraph(node_id_to_children)
    nodes = G.nodes()
    leaves = set(n for n in nodes if G.out_degree(n) == 0)
    inner_nodes = [n for n in nodes if G.out_degree(n) > 0]

    # Compute the size of each subtree
    subtree = dict((n, [n]) for n in leaves)
    for u in inner_nodes:
        children = set()
        node_list = list(node_id_to_children[u])
        while len(node_list) > 0:
            v = node_list.pop(0)
            children.add(v)
            node_list += node_id_to_children[v]
        subtree[u] = sorted(children & leaves)

    inner_nodes.sort(key=lambda n: len(subtree[n]))  # <-- order inner nodes ascending by subtree size, root is last

    # Construct the linkage matrix
    leaves = sorted(leaves)
    index = dict((tuple([n]), i) for i, n in enumerate(leaves))
    Z = []
    k = len(leaves)
    itr = 0.0
    for i, n in enumerate(inner_nodes):
        itr += 1
        children = node_id_to_children[n]
        x = children[0]
        for y in children[1:]:
            z = tuple(sorted(subtree[x] + subtree[y]))
            i, j = index[tuple(sorted(subtree[x]))], index[tuple(sorted(subtree[y]))]
            # print(i, j, itr, z)
            Z.append([i, j, itr, len(z)])  # <-- float is required by the dendrogram function
            index[z] = k
            subtree[z] = list(z)
            x = z
            k += 1

    # if verbose:
    labels = [node_labels[node_id] for node_id in leaves]

    max_modularity = 0
    max_idx = 0
    max_iter = 0

    for idx, part in enumerate(partition_list):
        largest_cc = 0

        modularity = part["modularity"]
        if modularity > max_modularity:
            max_modularity = modularity
            max_idx = idx
            max_iter = part["iteration"]

        for community in part["partition"]:
            if len(community) > largest_cc:
                largest_cc = len(community)
        if verbose:
            print("[%d] iteration: %d,\t #components: %d,\t max_component_size: %d,\t modularity: %.4f" % (
            idx, part["iteration"], len(part["partition"]), largest_cc, part["modularity"]))

    if verbose:
        print("\nLeaf ID and # of community memberships")
        leaf_community_sizes = {}
        for label_id in labels:
            leaf_community_sizes[label_id] = len(init_node2community_dict[label_id])
            # print(label_id, ":", len(init_node2community_dict[label_id]))
        print(leaf_community_sizes)

    print("\nMaximum modularity: %.4f @ split: %d (iteration: %d)." % (max_modularity, max_idx, max_iter))

    # Choosing the cutoff for default case
    if threshold is None:
        threshold = (len(partition_list) - 1) - max_idx + TOL
        # threshold = -1  # no cutoff
    else:
        threshold = threshold + TOL

    # dendrogram
    plt.figure(figsize=figsize)
    dendrogram(Z, color_threshold=threshold, labels=labels)
    plt.title('Dendrogram', fontsize=fontsize)
    plt.xlabel('Leaf community IDs', fontsize=fontsize)
    plt.ylabel('split index', fontsize=fontsize)
    plt.savefig('dendrogram.png', dpi=200)


def find_max_modularity_partition(partitions_list):
    """
    Find the iteration or split index that gives the maximum modularity.
    :param partitions_list: list[{"iteration", "partition", "modularity"}
    :return: idx_max, mod_max
    """

    idx_max = 0
    mod_max = 0
    for idx, part in enumerate(partitions_list):
        modularity = partitions_list[idx]['modularity']

        if modularity > mod_max:
            mod_max = modularity
            idx_max = idx

    return idx_max, mod_max
