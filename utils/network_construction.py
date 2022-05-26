"""
Gene Co-expression network construction module.

Author:

    Jayson Co Sia
    http://www-scf.usc.edu/~jsia/

Reference:


"""

import networkx as nx
import pandas as pd


def create_genex_network(csv_file, source='gene1', target='gene2', cor_type="Pearson",
                         soft_power=6, component="ALL", verbose="INFO"):
    """
    Network construction for gene co-expression correlation data.

    :param csv_file: str
        CSV file string containing the gene co-expression correlation data.
    :param source: str
        Source dataframe label.
    :param target: str
        Target dataframe label.
    :param cor_type: str
        Correlation type of the gene co-expression data. Ex. "Pearson", "Spearman".
    :param soft_power: int
        Soft thresholding power parameter used in the power adjacency function.
    :param component: str
        Choice between constructing the whole network or just the largest component.
        - "ALL": Output the complete network.
        - "LC: Output only the largest network component.
    :param verbose: str
        Verbose level.
        - INFO: Print network summary.
        - TRACE: Print detailed information.
        - ERROR: Print only error messages.

    :return:
        g: NetworkX graph class
    """

    df = pd.read_csv(csv_file)
    if verbose == "TRACE":
        print(df.head())
        print()

    g = nx.from_pandas_edgelist(df, source=source, target=target, edge_attr=cor_type)
    if verbose == "INFO" or verbose == "TRACE":
        print("==== Complete network information ====")
        print(nx.info(g))
        print("Number of connected components:", nx.number_connected_components(g))
    if verbose == "TRACE":
        degree_list = sorted(g.degree, key=lambda x: x[1], reverse=True)
        print("Top 5 highest degree nodes:", degree_list[0:5])

    # Get the largest component only
    if component == "LC":
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc)
        g = nx.Graph(g)  # unfreeze graph
        if verbose == "INFO" or verbose == "TRACE":
            print("\n==== Largest component network ====")
            print(nx.info(g))
        if verbose == "TRACE":
            degree_list = sorted(g.degree, key=lambda x: x[1], reverse=True)
            print("Top 5 highest degree nodes:", degree_list[0:5])

    # soft thresholding using power adjacency function  $ a_{ij} = s_{ij}^\beta $
    for edge in g.edges():
        corr = g.edges[edge][cor_type]
        adj_weight = corr ** soft_power
        g.edges[edge]['weight'] = adj_weight

    return g


def label_nodes(g, partition, label):

    return g

# def set_verbose(verbose="ERROR"):
#     if verbose == "INFO":
#         logger.setLevel(logging.INFO)
#     elif verbose == "TRACE":
#         logger.setLevel(logging.TRACE)
#     elif verbose == "DEBUG":
#         logger.setLevel(logging.DEBUG)
#     elif verbose == "ERROR":
#         logger.setLevel(logging.ERROR)
#     else:
#         print('Incorrect verbose level, option:["INFO","DEBUG","ERROR"], use "ERROR instead."')
#         logger.setLevel(logging.ERROR)