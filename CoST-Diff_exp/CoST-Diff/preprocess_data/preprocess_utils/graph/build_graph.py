import networkx as nx
import numpy as np

def build_graph_from_traj(traj_df, uli_index_dict, out_path):
    G = nx.DiGraph()

    for traj in traj_df['traj']:
        idx = [uli_index_dict[str(u)] for u in traj]
        for i in range(len(idx)-1):
            u, v = idx[i], idx[i+1]
            G.add_edge(u, v, weight=G.get_edge_data(u,v,{'weight':0})['weight']+1)

    for n in G.nodes:
        G.add_edge(n, n, weight=1)

    A = nx.adjacency_matrix(G)
    np.savetxt(out_path, A.todense(), delimiter=',')
