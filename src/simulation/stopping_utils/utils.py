

def check_connectivity_two_clusters(edge_list: list, f_community: list, s_community: list, min_connections: int) -> bool:
    """
    Check if minimum connections between both communities.

    Args:
        :param edge_list: edge list on which to check
        :param f_community: first node list
        :param f_community: second node list
        :param min_connections: minimum connection so that it evaluates to True
    """
    count = 0
    for u in f_community:
        for v in s_community:
            if (u, v) in edge_list:
                count += 1
                if count >= min_connections:
                    return True

    return False