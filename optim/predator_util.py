import open3d as o3d


def get_overlap_ratio(
    source,
    target,
    source_to_target,
    threshold=0.03,
):
    """
    We compute overlap ratio from source point cloud to target point cloud
    """
    source.transform(source_to_target)
    pcd_tree = o3d.geometry.KDTreeFlann(target)

    match_count = 0
    for i, point in enumerate(source.points):
        [count, _, _] = pcd_tree.search_radius_vector_3d(point, threshold)
        if(count!=0):
            match_count+=1

    overlap_ratio = match_count / len(source.points)
    return overlap_ratio
