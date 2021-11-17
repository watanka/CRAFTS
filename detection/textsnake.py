import numpy as np
import cv2
from misc import find_bottom, find_long_edges, split_edge_seqence, \
    norm2, vector_cos, vector_sin, split_long_edges


def norm2(x, axis=None):
    if axis:
        return np.sqrt(np.sum(x ** 2, axis=axis))
    return np.sqrt(np.sum(x ** 2))

def cos(p1, p2):
    return (p1 * p2).sum() / (norm2(p1) * norm2(p2))

def find_bottom(pts):

    if len(pts) > 4:
        e = np.concatenate([pts, pts[:3]])
        candidate = []
        for i in range(1, len(pts) + 1):
            v_prev = e[i] - e[i - 1]
            v_next = e[i+2 ] - e[i+1] # i+2, i+1
            
            if cos(v_prev, v_next) < -0.7: # cosine similarity, 수정 전, < -0.7
                candidate.append((i % len(pts), (i + 1) % len(pts), norm2(e[i] - e[i + 1])))

        if len(candidate) != 2 or candidate[0][0] == candidate[1][1] or candidate[0][1] == candidate[1][0]:
            # if candidate number < 2, or two bottom are joined, select 2 farthest edge
            mid_list = []
            for i in range(len(pts)):
                mid_point = (e[i] + e[(i + 1) % len(pts)]) / 2
                mid_list.append((i, (i + 1) % len(pts), mid_point))

            dist_list = []
            for i in range(len(pts)):
                for j in range(len(pts)):
                    s1, e1, mid1 = mid_list[i]
                    s2, e2, mid2 = mid_list[j]
                    dist = norm2(mid1 - mid2)
                    dist_list.append((s1, e1, s2, e2, dist))
            bottom_idx = np.argsort([dist for s1, e1, s2, e2, dist in dist_list])[-2:]
            bottoms = [dist_list[bottom_idx[0]][:2], dist_list[bottom_idx[1]][:2]]
        else:
            bottoms = [candidate[0][:2], candidate[1][:2]]

    else:
        d1 = norm2(pts[1] - pts[0]) + norm2(pts[2] - pts[3])
        d2 = norm2(pts[2] - pts[1]) + norm2(pts[0] - pts[3])
        bottoms = [(0, 1), (2, 3)] if d1 < d2 else [(1, 2), (3, 0)]
    assert len(bottoms) == 2, 'fewer than 2 bottoms'
    return bottoms

def find_long_edges(points, bottoms):
    b1_start, b1_end = bottoms[0]
    b2_start, b2_end = bottoms[1]
    
    
    n_pts = len(points)
    i = (b1_end + 1) % n_pts
    long_edge_1 = []

    while (i % n_pts != b2_end): #b2_end
        start = (i - 1) % n_pts
        end = i % n_pts
        long_edge_1.append((start, end))
        i = (i + 1) % n_pts

    i = (b2_end + 1) % n_pts
    long_edge_2 = []
    while (i % n_pts != b1_end): #b1_end
        start = (i - 1) % n_pts
        end = i % n_pts
        long_edge_2.append((start, end))
        i = (i + 1) % n_pts
    return long_edge_1, long_edge_2


def split_long_edges(points, bottoms):
    """
    Find two long edge sequence of and polygon
    """
    b1_start, b1_end = bottoms[0]
    b2_start, b2_end = bottoms[1]
    n_pts = len(points)

    i = b1_end + 1
    long_edge_1 = []
    while (i % n_pts != b2_end):
        long_edge_1.append((i - 1, i))
        i = (i + 1) % n_pts

    i = b2_end + 1
    long_edge_2 = []
    while (i % n_pts != b1_end):
        long_edge_2.append((i - 1, i))
        i = (i + 1) % n_pts
    return long_edge_1, long_edge_2

def split_edge_seqence(points, long_edge, n_parts):
    pt_num = len(points)
    long_edge = [(e[0]%pt_num,e[1]%pt_num) for e in long_edge]
    edge_length = [norm2(points[e1] - points[e2]) for e1, e2 in long_edge]
    point_cumsum = np.cumsum([0] + edge_length)
    total_length = sum(edge_length)
    length_per_part = total_length / n_parts

    cur_node = 0  # first point
    splited_result = []

    for i in range(1, n_parts):
        cur_end = i * length_per_part

        while(cur_end > point_cumsum[cur_node + 1]):
            cur_node += 1

        e1, e2 = long_edge[cur_node]
        e1, e2 = points[e1], points[e2]

        # start_point = points[long_edge[cur_node]]
        end_shift = cur_end - point_cumsum[cur_node]
        ratio = end_shift / edge_length[cur_node]
        new_point = e1 + ratio * (e2 - e1)
        # print(cur_end, point_cumsum[cur_node], end_shift, edge_length[cur_node], '=', new_point)
        splited_result.append(new_point)

    # add first and last point
    p_first = points[long_edge[0][0]]
    p_last = points[long_edge[-1][1]]
    splited_result = [p_first] + splited_result + [p_last]
    return np.stack(splited_result)


def get_char_coordinates(splited_result1, splited_result2) :
    '''assume that the splited_result1 is the upper part'''
    assert len(splited_result1) == len(splited_result2)
    
    reverse_splited_result1 = splited_result1[::-1]
    if splited_result2[0][1] < reverse_splited_result1[0][1] :
        splited_result1, splited_result2 = splited_result2, splited_result1
        reverse_splited_result1 = splited_result1
        splited_result2 = splited_result2[::-1]
        
    split_num = len(splited_result1)
    
    char_coords = []
    
    for idx in range(split_num - 1) :
        char_coord = np.array([reverse_splited_result1[idx], 
                               reverse_splited_result1[idx+1],
                               splited_result2[idx+1],
                               splited_result2[idx]
                                    ])
                              
        char_coords.append(char_coord)
                              
    return char_coords


class TextInstance(object):

    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text
        self.checked = False

        remove_points = []

        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area) / ori_area < 0.017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)

    def find_bottom_and_sideline(self):
        self.bottoms = find_bottom(self.points)  # find two bottoms of this Text
        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence

    def disk_cover(self, n_disk=15):
        """
        cover text region with several disks
        :param n_disk: number of disks
        :return:
        """
        self.inner_points1 = split_edge_seqence(self.points, self.e1, n_disk)
        inner_points2 = split_edge_seqence(self.points, self.e2, n_disk)
        self.inner_points2 = inner_points2[::-1]  # innverse one of long edge

        self.center_points = (self.inner_points1 + self.inner_points2) / 2  # disk center
        self.radii = norm2(self.inner_points1 - self.center_points, axis=1)  # disk radius

    def get_coordinates(self) :
        if self.inner_points1[0][1] > self.inner_points2[0][1] :
            self.inner_points1, self.inner_points2 = self.inner_points2, self.inner_points1 
            
        
        
        return np.concatenate([self.inner_points1, self.inner_points2])

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)
