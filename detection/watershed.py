import cv2
import numpy as np
import math
import Polygon as plg


def watershed1(image, viz=False):
    boxes = []
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if viz:
        cv2.imshow("gray", gray)
        cv2.waitKey()
    ret, binary = cv2.threshold(gray, 0.6 * np.max(gray), 255, cv2.THRESH_BINARY)
    if viz:
        cv2.imshow("binary", binary)
        cv2.waitKey()
    # 形态学操作，进一步消除图像中噪点
    kernel = np.ones((3, 3), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)  # iterations连续两次开操作
    sure_bg = cv2.dilate(mb, kernel, iterations=3)  # 3次膨胀,可以获取到大部分都是背景的区域
    if viz:
        cv2.imshow("sure_bg", sure_bg)
        cv2.waitKey()

    # 距离变换
    dist = cv2.distanceTransform(mb, cv2.DIST_L2, 5)
    if viz:
        cv2.imshow("dist", dist)
        cv2.waitKey()
    ret, sure_fg = cv2.threshold(dist, 0.2 * np.max(dist), 255, cv2.THRESH_BINARY)
    surface_fg = np.uint8(sure_fg)  # 保持色彩空间一致才能进行运算，现在是背景空间为整型空间，前景为浮点型空间，所以进行转换
    if viz:
        cv2.imshow("surface_fg", surface_fg)
        cv2.waitKey()
    unknown = cv2.subtract(sure_bg, surface_fg)
    # 获取maskers,在markers中含有种子区域
    ret, markers = cv2.connectedComponents(surface_fg)

    # 分水岭变换
    markers = markers + 1
    markers[unknown == 255] = 0

    if viz:
        color_markers = np.uint8(markers)
        color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)
        cv2.imshow("color_markers", color_markers)
        cv2.waitKey()

    markers = cv2.watershed(image, markers=markers)
    image[markers == -1] = [0, 0, 255]
    if viz:
        cv2.imshow("image", image)
        cv2.waitKey()
    for i in range(2, np.max(markers) + 1):
        np_contours = np.roll(np.array(np.where(markers == i)), 1, axis=0).transpose().reshape(-1, 2)
        # print(np_contours.shape)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)
        boxes.append(box)
    return np.array(boxes)


def getDetCharBoxes_core(textmap, text_threshold=0.5, low_text=0.4):
    # prepare data
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score.astype(np.uint8),
                                                                         connectivity=4)

    det = []
    mapper = []
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        # segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0: sx = 0
        if sy < 0: sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)
        mapper.append(k)

    return det, labels, mapper


def watershed2(image, viz=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray) / 255.0
    boxes, _, _ = getDetCharBoxes_core(gray)
    return np.array(boxes)


def watershed(oriimage, image, viz=False):
    # viz = True
    boxes = []
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    if viz:
        plt.show()
        print('gray')
        plt.imshow(gray, cmap='gray')
    ret, binary = cv2.threshold(gray, 0.2 * np.max(gray), 255, cv2.THRESH_BINARY)
    if viz:
        plt.show()
        print('binary')
        plt.imshow(binary)

    kernel = np.ones((3, 3), np.uint8)
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)  
    sure_bg = cv2.dilate(mb, kernel, iterations=3)  
    sure_bg = mb
    if viz:
        cv2.imshow("sure_bg", mb)
        cv2.waitKey()
    if viz:
        plt.show()
        print('morphology')
        plt.imshow(mb)
    ret, sure_fg = cv2.threshold(gray, 0.6 * gray.max(), 255, cv2.THRESH_BINARY)
    surface_fg = np.uint8(sure_fg) 
    if viz:
        plt.show()
        print('surface_fg')
        plt.imshow(surface_fg)
    unknown = cv2.subtract(sure_bg, surface_fg)
    ret, markers = cv2.connectedComponents(surface_fg)

    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(surface_fg,
                                                                         connectivity=4)
    
    markers = labels.copy() + 1
    markers[unknown == 255] = 0

    if viz:
        color_markers = np.uint8(markers)
        color_markers = color_markers / (color_markers.max() / 255)
        color_markers = np.uint8(color_markers)
        color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)
        plt.show()
        print('applyColormap')
        plt.imshow( color_markers)
    # a = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    markers = cv2.watershed(oriimage, markers=markers)
    oriimage[markers == -1] = [0, 0, 255]

    if viz:
        color_markers = np.uint8(markers + 1)
        color_markers = color_markers / (color_markers.max() / 255)
        color_markers = np.uint8(color_markers)
        color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)
        plt.show()
        print('color markers')
        plt.imshow(color_markers)

    if viz:
        plt.show()
        print('image')
        plt.imshow( oriimage)
    for i in range(2, np.max(markers) + 1):
        np_contours = np.roll(np.array(np.where(markers == i)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        poly = plg.Polygon(box)
        area = poly.area()
        if area < 10:
            continue
        box = np.array(box)
        boxes.append(box)
    return np.array(boxes)

def watershed_wo_net(oriimage, image, viz=False, savename = None):
    # viz = True
    boxes = []
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    if viz:
        plt.imshow(gray, cmap='gray')
        print('gray')
        plt.show()
    ret, binary = cv2.threshold(gray, 0 * np.max(gray), 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #CHANGE
    if viz:
        plt.imshow(binary)
        print('binary')
        plt.show()        
    kernel = np.ones((3, 3), np.uint8) #CHANGE
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)  # iterations连续两次开操作
    mb = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    sure_bg = cv2.dilate(mb, kernel, iterations=1)  # 3次膨胀,可以获取到大部分都是背景的区域
#     sure_bg = mb
    if viz:
        plt.imshow(mb)
        print('mb')
        plt.show()
#         cv2.waitKey()
    # 距离变换
    dist = cv2.distanceTransform(mb, cv2.DIST_L2, 5) # https://webnautes.tistory.com/1280 중앙점에서의 거리를 기준으로 픽셀값 변화
    if viz:
        plt.imshow(dist)
        print('dist')
        plt.show()
    ret, sure_fg = cv2.threshold(dist, 0.2 * dist.max(), 255, cv2.THRESH_BINARY)
    surface_fg = np.uint8(sure_fg) 
    if viz:
        plt.imshow(surface_fg)
        print('surface_fg')
        plt.show()
    unknown = cv2.subtract(sure_bg, surface_fg)
    ret, markers = cv2.connectedComponents(surface_fg)
    markers = markers + 1
    markers[unknown==255] = 0


    if viz:
        color_markers = np.uint8(markers)
        color_markers = color_markers / (color_markers.max() / 255)
        color_markers = np.uint8(color_markers)
        color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)
        plt.imshow(color_markers)
        print('color_markers')
        plt.show()

    markers = cv2.watershed(oriimage, markers=markers)
    oriimage[markers == -1] = [0, 0, 255]

    if viz:
        plt.imshow(oriimage)
        print('oriimage')
        plt.show()
#         cv2.waitKey()
    ### 저장 부분
    # canvas = Image.new('L', image.shape[:2][::-1])
    # draw = ImageDraw.Draw(canvas)
    bbox_height, bbox_width = gray.shape[:2]
    for i in range(2, np.max(markers) + 1):
        np_contours = np.roll(np.array(np.where(markers == i)), 1, axis=0).transpose().reshape(-1, 2)
        segmap = np.zeros(gray.shape, dtype=np.uint8)
        segmap[markers == i] = 255
        size = np_contours.shape[0]
        x, y, w, h = cv2.boundingRect(np_contours)
        if w == 0 or h == 0 or w*h < bbox_width*bbox_height*0.001:
            continue
        
#         niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
#         sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
#         # boundary check
#         if sx < 0: sx = 0
#         if sy < 0: sy = 0
#         if ex >= gray.shape[1]: ex = gray.shape[1]
#         if ey >= gray.shape[0]: ey = gray.shape[0]
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
#         segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
#         np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        
        
        
#         rectangle = cv2.minAreaRect(np_contours)
#         box = cv2.boxPoints(rectangle)

#         # draw.polygon(np.roll(box, -1, axis = 0), outline = (255))
        
#         startidx = box.sum(axis=1).argmin()
#         box = np.roll(box, 4 - startidx, 0)
        box = np.array([[x,y],
                       [x+w,y],
                       [x+w,y+h],
                       [x,y+h]])
        
        xs = np.where(box[:,0]<0,0, box[:,0])# x좌표
        xs = np.where(box[:,0]>bbox_width,bbox_width, box[:,0])# x좌표
        
        ys = np.where(box[:,1]<0,0, box[:,1])
        ys = np.where(box[:,1]>bbox_height,bbox_height, box[:,1])
        
        box = np.stack([xs,ys], -1)

        boxes.append(box)
    
    boxes = np.array(boxes)

    padding_x = 5
    if len(boxes) != 0  : 
        sorted_idx = np.argsort(boxes[:,0,0])
        bboxes = boxes[sorted_idx]

        threshold = 0.4

        group = defaultdict(list)
        grp_idx = 0
        group[grp_idx].append(0)
        previous_box = bboxes[0]
        prev_xmin, prev_xmax = previous_box[0][0], previous_box[1][0]
        for i in range(1, len(bboxes)) :
            
            current_box = bboxes[i]
            cur_xmin, cur_xmax = current_box[0][0], current_box[1][0]
            cur_ymin, cur_ymax = current_box[0][1], current_box[2][1]
            
            overlap_rate = (min(prev_xmax, cur_xmax) - max(prev_xmin, cur_xmin - padding_x)) \
                                / min(prev_xmax - prev_xmin, cur_xmax - cur_xmin)
            
            if overlap_rate > threshold or ((cur_ymax - cur_ymin)/(cur_xmax - cur_xmin) > 3 and (cur_xmax - cur_xmin) < bbox_width*0.01): #자음
                group[grp_idx].append(i)
                prev_xmin, prev_xmax = prev_xmin, cur_xmax
            else :
                grp_idx += 1
                group[grp_idx].append(i)
                prev_xmin, prev_xmax = cur_xmin, cur_xmax

        new_bboxes = [] 

        for grp in group.values() :
            
            # 
            
            xmin, xmax = np.min(bboxes[grp][:,0,0]), np.max(bboxes[grp][:,1,0])
            ymin, ymax = np.min(bboxes[grp][:,0,1]), np.max(bboxes[grp][:,2,1])

            new_box = np.array([[xmin, ymin],
                                [xmax, ymin],
                                [xmax, ymax],
                                [xmin, ymax]])

            new_bboxes.append(new_box)

        return np.array(new_bboxes)
    
    else :
        return boxes



if __name__ == '__main__':
    image = cv2.imread('images/standard.jpg', cv2.IMREAD_COLOR)
    boxes = watershed(image, True)
    print(boxes)