import math
import sys

import maxflow
import nibabel as nib
import numpy as np
import scipy.ndimage.morphology as morph

from remove_backgr_mouse import grow_middle


def histogram_remove(img):
    i2 = np.zeros(img.shape)


def combine_image(images):
    res = None
    for i in images:
        if res is None:
            res = i.astype(int)
        else:
            res = res + i.astype(int)
    res = res / len(images)
    return res


def distance_metrics(mask):
    res_dist = np.zeros(mask.shape)
    # (l,w,h) = header.get_zooms()

    tmp_0 = mask.copy()
    tmp_1 = mask.copy()
    cnt = 1
    while True:
        tmp_1 = morph.binary_erosion(tmp_0)
        tmp_0 = np.logical_xor(tmp_1, tmp_0)
        if not tmp_0.max():
            break
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                for k in range(mask.shape[2]):
                    if tmp_0[i, j, k]:
                        res_dist[i, j, k] = cnt
        cnt = cnt + 1
        tmp_0 = tmp_1.copy()
    return res_dist


def var8(Mat, x, y, z, size):
    indices = np.zeros(8)
    indices[0] = Mat[x, y, z]
    indices[1] = Mat[x, y, z + size]
    indices[2] = Mat[x, y + size, z]
    indices[3] = Mat[x, y + size, z + size]
    indices[4] = Mat[x + size, y, z]
    indices[5] = Mat[x + size, y, z + size]
    indices[6] = Mat[x + size, y + size, z]
    indices[7] = Mat[x + size, y + size, z + size]
    return indices.var()


def seed_select(img, mean_thre, var_thre):
    variance8V = 0
    weightedVariance = 100000

    tempX = 0
    tempY = 0
    tempZ = 0
    tempMean = 0
    tempVariance = 0
    size = 5
    qualifiedFlag = True

    shortlist_mean = np.zeros(10)
    shortListVariance = np.full(10, 100000)
    shortListSeedX = np.full(10, -1)
    shortListSeedY = np.full(10, -1)
    shortListSeedZ = np.full(10, -1)
    neibourghood6 = [[size, 0, 0], [-size, 0, 0], [0, size, 0], [0, -size, 0], [0, 0, size], [0, 0, -size]]

    for i in range(size, img.shape[0] - size, size):
        for j in range(size, img.shape[1] - size, size):
            for k in range(size, img.shape[2] - size, size):
                cube = img[i:i + size, j:j + size, k:k + size]
                mean5x5 = cube.mean()
                variance5x5 = cube.var()
                v8 = var8(cube, 0, 0, 0, size - 1)

                if (mean5x5 > mean_thre) & (variance5x5 < var_thre):
                    for l in range(6):
                        tempX = i + neibourghood6[l][0]
                        tempY = j + neibourghood6[l][1]
                        tempZ = k + neibourghood6[l][2]
                        if ((tempX >= img.shape[0] - size)
                                | (tempY >= img.shape[1] - size)
                                | (tempZ >= img.shape[2] - size)):
                            qualifiedFlag = False
                            break
                        t_cube = img[tempX:tempX + size, tempY: tempY + size, tempZ:tempZ + size]
                        tempMean = t_cube.mean()
                        tempVariance = t_cube.var()
                        if (~((tempMean > mean_thre) &
                              (tempVariance < var_thre))):
                            qualifiedFlag = False
                            break
                        else:
                            qualifiedFlag = True

                    if qualifiedFlag:
                        weightedVariance = 0.5 * variance5x5 + 0.5 * variance8V

                        for m in range(10):
                            if shortListVariance[m] >= weightedVariance:
                                n = 9
                                while n > m:
                                    shortlist_mean[n] = shortlist_mean[n - 1]
                                    shortListVariance[n] = shortListVariance[n - 1]
                                    shortListSeedX[n] = shortListSeedX[n - 1]
                                    shortListSeedY[n] = shortListSeedY[n - 1]
                                    shortListSeedZ[n] = shortListSeedZ[n - 1]
                                    n = n - 1
                                shortlist_mean[m] = mean5x5
                                shortListVariance[m] = weightedVariance
                                shortListSeedX[m] = i
                                shortListSeedY[m] = j
                                shortListSeedZ[m] = k
                                break

    lowestMean = 10000000
    target = 0
    for k in range(10):
        if (shortlist_mean[k] < lowestMean):
            lowestMean = shortlist_mean[k]
            target = k

    seedX = shortListSeedX[target]
    seedY = shortListSeedY[target]
    seedZ = shortListSeedZ[target]
    return [seedX, seedY, seedZ]


def Sub2Ind3D(sX, sY, sZ, iSizeX, iSizeY):
    ind = sX + (sY * iSizeX) + (sZ * iSizeX * iSizeY)
    return ind


def region_growing(img, seed, lmdT, umdT, nmdT, vT):
    ptrLMeanDiffThreshold = lmdT
    ptrUMeanDiffThreshold = umdT
    ptrNMeanDiffThreshold = nmdT
    ptrVarianceThreshold = vT

    (iSizeX, iSizeY, iSizeZ) = img.shape

    iSeedPosX = seed[0]
    iSeedPosY = seed[1]
    iSeedPosZ = seed[2]

    ptrVisited = np.zeros(img.shape)
    MLabel = np.zeros(img.shape)
    cube = img[iSeedPosX:iSeedPosX + 3, iSeedPosY:iSeedPosY + 3, iSeedPosZ:iSeedPosZ + 3]
    meanSeed = cube.mean()
    varianceSeed = cube.var()
    variance8VSeed = var8(cube, 0, 0, 0, 3 - 1)

    indexSeed = Sub2Ind3D(iSeedPosX, iSeedPosY, iSeedPosZ, iSizeX, iSizeY)
    seedNode = [iSeedPosZ, iSeedPosY, iSeedPosZ, indexSeed, meanSeed, varianceSeed, variance8VSeed]

    ptrQueue = []
    ptrQueue.append(seedNode)

    MLabel[iSeedPosX:iSeedPosX + 3, iSeedPosY:iSeedPosY + 3, iSeedPosZ:iSeedPosZ + 3] = 1
    ptrVisited[iSeedPosX:iSeedPosX + 3, iSeedPosY:iSeedPosY + 3, iSeedPosZ:iSeedPosZ + 3] = 1

    minMean = meanSeed - ptrLMeanDiffThreshold
    maxMean = meanSeed + ptrUMeanDiffThreshold

    ptrCurrentNode = ptrQueue.pop()

    meanCurrentNode = 0
    varianceCurrentNode = 0
    variance8VCurrentNode = 0
    subXCurrentNode = 0
    subYCurrentNode = 0
    subZCurrentNode = 0
    indexCurrentNode = 0

    meanNeighborNode = 0
    varianceNeighborNode = 0
    variance8VNeighborNode = 0
    subXNeighborNode = 0
    subYNeighborNode = 0
    subZNeighborNode = 0
    indexNeighborNode = 0

    sixNeighbourOffset = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]

    while ptrCurrentNode != None:
        meanCurrentNode = ptrCurrentNode[4]
        subXCurrentNode = ptrCurrentNode[0]
        subYCurrentNode = ptrCurrentNode[1]
        subZCurrentNode = ptrCurrentNode[2]

        for n1 in range(6):
            subXNeighborNode = subXCurrentNode + sixNeighbourOffset[n1][0]
            subYNeighborNode = subYCurrentNode + sixNeighbourOffset[n1][1]
            subZNeighborNode = subZCurrentNode + sixNeighbourOffset[n1][2]

            if ((subXNeighborNode <= 3) |
                    (subXNeighborNode >= iSizeX - 3) |
                    (subYNeighborNode <= 3) |
                    (subYNeighborNode >= iSizeY - 3) |
                    (subZNeighborNode <= 3) |
                    (subZNeighborNode >= iSizeZ - 3)):  # skip the pixel out of the image
                continue
            indexNeighborNode = Sub2Ind3D(subXNeighborNode,
                                          subYNeighborNode,
                                          subZNeighborNode,
                                          iSizeX, iSizeY)
            if ptrVisited[subXNeighborNode][subYNeighborNode][subZNeighborNode] != 1:
                cube_t = img[subXNeighborNode:subXNeighborNode + 3, subYNeighborNode:subYNeighborNode + 3,
                         subZNeighborNode:subZNeighborNode + 3]
                meanNeighborNode = cube_t.mean()
                varianceNeighborNode = cube_t.var()

                variance8VNeighborNode = var8(cube_t, 0, 0, 0, 2)

                ptrVisited[subXNeighborNode:subXNeighborNode + 3,
                subYNeighborNode:subYNeighborNode + 3,
                subZNeighborNode:subZNeighborNode + 3] = 1

                if ((math.fabs(meanCurrentNode - meanNeighborNode) < ptrNMeanDiffThreshold)
                        & (varianceNeighborNode < ptrVarianceThreshold)
                        & (meanNeighborNode > minMean)
                        & (meanNeighborNode < maxMean)):
                    meanNeighborNode = (meanNeighborNode + meanCurrentNode) / 2
                    ptrCurrentNode = [subXNeighborNode,
                                      subYNeighborNode,
                                      subZNeighborNode,
                                      indexNeighborNode,
                                      meanNeighborNode,
                                      varianceNeighborNode,
                                      variance8VNeighborNode]
                    ptrQueue.append(ptrCurrentNode)

                    MLabel[subXNeighborNode:subXNeighborNode + 3,
                    subYNeighborNode:subYNeighborNode + 3,
                    subZNeighborNode:subZNeighborNode + 3] = 1
        if len(ptrQueue) > 0:
            ptrCurrentNode = ptrQueue.pop()
        else:
            ptrCurrentNode = None

    return MLabel


def preprocessing(img):
    result = np.zeros(img.shape)
    # copy from realisation
    mean_thre = 0.45 * 160
    varianceThreshold = 0.004 * 160 * 160

    seed = seed_select(img, mean_thre, varianceThreshold)

    repetition = 1
    while ((seed[0] == -1) | (seed[1] == -1) | (seed[2] == -1)):
        mean_thre = mean_thre - 0.05 * 160
        seed = seed_select(img, mean_thre, varianceThreshold)
        repetition = repetition + 1
        if (repetition >= 8):
            break

    if seed[0] == -1 | seed[1] == -1 | seed[2] == -1:
        # failed
        return -1

    seed[0] = seed[0] + 2
    seed[1] = seed[1] + 2
    seed[2] = seed[2] + 2

    meanDiffThreshold = 0.03 * 160

    varianceDiffThreshold = 0.004 * 160 * 160

    lmdT = 0.05 * 160

    umdT = 0.25 * 160

    label = region_growing(img, seed, lmdT, umdT, meanDiffThreshold, varianceDiffThreshold)

    cube = img[label == 1]
    print("white mean : " + str(cube.mean()) + " seed num" + str(len(cube)))

    return cube.mean()


def block_dist(img):
    res_dist = np.zeros(img.shape)
    (l, w, h) = img.shape
    res_dist[img > 0] = -1
    # mark borders
    res_dist[0, :, :] = 0
    res_dist[:, 0, :] = 0
    res_dist[:, :, 0] = 0
    res_dist[l - 1, :, :] = 0
    res_dist[:, w - 1, :] = 0
    res_dist[:, :, h - 1] = 0

    # (l,w,h) = header.get_zooms()

    for x in range(19):
        for i in range(1, l - 1):
            for j in range(1, w - 1):
                for k in range(1, h - 1):
                    # upgrade 8neib
                    if res_dist[i, j, k] == -1:
                        if ((res_dist[i, j, k - 1] == x) | (res_dist[i, j, k + 1] == x) |
                                (res_dist[i, j - 1, k] == x) | (res_dist[i, j + 1, k] == x) |
                                (res_dist[i - 1, j, k] == x) | (res_dist[i + 1, j, k] == x)):
                            res_dist[i, j, k] = x + 1
    for i in range(l):
        for j in range(w):
            for k in range(h):
                if res_dist[i, j, k] == -1:
                    res_dist[i, j, k] = 20
                if res_dist[i, j, k] == 0:
                    res_dist[i, j, k] = 1
    return res_dist


def decide_bound(img, threshold):
    factor = 1.2
    tf = factor * threshold
    x_start, y_start, z_start = 0, 0, 0
    x_end = img.shape[0] - 1
    y_end = img.shape[1] - 1
    z_end = img.shape[2] - 1
    # xstart
    bExit = 0
    x, y, z = 0, 0, 0
    while ((x < img.shape[0]) & (bExit == 0)):
        while ((y < img.shape[1]) & (bExit == 0)):
            while ((z < img.shape[2]) & (bExit == 0)):
                if img[x, y, z] > tf:
                    x_start = x
                    bExit = 1
                z = z + 1
            y = y + 1
        x = x + 1

    # xend
    bExit = 0
    x, y, z = img.shape[0] - 1, 0, 0
    while ((x > x_start) & (bExit == 0)):
        while ((y < img.shape[1]) & (bExit == 0)):
            while ((z < img.shape[2]) & (bExit == 0)):
                if img[x, y, z] > tf:
                    x_end = x
                    bExit = 1
                z = z + 1
            y = y + 1
        x = x - 1

    # y_start
    bExit = 0
    x, y, z = 0, 0, 0
    while ((y < img.shape[1]) & (bExit == 0)):
        while ((x < img.shape[0]) & (bExit == 0)):
            while ((z < img.shape[2]) & (bExit == 0)):
                if img[x, y, z] > tf:
                    y_start = y
                    bExit = 1
                z = z + 1
            x = x + 1
        y = y + 1

    # yend
    bExit = 0
    x, y, z = 0, img.shape[1] - 1, 0
    while ((y > y_start) & (bExit == 0)):
        while ((x < img.shape[1]) & (bExit == 0)):
            while ((z < img.shape[2]) & (bExit == 0)):

                if img[x, y, z] > tf:
                    y_end = y
                    bExit = 1
                z = z + 1
            x = x + 1
        y = y - 1

    # z_start
    bExit = 0
    x, y, z = 0, 0, 0
    while ((z < img.shape[2]) & (bExit == 0)):
        while ((x < img.shape[0]) & (bExit == 0)):
            while ((y < img.shape[1]) & (bExit == 0)):
                if img[x, y, z] > tf:
                    z_start = z
                    bExit = 1
                y = y + 1
            x = x + 1
        z = z + 1

    # zend
    bExit = 0
    x, y, z = 0, 0, img.shape[2] - 1
    while ((z > z_start) & (bExit == 0)):
        while ((x < img.shape[0]) & (bExit == 0)):
            while ((y < img.shape[1]) & (bExit == 0)):
                if img[x, y, z] > tf:
                    z_end = z
                    bExit = 1
                y = y + 1
            x = x + 1
        z = z - 1

    return [x_start, x_end, y_start, y_end, z_start, z_end]


def max_flow(weights, Fw, Bw):
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            for k in range(weights.shape[2]):
                pass


def gc_method(img,
              narrow_mask,
              init_mask,
              threshold,
              wm,
              k_v):
    # ===============PREPROC_ get
    # get overlap window to not process the all dataset (maybe)
    print(1)
    # img = (img - img.min()) / (img.max() - img.min()) # scale to [0,1]
    # di = 1/256
    # for i in range(256):
    #     if i < 255:
    #         img[(img>= di*i) & (img <di*(i+1))] = i
    #     else:
    #         img[(img >= di * i) & (img <= di * (i + 1))] = i
    # [x0, x1, y0, y1, z0, z1] = decide_bound(img,threshold)
    # calculate mean in narrow mask (in paper it's WM)
    # WM_mean = wm


    # WM FILTER
    WM_mean = (np.ma.masked_array(img, mask=~narrow_mask)).mean()
    thr = 0.6 * WM_mean
    img[img < (thr + 1)] = 0
    # Calculate outside mean value (paper threshold)

    thr = (np.ma.masked_array(img, mask=init_mask)).mean()

    # thr = threshold
    k_val = k_v / (WM_mean - thr)

    # [x0, x1, y0, y1, z0, z1] = decide_bound(img, thr)

    # get label > thr
    label = img > (thr * 1.26)
    # x0, x1, y0, y1, z0, z1 = np.where(label == True)[0].min(), np.where(label == True)[0].max(), \
    #                          np.where(label == True)[1].min(), np.where(label == True)[1].max(), \
    #                          np.where(label == True)[2].min(), np.where(label == True)[2].max()
    x0, x1, y0, y1, z0, z1 = np.where(init_mask == True)[0].min(), np.where(init_mask == True)[0].max(), \
                             np.where(init_mask == True)[1].min(), np.where(init_mask == True)[1].max(), \
                             np.where(init_mask == True)[2].min(), np.where(init_mask == True)[2].max()
    l, w, h = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
    # GET DISTANCE METRICS

    # dist_marker = distance_metrics(narrow_mask)
    dist_marker = block_dist(img)
    # Calculate FOREGROUND AND BACKGROUND SEED

    Fw, Bw = np.zeros(shape=img.shape), np.zeros(shape=img.shape)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                # if not init_mask[i, j, k]:
                #     Bw[i, j, k] = 4000
                if narrow_mask[i, j, k]:
                    Fw[i, j, k] = 4000#infinity foreground
                elif img[i, j, k] <= 0:
                    Bw[i, j, k] = 4000#0 = background
                pass

    # --------------------- Calculate weights ----------------------
    # 3x3 neibroughoud

    #
    weights = np.zeros((l, w, h,
                        3,  # we assing weight to point from left to right from
                        # bottom to top  (x,y,z,(x+1), ) weight between point and point
                        ))

    for i in range(l):
        for j in range(w):
            for k in range(h):
                #             x + 1
                # euclid
                if i + x0 + 1 < x1:

                    weights[i, j, k, 0] = (max(
                        [dist_marker[i + x0, j + y0, k + z0], dist_marker[i + x0 + 1, j + y0, k + z0]])) ** 2
                    if (weights[i, j, k, 0] > 1) & (weights[i, j, k, 0] < 6):
                        weights[i, j, k, 0] = 6
                    if (weights[i, j, k, 0] != 1) & (weights[i, j, k, 0] != 6) & (weights[i, j, k, 0] != 0):
                        # narrow_mask[i + x0, j + y0, k + z0] == False):  # maybe need to add smth else

                        # set weight for 0weight
                        # if weights[i, j, k, 0] == 0:
                        #     weights[i, j, k, 0] = 0.5
                        t_val = min([img[i + x0, j + y0, k + z0], img[i + x0 + 1, j + y0, k + z0]])
                        weights[i, j, k, 0] = weights[i, j, k, 0] * abs(math.exp(k_val * (t_val - thr)) - 1)
                    if (weights[i, j, k, 0] > 1) & (weights[i, j, k, 0] < 6):
                        weights[i, j, k, 0] = 6
                    if (weights[i, j, k, 0] > 0) & (weights[i, j, k, 0] < 1):
                        weights[i, j, k, 0] = 1

                    if weights[i, j, k, 0] == 0:
                        weights[i, j, k, 0] = 1000
                else:#dummy
                    weights[i, j, k, 0] = -1

                ###########################################################################
                # ------------------ y + 1 -------------------------------------------------
                if j + y0 + 1 < y1:
                    weights[i, j, k, 1] = (max(
                        [dist_marker[i + x0, j + y0, k + z0], dist_marker[i + x0, j + y0 + 1, k + z0]])) ** 2
                    if (weights[i, j, k, 1] > 1) & (weights[i, j, k, 1] < 6):
                        weights[i, j, k, 1] = 6
                    if (weights[i, j, k, 1] != 1) & (weights[i, j, k, 1] != 6) & (weights[i, j, k, 1] != 0):
                        # set weight for 0weight
                        # if weights[i, j, k, 1] == 0:
                        #     weights[i, j, k, 1] = 0.5
                        t_val = min([img[i + x0, j + y0, k + z0], img[i + x0, j + y0 + 1, k + z0]])
                        weights[i, j, k, 1] = weights[i, j, k, 1] * abs(math.exp(k_val * (t_val - thr)) - 1)
                    if (weights[i, j, k, 1] > 1) & (weights[i, j, k, 1] < 6):
                        weights[i, j, k, 1] = 6
                    if (weights[i, j, k, 1] > 0) & (weights[i, j, k, 1] < 1):
                        weights[i, j, k, 1] = 1
                    if weights[i, j, k, 1] == 0:
                        weights[i, j, k, 1] = 1000
                else:
                    weights[i, j, k, 1] = -1
                # ##########################################################
                #             z + 1
                # lase el hiding
                if k + z0 + 1 < z1:

                    weights[i, j, k, 2] = (max(
                        [dist_marker[i + x0, j + y0, k + z0], dist_marker[i + x0, j + y0, k + z0 + 1]])) ** 2
                    if (weights[i, j, k, 2] > 1) & (weights[i, j, k, 2] < 6):
                        weights[i, j, k, 2] = 6
                    if (weights[i, j, k, 2] != 1) & (weights[i, j, k, 2] != 6) & (weights[i, j, k, 2] != 0):

                        # set weight for 0weight
                        if weights[i, j, k, 2] == 0:
                            weights[i, j, k, 2] = 0.5
                        t_val = min([img[i + x0, j + y0, k + z0], img[i + x0, j + y0, k + z0 + 1]])
                        weights[i, j, k, 2] = weights[i, j, k, 2] * abs(math.exp(k_val * (t_val - thr)) - 1)
                    if (weights[i, j, k, 2] > 1) & (weights[i, j, k, 2] < 6):
                        weights[i, j, k, 2] = 6
                    if (weights[i, j, k, 2] > 0) & (weights[i, j, k, 2] < 1):
                        weights[i, j, k, 2] = 1
                    if weights[i, j, k, 2] == 0:
                        weights[i, j, k, 2] = 1000
                else:
                    weights[i, j, k, 2] = -1
    # BUILD EDGES TO PIPELINE

    # gr = graph1.
    graph1 = maxflow.Graph[float](img.shape[1] ** 3, img.shape[1] ** 3)

    # grid = graph.add_grid_nodes(img.shape)

    # length,width,height = img.shape[0],img.shape[1],img.shape[2]
    grid_ids = graph1.add_grid_nodes((l, w, h))

    for i in range(l):
        for j in range(w):
            for k in range(h):
                ind = grid_ids[i, j, k]

                graph1.add_tedge(ind, Bw[i + x0, j + y0, k + z0], Fw[i + x0, j + y0, k + z0], )
                if i != (l - 1):
                    right = grid_ids[i + 1, j, k]
                    graph1.add_edge(ind, right, weights[i, j, k, 0], weights[i, j, k, 0])
                if j != (w - 1):
                    bottom = grid_ids[i, j + 1, k]
                    graph1.add_edge(ind, bottom, weights[i, j, k, 1], weights[i, j, k, 1])
                if k != (h - 1):
                    forward = grid_ids[i, j, k + 1]
                    graph1.add_edge(ind, forward, weights[i, j, k, 2], weights[i, j, k, 2])

    graph1.maxflow()
    #

    sgm = graph1.get_grid_segments(grid_ids)
    res = np.zeros(img.shape)
    for i in range(l):
        for j in range(w):
            for k in range(h):
                if ~(sgm[i, j, k] == False):
                    res[i + x0, j + y0, k + z0] = 1
                else:
                    res[i + x0, j + y0, k + z0] = 0
    return [res,[weights,Bw,Fw]]
    # nodeids = graph.add_nodes(length*width*height)

    # # add grid
    # res_MF = graph.maxflow()
    # I_o = graph.get_grid_segments(grid)

    pass


def post_processing(img,num_dil):
    x0, x1, y0, y1, z0, z1 = np.where(img == True)[0].min(), np.where(img == True)[0].max(), \
                             np.where(img == True)[1].min(), np.where(img == True)[1].max(), \
                             np.where(img == True)[2].min(), np.where(img == True)[2].max()

    a = [l10,l11,l20,l21,l30,l31] = [x0, img.shape[0]-x1,y0, img.shape[1]-y1,z0, img.shape[2]-z1]
    mn = min(a)
    res = img.copy()
    res = morph.binary_fill_holes(res)
    if mn < num_dil:

        r1 = img.copy()
        r1[num_dil:-num_dil,num_dil:-num_dil,num_dil:-num_dil] = 0

        r2 = img.copy()
        r2[:num_dil,:,:] = 0
        r2[-num_dil:, :, :] = 0
        r2[:, :num_dil, :] = 0
        r2[:, -num_dil:, :] = 0
        r2[:, :, :num_dil] = 0
        r2[:, :, -num_dil:] = 0

        r2 = morph.binary_dilation(r2,iterations=num_dil)
        r2 = morph.binary_erosion(r2,iterations=num_dil)
        r1[num_dil:-num_dil,num_dil:-num_dil,num_dil:-num_dil] = r2[num_dil:-num_dil,num_dil:-num_dil,num_dil:-num_dil]
        res = r1
    else:
        res = morph.binary_dilation(res,iterations=num_dil)
        res = morph.binary_erosion(res,iterations=num_dil)

    return res






if __name__ == "__main__":
    im_file = nib.load(sys.argv[1])

    # im_seg = nib.load(sys.argv[2])
    res_path = sys.argv[2]
    out = sys.argv[3]

    img = im_file.get_fdata()
    #


    # whitemean = preprocessing(img)
    _t = 0.4
    # threshold = whitemean * _t
    # print("threshold set to: %f*%f=%f\n", whitemean, _t, threshold)

    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         for k in range(img.shape[2]):
    #             if img[i, j, k] < threshold + 1:
    #                 img[i, j, k] = 0

    # nif = nib.Nifti1Image(img.astype(np.int), im_file.affine)
    # nib.save(nif, res_path + '/' + "rescaled256.nii.gz")


    init_res = grow_middle(img,[0.5,4],0.9,2)
    # nif = nib.Nifti1Image(init_res.astype(np.int), im_file.affine)
    # nib.save(nif, res_path + '/' + "ini222343tmask.nii.gz")
    init_mask = init_res[0] > 0  # 0
    init_fore = init_res[1] > 0  # 0.8
    img = (img - img.min()) / (img.max() - img.min())  # scale to [0,255]
    # di = 1 / 256
    # for i in range(256):
    #     if i < 255:
    #         img[(img >= di * i) & (img < di * (i + 1))] = i
    #     else:
    #         img[(img >= di * i) & (img <= di * (i + 1))] = i
    img = img*255
    nif = nib.Nifti1Image(init_mask.astype(np.int), im_file.affine)
    nib.save(nif, res_path + '/' + "init_fore.nii.gz")
    nif = nib.Nifti1Image(init_fore.astype(np.int), im_file.affine)
    nib.save(nif, res_path + '/' + "init_mask.nii.gz")
    imares = gc_method(img, init_fore, init_mask, 0, 0, -2)
    imares2 = post_processing(imares[0],5)
    nif = nib.Nifti1Image(imares2.astype(np.int), im_file.affine)
    nib.save(nif, res_path + '/' + out)

    nif = nib.Nifti1Image(imares[1][1].astype(np.int), im_file.affine)
    nib.save(nif, res_path + '/' + "BW.nii.gz")
    nif = nib.Nifti1Image(imares[1][2].astype(np.int), im_file.affine)
    nib.save(nif, res_path + '/' + "Wm.nii.gz")

    # imares = sg.morphological_chan_vese(img, iterations=2, init_level_set=imares > 0)
    # imares = morph.binary_fill_holes(imares)
    # imares = morph.binary_erosion(imares)
    # imares = morph.binary_dilation(imares)
    # imares = imares.astype(np.int8)
    #
    #
    # nif = nib.Nifti1Image(imares, im_file.affine)
    # nib.save(nif, res_path + '/output.nii.gz')

    print(1)
