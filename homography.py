"""
Copyright 2023 ground0state
Released under the MIT license
"""
from random import Random
from os.path import join

import os
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression


def get_n_trials(n, p=0.99, w=0.2):
    nume = np.log(1-p)
    deno = np.log(1-np.power(w, n))
    return int(nume / deno) + 1


def get_sift_matches(img1, img2, n_matches=None):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)[:n_matches]
    return matches, keypoints1, keypoints2


def _create_homography_matrix(src_point, dst_point):
    x1, y1 = dst_point
    x2, y2 = src_point
    matrix = [[x2, y2, 1, 0, 0, 0, -x1*x2, -x1*y2],
              [0, 0, 0, x2, y2, 1, -y1*x2, -y1*y2]]
    return matrix, [x1, y1]


def _estimate_homography(
        src_points, dst_points, max_iters=5000, reprojection_threshold=5, seed=0, verbose=0):
    random = Random(seed)

    X, y = zip(*[_create_homography_matrix(src_pt, dst_pt)
                 for src_pt, dst_pt in zip(src_points, dst_points)])
    X = np.concatenate(X).astype(np.float32)
    y = np.concatenate(y).astype(np.float32)

    n_points = len(src_points)
    n_sample = 4
    magic_number = 15
    n_trials = min(max_iters, get_n_trials(n_sample, w=magic_number/n_points))
    if verbose >= 1:
        print("  n_trials:", n_trials)

    n_inlier = 0
    for i_iter in range(n_trials):
        sampled_idx = random.sample(range(n_points), n_sample)
        sampled_X, sampled_y = zip(
            *[_create_homography_matrix(src_points[i], dst_points[i]) for i in sampled_idx])
        sampled_X = np.concatenate(sampled_X)
        sampled_y = np.concatenate(sampled_y)

        estimator = LinearRegression(
            fit_intercept=False).fit(sampled_X, sampled_y)
        y_pred = estimator.predict(X)

        residual_distance = np.linalg.norm(
            y_pred.reshape(-1, 2) - y.reshape(-1, 2), axis=1)

        temp_inlier_mask = residual_distance <= reprojection_threshold
        temp_n_inlier = temp_inlier_mask.sum()

        if temp_n_inlier > n_inlier:
            n_inlier = temp_n_inlier
            inlier_mask = temp_inlier_mask

            if verbose >= 2:
                print(f"  trial_id: {i_iter}, n_inlier_points: {n_inlier}")

    if verbose >= 1:
        print(f"  n_inlier_points: {n_inlier}")

    selected_X, selected_y = zip(*[_create_homography_matrix(
        src_points[i], dst_points[i]) for i in range(len(inlier_mask)) if inlier_mask[i]])

    selected_X = np.concatenate(selected_X)
    selected_y = np.concatenate(selected_y)

    estimator = LinearRegression(
        fit_intercept=False).fit(selected_X, selected_y)
    h = estimator.coef_.copy()
    h = np.concatenate([h, [1]])
    h = h.reshape(3, 3)
    return h


def apply_homography(src_image, dst_image, h):
    result = cv2.warpPerspective(src_image, h, dst_image.shape[:2][::-1])
    return result


def blend_image(src_image, dst_image):
    result = cv2.addWeighted(dst_image, 0.5, src_image, 0.5, 0)
    return result


def find_homography_opencv(
    matches,
    src_keypoints,
    dst_keypoints,
    max_iters=5000,
    reprojection_threshold=5.0,
):
    dst_kpt = np.float32(
        [dst_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    src_kpt = np.float32(
        [src_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    h, mask = cv2.findHomography(
        src_kpt, dst_kpt, cv2.RANSAC,
        ransacReprojThreshold=reprojection_threshold,
        maxIters=max_iters)
    return h

# src_imageをdst_imageに位置合わせしたグレースケール画像を返す
def homography(src_image, dst_image, n_matches, max_iters):
    dst_image_gray = cv2.cvtColor(dst_image, cv2.COLOR_BGR2GRAY)
    src_image_gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)

    matches, dst_keypoints, src_keypoints = get_sift_matches(
        dst_image_gray, src_image_gray, n_matches)

    h = find_homography_opencv(matches, src_keypoints, dst_keypoints,
                               max_iters=max_iters)

    result = apply_homography(src_image, dst_image, h)
    return result
    return cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
