#!/usr/bin/env python3
import sys
import cv2 as cv
import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata
from os import listdir
from os.path import isfile, join
import csv
import datetime

position_changed = True
pts_selection = 1
marker_position = 1
pts1 = dict()
pts1 = {1: [242, 157], 2: [838, 152], 3: [233, 726], 4: [845, 733]}
pts2 = dict()
pts2 = {1: [235, 581], 2: [234, 278], 3: [509, 564], 4: [499, 294], 5:[0,0]}

def get_map_label(pts_file):
    '''
    Loads a csv textfile containing a maps reference points and
    returns the first line of the file as the "label" for the map
    '''
    print("Getting label for "+str(pts_file)+" ...")
    label = None
    try:
        f = open(pts_file,"r")
        lines = f.readlines()
        label = lines[0].split(',')[0]
    
        print(f"Map Label: {label}")
    except:
        print("GET LABEL FAILED")
    
    return label

def import_reference_pts(pts_file):
    '''
    Loads a csv textfile into a dictionary, skipping the first line
    Line format should be "ref point label, x value, y value"
    '''
    point_dict = dict()
    print("Importing "+str(pts_file)+" ...")
    try:
        f = open(pts_file,"r")
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            data = line.split(',')
            # print(data)
            # marker = int(data[0])
            marker = str(data[0])
            x = float(data[1])
            y = float(data[2][:-1])
            coords = [x,y]
            point_dict[marker] = coords
        print("Imported reference point dict:")
        print(point_dict)
    except:
        print("PTS IMPORT FAILED")
    
    return point_dict

def export_reference_pts(pts_dict,name,dir=""):
    pass
    try:
        filename = name+".pts"
        filepath = os.path.join(dir,filename)
    except:
        print("Error exporting points.")
        return
    f = open(filepath,"w+")
    f.write(name+"\n")
    for p in sorted(pts_dict.keys()):
        write_str = str(p)+","+str(pts_dict[p][0])+","+str(pts_dict[p][1])+"\n"
        f.write(write_str)
    f.close()
    print("POINT DICT EXPORTED TO " + filename)
    return

    
def get_scale_factor(pix,mm):
    '''Set scaling factor by providing pixels and the equivalent distance in millimeters'''
    global scale
    print("\nSetting scale")
    print(str(pix)+"px == "+str(mm)+"mm")
    scale = float(mm) / float(pix)
    print("Scaling factor = "+str(scale))
    return scale

def scale_image_with_aspect_ratio(image, scale_factor):
    if image is None:
        raise ValueError("Image not found or cannot be read.")

    # Get the original image dimensions
    original_height, original_width = image.shape[:2]

    # Calculate the new dimensions while preserving the aspect ratio
    new_height = int(original_height * scale_factor)
    new_width = int(original_width * scale_factor)

    # Resize the image using the calculated dimensions
    resized_image = cv.resize(image, (new_width, new_height))
    # cv.imshow("Orig",image)
    # cv.imshow("Resize",resized_image)
    # cv.waitKey(0)
    return resized_image

def calculateDistance(x1,y1,x2,y2,precision=1):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    dist = round(dist,precision)
    return dist


def get_dict_midpoint(test_dict):
    test_midpoint = [0, 0]
    for k in test_dict:
        test_midpoint = [test_midpoint[0]+test_dict[k][0],test_midpoint[1]+test_dict[k][1]]
    # test_midpoint = [test_midpoint[0]//len(test_dict),test_midpoint[1]//len(test_dict)]
    test_midpoint = [test_midpoint[0]/len(test_dict),test_midpoint[1]/len(test_dict)]
    return test_midpoint


def get_dict_bounds(test_dict, dim=2):
    """
    Returns a list of the bounds of a reference point dictionary of a specified dimension (default is 2D).
    List is in the format [ [d1 min, d1 max], [d2 min, d2 max] , ... ]
    """
    bounds = []
    for i in range(dim):
        bounds.append([9999,0])
    
    for p in test_dict:
        for i in range(dim):
            if test_dict[p][i] < bounds[i][0]: bounds[i][0] = test_dict[p][i] 
            if test_dict[p][i] > bounds[i][1]: bounds[i][1] = test_dict[p][i] 
    
    print("BOUNDS:\n", bounds)
    return bounds

def translate_dict(test_dict, translation=[0,0]):
    for k in test_dict:
        test_dict[k] = [test_dict[k][0]+translation[0],test_dict[k][1]+translation[1]]
    return test_dict

def move_dict_to(test_dict,move_to=[0,0]):
    verbose = True
    if verbose: print(f'Move to: {move_to}')
    mp = get_dict_midpoint(test_dict)
    if verbose: print(f"MP:{mp}")
    translation = [move_to[0]-mp[0],move_to[1]-mp[1]]
    if verbose: print(f"Tran: {translation}")
    test_dict = translate_dict(test_dict,translation)
    return test_dict

def rotate(origin, point, angle):
    '''
    Rotate a point counterclockwise by a given angle around a given origin. 
    The angle should be given in radians.'''
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def click_event(event,x,y,flags,param):
    global marker_position, position_changed, pts_selection
    
    if event == cv.EVENT_LBUTTONUP:
        if (pts_selection == 1):
            pts1[marker_position] = [x, y]
            print("PTS1",pts1)
        elif (pts_selection == 2):
            pts2[marker_position] = [x, y] 
            print("PTS2",pts2)

        marker_position = marker_position + 1   
        position_changed = True

def calc_coverage(truth_dict,eval_dict,verbose=True):
    '''
    RETURNS [Matching Truth Dict, Matching Eval Dict, Coverage Score]
    '''
    # Make sure parameters are dictionaries
    if verbose: print("CALCULATING COVERAGE")
    if isinstance(truth_dict, dict):
        truth_dict_copy = truth_dict.copy()
    else:
        print("ERROR: truth_dict was not of type dictionary.")
        return -1
    if isinstance(eval_dict, dict):
        eval_dict_copy = eval_dict.copy()
    else:
        print("ERROR: eval_dict was not of type dictionary.")
        return -1

    # Make sure both dicts only contain matching entries
    for p in truth_dict:
        if eval_dict_copy.get(p) is None:
            if verbose: print("Entry "+str(p)+" NOT FOUND in EVAL")
            truth_dict_copy.pop(p)
    for p in eval_dict:
        if truth_dict_copy.get(p) is None:
            if verbose: print("Entry "+str(p)+" NOT FOUND in GROUND TRUTH")
            eval_dict_copy.pop(p)
    
    # Calculate coverage (% of truth_dict points remaining in the copy)
    if len(truth_dict) != 0:
        coverage = (float(len(truth_dict_copy)) / len(truth_dict)) * 100
    else:
        print("Ref dict empty")
        coverage = 100

    coverage = round(coverage,1)
    if verbose: print("Coverage Found: ",coverage)
    return truth_dict_copy, eval_dict_copy, coverage

# NEW VERSION, NO IMAGE, RETURNS DICT OF POINT ERRORS
def calc_error(truth_dict,eval_dict,precision=4,verbose=True):
    '''
    RETURNS [POINT ERROR DICT, SCALE FACTORS DICT, GLOBAL ERROR, GLOBAL ERROR STDDEV]
    '''
    # TODO: Implement draw_debug_image parameter that returns an image to 
    #       represent the calculation
    if verbose: print("CALCULATING ERROR")
    dict1 = truth_dict.copy()
    dict2 = eval_dict.copy()

    point_error_list = []
    point_errors = dict()
    dist_factors = dict()

    if verbose: print("POINT\tERROR")
    # For each reference point...
    for p in sorted(dict1.keys()):
        p1 = [dict1[p][0], dict1[p][1]]
        p2 = [dict2[p][0], dict2[p][1]]
        d1 = []
        d2 = []
        dist_factors[p] = dict()
        # Get distance to each other reference point...
        for j in sorted(dict2.keys()):
            p3 =[dict1[j][0], dict1[j][1]]
            p4 =[dict2[j][0], dict2[j][1]]
            if p1 == p3: continue
            gt_dist = calculateDistance(p1[0],p1[1],p3[0],p3[1],precision)
            eval_dist = calculateDistance(p2[0],p2[1],p4[0],p4[1],precision)
            scale_factor = gt_dist / eval_dist
            # if verbose:
            #     print("{}->{}: GT:{}   Eval:{}   Scale:{}".format(p,j,gt_dist,eval_dist,scale_factor))
            dist_factors[p][j] = scale_factor
            d1.append(gt_dist)
            d2.append(eval_dist)

        # Subtract the distances to get an error for each other reference 
        e = np.asarray(d1) - np.asarray(d2)
        e = np.abs(e)

        # Then get an average error for that reference point
        e = round(float(np.average(e)),precision)
        if verbose: print("{}\t{}".format(p,e))
        point_errors[p] = e
        point_error_list.append(e)

    # Average the average error for each point for a final score
    px_error = round(np.average(np.asarray(point_error_list)),precision)
    px_std = round(np.std(point_error_list),precision)

    if verbose: print("Global Error(px): ",px_error)
    if verbose: print("Std Dev(px): ",px_std,"\n")

    return [point_errors, dist_factors, px_error, px_std]


def calc_error_3d(ref_dict,test_dict,label="Not Specified",verbose=True):
    if verbose: print("CALCULATING ERROR")
    dict1 = ref_dict.copy()
    dict2 = test_dict.copy()

    largest_distance = 0
    largest_dist_string = "Unknown"
    dlist1 = []
    dlist2 = []
    errors = []
    error_avg = []

    # For each reference point...
    for p in sorted(dict1.keys()):
        print("\nGetting distances wrt ",p,"...")
        p1 = [dict1[p][0], dict1[p][1], dict1[p][2]]
        p2 = [dict2[p][0], dict2[p][1], dict2[p][2]]
        d1 = []
        d2 = []
        check_str = ""
        print("REF ",p,": ",p1)
        print("EVL ",p,": ",p2)
        # Get distance to each other reference point...
        for j in sorted(dict2.keys()):
            p3 =[dict1[j][0], dict1[j][1], dict1[j][2]]
            p4 =[dict2[j][0], dict2[j][1], dict2[j][2]]
            if p1 == p3: continue
            check_str = check_str + j + " "
            # Compute distances
            dist1 = np.linalg.norm(np.asarray(p1) - np.asarray(p3))
            if dist1 > largest_distance:
                largest_distance = dist1
                largest_dist_string = "(" + str(p) + " to " + str(j) + ")"
            d1.append(dist1)
            d2.append(np.linalg.norm(np.asarray(p2) - np.asarray(p4)))

        # Subtract the distances to get an error for each other reference 
        print("Checked in order: ", check_str)
        #print("D1\n",d1)
        #print("D2\n",d2)
        e = np.asarray(d1) - np.asarray(d2)
        e = np.around(e,decimals=2)
        
        e = np.abs(e)
        print("Distance Errors: ",e)
        # if verbose:  print(e)
        errors.append(list(e))
        # Then get an average error for that reference point
        e = round(float(np.average(e)),2)
        error_avg.append(e)
        dlist1.append(d1)
        dlist2.append(d2)

    if verbose: print("\nLargest Groundtruth Distance (For Scale): ",round(largest_distance,2), largest_dist_string)
    if verbose: print("Error Averages: ",error_avg)
    # Average the average error for each point for a final score
    px_error = round(np.average(np.asarray(error_avg)),2)
    px_std = round(np.std(error_avg),2)

    if verbose: print("Error Metric: ",px_error)
    if verbose: print("Std dev: ",px_std,"\n")

    return px_error, px_std

def auto_align(truth_dict,eval_dict,verbose=True):
    '''
    Translates the eval_dict so that it's midpoint is aligned with
    the midpoint of the truth_dict
    Returns aligned_dict, [x_offset,y_offset]
    '''
    # print(verbose)
    verbose = True
    if verbose: print("AUTO ALIGN")
    align_dict = eval_dict.copy()

    # Find midpoints of each reference point dictionary
    test_midpoint = get_dict_midpoint(align_dict)
    if verbose: print("Evaluation Midpoint: ",test_midpoint)
    ref_midpoint = get_dict_midpoint(truth_dict)
    if verbose: print("Ground Truth Midpoint: ",ref_midpoint)

    # Determine midpoint offset
    xy_off = [ref_midpoint[0]-test_midpoint[0],ref_midpoint[1]-test_midpoint[1]]
    if verbose: print("Eval Midpoint Translation:",xy_off)
    
    # Perform the translation to align the midpoints
    align_dict = translate_dict(align_dict,xy_off)

    return align_dict, xy_off

def scale_dict(pts_dict, scale_factor,midpoint=None,verbose=True):
    if midpoint is None:
        midpoint = get_dict_midpoint(pts_dict)
    # Scale Test Dict
    print("Scaling Point Dictionary by factor of {}".format(scale_factor))
    scaled_dict = pts_dict.copy()
    for m in scaled_dict:
        # scaled_x = float((scaled_dict[m][0]-midpoint[0]) * scale_factor)+midpoint[0]
        # scaled_y = float((scaled_dict[m][1]-midpoint[1]) * scale_factor)+midpoint[1]
        scaled_x = midpoint[0] + scale_factor * (scaled_dict[m][0] - midpoint[0])
        scaled_y = midpoint[1] + scale_factor * (scaled_dict[m][1] - midpoint[1])
        # scaled_x = float(scaled_dict[m][0] * scale_factor)
        # scaled_y = float(scaled_dict[m][1] * scale_factor)
        scaled_dict[m] = [scaled_x, scaled_y]
        if verbose:
            print("{} -> {}".format(pts_dict[m],scaled_dict[m]))
    return scaled_dict

def auto_scale(truth_dict,eval_dict,draw_debug_img=False,verbose=True):
    '''
    Returns [min_scaled_dict, min_scale, None]
    '''
    # TODO: Implement draw_debug_image, parameterize scale_step, min_step
    scale_step = -0.05
    min_step = scale_step / 8
    current_scale = 1.0
    
    if verbose:
        print("Attempting Autoscaling Global Error")
        # print("Ground Truth dict:", truth_dict)
        # print("Eval dict:",eval_dict)
        print("Scale Step: {}\tMin Step: {}".format(scale_step,min_step))
    

    #Get baseline error at 100% scale
    _, _, error_px, std_px = calc_error(truth_dict, eval_dict, verbose=False)
    last_error = error_px
    if verbose: print("Baseline Error: {}".format(error_px))

    min_error = error_px
    min_scale = current_scale
    min_scaled_dict = eval_dict.copy()
    
    # # Adjust test_pts so the midpoints of both sets are the same
    test_midpoint = get_dict_midpoint(eval_dict)

    count = 1
    while abs(scale_step) > abs(min_step):
        current_scale = current_scale + scale_step
        
        scale_factor = current_scale
        scale_percent = round(scale_factor*100,6)
        
        # Scale Test Dict
        scaled_dict = eval_dict.copy()
        for m in scaled_dict:
            scaled_x = int((scaled_dict[m][0]-test_midpoint[0]) * scale_factor)+test_midpoint[0]
            scaled_y = int((scaled_dict[m][1]-test_midpoint[1]) * scale_factor)+test_midpoint[1]
            scaled_dict[m] = [scaled_x, scaled_y]

        # Calculate new scale error
        _, _, error_px, std_px = calc_error(truth_dict, scaled_dict,verbose=False)
        if verbose:
            print("[Step {}] Scale(%): {}\t Step Size: {}\t Error: {}".format(count,scale_percent,scale_step,error_px))

        # If a new minimum is found, this is the best scale so far
        if min_error is None or error_px < min_error:
            min_error = error_px
            min_scale = scale_factor
            min_scaled_dict = scaled_dict.copy()
        else:
            # Otherwise, if the error increased, decrease the step size and go in the other direction
            print("Reversing...")
            if count > 1:
                scale_step = scale_step / 2
            else:
                current_scale = 1.0
            scale_step = -scale_step 

        if abs(error_px-last_error) < 0.25:
            break
        last_error = error_px
        count = count + 1
        if count > 50:
            if verbose: print("Step limit reached, breaking. ")
            break

    min_scale = round(min_scale,4)
    print("Best Scale: {}\tError: {}".format(min_scale,min_error))
    
    if draw_debug_img:
        # Implement debug draw image
        return min_scaled_dict, min_scale, None # draw_img
    else:
        return min_scaled_dict, min_scale, None

def rotate_dict(point_dict,degrees):
    rot_dict = point_dict.copy()
    test_mp = get_dict_midpoint(rot_dict)
    angle_rad = math.radians(degrees)
    for k in rot_dict:
            rx, ry = rotate(test_mp, rot_dict[k], angle_rad)
            rot_dict[k] = [float(rx),float(ry)]
    return rot_dict 


# Doesnt work yet
def auto_rotate(ref_dict,test_dict,draw_img,verbose=False):
    original_test_dict = test_dict.copy()
    test_mp = get_dict_midpoint(test_dict)

    counter = 1 
    iterations = 3

    rot_initial = 0
    rot_step = 45
    # min_step = rot_step / 8
    
    rot_range = 360
    scores_list = []
    
    current_rot = rot_initial
    min_rot = rot_initial
    min_d_sum = None
    last_d_sum = None

    while counter <= iterations:
        print("Iter {}. Init: {}  Step: {}  Range:{}".format(counter,rot_initial,rot_step,rot_range))
        cval = 255-(counter*(255//iterations))
        while current_rot <= rot_initial + rot_range:
            #For current angle, get distance sum
            d_sum = 0   
            angle_rad = math.radians(current_rot)
            test_dict = original_test_dict.copy()
            for k in test_dict:
                rx, ry = rotate(test_mp, test_dict[k], angle_rad)
                test_dict[k] = [float(rx),float(ry)]
                dist = calculateDistance(test_dict[k][0], test_dict[k][1], ref_dict[k][0], ref_dict[k][1])
                d_sum = d_sum + dist
                cv.circle(draw_img, test_dict[k], 2, (cval,cval,cval), -1)

            d_sum = int(d_sum)

            if min_d_sum is None or d_sum < min_d_sum:
                min_d_sum = d_sum
                min_rot = current_rot
                min_rot_dict = test_dict.copy()
                print("|--New Minimum R: {}  D_SUM: {}".format(min_rot,min_d_sum))

            scores_list.append([round(current_rot,4),d_sum])
            if verbose: print(str(counter)+". ROTATION: "+str(current_rot)+"  D_SUM: "+str(d_sum))

            current_rot = current_rot + rot_step
            last_d_sum = d_sum
            
            
        counter = counter + 1
        # if counter > counter_limit:
        #         print("Loop limit of {} reached, breaking.".format(counter_limit))
        #         break

        rot_initial = min_rot - (rot_step/2)
        tmp_step = rot_step
        rot_step = rot_step * (rot_step/rot_range)
        rot_range = tmp_step
        current_rot = rot_initial

    #Overlay reference points and final rotation
    # for k in original_test_dict:
            # test_dict[k] = [test_dict[k][0]-x_avg,test_dict[k][1]-y_avg]
            # cv.circle(draw_img, original_test_dict[k], 4, (160,160,160), -1)
            # cv.circle(draw_img, min_rot_dict[k], 4, (0,0,255), -1)
            # cv.circle(draw_img, ref_dict[k], 4, (0,255,0), -1)
    print("Best Rotation: {}  D_SUM: {}".format(min_rot,min_d_sum))
    return min_rot_dict, min_rot#, draw_img

def auto_fullprocess(ref_dict,test_dict,test_img,v=True,visualize=True):
    # original_ref_dict = ref_dict.copy()
    # original_test_dict = test_dict.copy()
    processed_img = test_img.copy()
    draw_img = test_img.copy()
    draw_img.fill(255)
    print(v)
    #1. Remove entries that aren't present in both dicts, calculate coverage
    ref_dict, test_dict, coverage = calc_coverage(ref_dict, test_dict, verbose=v)
    
    #2. Get the midpoints for each dict and align test set to the reference
    aligned_dict, xy_off = auto_align(ref_dict, test_dict, draw_img)
    if visualize:
        cv.imshow("Map Comparison",draw_img)
        cv.waitKey(0)

    #3. Find the proper scale to minimize error metric

    scaled_dict, min_scale, scale_img = auto_scale(ref_dict,aligned_dict,draw_img, verbose=v)
    if visualize:
        cv.imshow("Map Comparison",scale_img)
        cv.waitKey(0)
    
    #4. Find proper rotation by minimizing the sum of distances between corresponding markers
    rot_dict, min_rot, rot_img = auto_rotate(ref_dict, scaled_dict, scale_img, verbose=v)
    if visualize:
        cv.imshow("Map Comparison", rot_img)
        cv.waitKey(0)

    # test_img = trans_image.image.copy()
    if v:
        print("FINAL TRANSFORM RESULTS")
        print("Coverage:{}  XY_Offset:{}  Scale:{}  Rotation:{}".format(coverage,xy_off,min_scale,min_rot))
    
    #5. Get transform matrix, apply it to test image
    marker_mp = get_dict_midpoint(test_dict)
    rot_mat = cv.getRotationMatrix2D(marker_mp, -min_rot, min_scale)
    rot_mat[0][2] = rot_mat[0][2] - xy_off[0]
    rot_mat[1][2] = rot_mat[1][2] - xy_off[1]
    rotated_img = cv.warpAffine(processed_img, rot_mat, processed_img.shape[1::-1], flags=cv.INTER_LINEAR,borderValue=(255,255,255))
    
    # if visualize:
    #     cv.imshow("Map Comparison", rotated_img)
    #     cv.waitKey()

    # Return tranformed image and marker dict
    return rotated_img, rot_dict

def generate_pointerror_contour_plot(truth_dict,eval_dict,metrics,image=None,save_file=None,title="Point Error Contour",units="px",note=""):
    scores_dict = metrics[0]
    flip_x = False
    flip_y = True
    draw_labels = True
    sparse_x = []
    sparse_y = []
    sparse_error = []
    for k in eval_dict:
        sparse_x.append(eval_dict[k][0])
        sparse_y.append(eval_dict[k][1])
        sparse_error.append(scores_dict[k])

    vmin = min(sparse_error)
    vmax = max(sparse_error)
    # vmin = 0 
    # vmax = 50

    # Create a regular grid for interpolation
    grid_resolution = 250
    pad = 50
    x = np.linspace(min(sparse_x)-pad, max(sparse_x)+pad, grid_resolution)
    y = np.linspace(min(sparse_y)-pad, max(sparse_y)+pad, grid_resolution)
    X, Y = np.meshgrid(x, y)

    # Interpolate sparse data onto the regular grid
    Z = griddata((sparse_x, sparse_y), sparse_error, (X, Y), method='linear')

    # Create a contour plot
    # Create a figure with a subplot grid (2 rows, 1 column)
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 1, height_ratios=[4, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    # plt.figure(figsize=(16, 9))
    if image is not None:
        print("Adding image to contour plot")
        ax1.imshow(image,zorder=1)

    contour = ax1.contourf(X, Y, Z, levels=20, cmap='RdYlGn_r',alpha=0.7,linewidths=0,vmin=vmin, vmax=vmax)
    # Draw Scatter Points
    ax1.scatter(sparse_x, sparse_y, c='black', cmap='rainbow_r', alpha=0.5, s=4)

    # Set Axis
    if flip_x:
        x_min = max(x)
        x_max = min(x)
    else:
        x_min = min(x)
        x_max = max(x)
    if flip_y:
        y_min = max(y)
        y_max = min(y)
    else:
        y_min = min(y)
        y_max = max(y)
    ax1.axis([x_min, x_max, y_min, y_max])
    ax2.axis([0, 1, 0, 1])
    colorbar = plt.colorbar(contour,label="Global Error ({})".format(units))
    # colorbar.set_clim(0, max(sparse_error))   

    # for x, y, label in zip(sparse_x, sparse_y, sparse_error):
    if draw_labels:
        for k in eval_dict:
            ax1.text(eval_dict[k][0], eval_dict[k][1], k, color='black', fontsize=8, ha='left', va='top')

    
    label_str = "Coverage: {}%\nGlobal Error: {}{}\nStd. Dev: {}{}\n{}".format(
        metrics[4],metrics[2],units,metrics[3],units,note
    )
    # Add text or annotations to the second subplot (ax2)
    ax2.text(0.01, 0.01, label_str, fontsize=12, color='black')
    # Remove axis labels and ticks for the second subplot
    ax2.axis('off')
    # Adjust the layout spacing
    # plt.tight_layout()

    ax1.set_title(title)
    if save_file is not None:
        # plt.show()
        plt.savefig(save_file)
        print("Saved: {}".format(save_file))
        plt.close()
    else:
        plt.show()
        
def generate_scalefactor_plot(truth_dict,eval_dict,metrics,excludestd=0,image=None,save_file=None,title="Scale Factor Plot",units="px",note=""):
    n = len(metrics[0])
    print(f"Using histogram with 5N ({n*5}) bins. (N={n})")
    precision = 7
    flip_x = False
    flip_y = True
    draw_labels = True

    # Coordinates for all point pairs and their corresponding scale value
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    scale = []
    dist_dict = metrics[1]
    for p in dist_dict:
        for k in dist_dict[p]:
            x1.append(eval_dict[p][0])
            y1.append(eval_dict[p][1])
            x2.append(eval_dict[k][0])
            y2.append(eval_dict[k][1])
            scale.append(dist_dict[p][k])

    # Coordinates used for scatter plot
    sparse_x = []
    sparse_y = []
    for k in eval_dict:
        sparse_x.append(eval_dict[k][0])
        sparse_y.append(eval_dict[k][1])


    # Calculate stats on scale values
    vmin = min(scale)
    vmax = max(scale)
    print("Scale min:{}  max:{}".format(vmin,vmax))
    scale_avg = round(np.average(np.asarray(scale)),precision)
    scale_stddev = round(np.std(scale),precision)
    print("Avg:{}  StdDev:{}".format(scale_avg,scale_stddev))
    # Set vmin and vmax for the colorscale
    if scale_stddev < 0.0000001:
        vmin = scale_avg - 0.2
        vmax = scale_avg + 0.2
    else:
        vmin = scale_avg - (3 * scale_stddev)
        vmax = scale_avg + (3 * scale_stddev)
    # vmin = 0 
    # vmax = 50

    # Create a regular grid for interpolation
    grid_resolution = 250
    pad = abs((max(sparse_x)-min(sparse_x))*0.05)
    print("AX1 Padding: ",pad)
    x = np.linspace(min(sparse_x)-pad, max(sparse_x)+pad, grid_resolution)
    y = np.linspace(min(sparse_y)-pad, max(sparse_y)+pad, grid_resolution)

    # Create a figure with a subplot grid (2 rows, 1 column)
    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # Add background image to figure
    if image is not None:
        print("Adding image to contour plot")
        ax1.imshow(image,zorder=1)

    # Create custom color map
    cmap1 = plt.get_cmap('RdYlGn')
    cmap2 = plt.get_cmap('RdYlGn_r')
    colors = []
    n_segments = 128
    for i in range(n_segments):
        t = i / (n_segments - 1)
        colors.append(cmap1(t))
    for i in range(n_segments):
        t = i / (n_segments - 1)
        colors.append(cmap2(t))
    custom_cmap = LinearSegmentedColormap.from_list('custom_RdYlGn_RdYlGn_r', colors, N=n_segments*2)

    # colormap = mpl.colormaps['RdYlGn']
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    
    for i in range(len(x1)):
        if scale[i] <= scale_avg-(scale_stddev*excludestd) or scale[i] >= scale_avg+(scale_stddev*excludestd):
            color = sm.to_rgba(scale[i])  # Map the value to a color
            ax1.plot([x1[i], x2[i]], [y1[i], y2[i]], color=color, alpha=0.5)

    # Plot Scale Factor Data
    hist, bins, _ = ax2.hist(scale, bins=5*n, edgecolor='black', alpha=1.0)
    max_value = max(hist)
    # Draw Exclusion Zone
    ax2.plot([scale_avg+(scale_stddev*excludestd),scale_avg+(scale_stddev*excludestd)],[0,max_value],color='black')
    ax2.plot([scale_avg-(scale_stddev*excludestd),scale_avg-(scale_stddev*excludestd)],[0,max_value],color='black')
    ax2.add_patch(Rectangle((scale_avg-(scale_stddev*excludestd),0),(scale_stddev*excludestd*2),max_value,facecolor = 'black',fill=True,alpha=0.3))
    # Draw Mean and Std Dev markers
    ax2.plot([scale_avg,scale_avg],[0,max_value],color='green')
    ax2.plot([scale_avg+scale_stddev,scale_avg+scale_stddev],[0,max_value],color='yellow')
    ax2.plot([scale_avg-scale_stddev,scale_avg-scale_stddev],[0,max_value],color='yellow')
    ax2.plot([scale_avg+(scale_stddev*2),scale_avg+(scale_stddev*2)],[0,max_value],color='orange')
    ax2.plot([scale_avg-(scale_stddev*2),scale_avg-(scale_stddev*2)],[0,max_value],color='orange')
    ax2.plot([scale_avg+(scale_stddev*3),scale_avg+(scale_stddev*3)],[0,max_value],color='red')
    ax2.plot([scale_avg-(scale_stddev*3),scale_avg-(scale_stddev*3)],[0,max_value],color='red')
    label_str = "Mean: {:f}\nStdDev: {:f}\nNormStdDev: {:f}".format(scale_avg,scale_stddev,scale_stddev/scale_avg)
    print(label_str)
    # ax2.text(bins[0], max_value, label_str, fontsize=10, color='black',va='top')
    # Draw Scatter Points
    ax1.scatter(sparse_x, sparse_y, c='black', alpha=0.5, s=4)

    # Set Axis
    if flip_x:
        x_min = max(x)
        x_max = min(x)
    else:
        x_min = min(x)
        x_max = max(x)
    if flip_y:
        y_min = max(y)
        y_max = min(y)
    else:
        y_min = min(y)
        y_max = max(y)
    ax1.axis([x_min, x_max, y_min, y_max])
    print("AX1 Axis: ",[x_min, x_max, y_min, y_max])
    # ax2.axis([0, vmax, 0, 20])
    # ax2.axis([0, 1, 0, 1])
    cbar = plt.colorbar(sm, label='Scale Values',ax=ax1)
    # colorbar.set_clim(0, max(sparse_error))   

    # for x, y, label in zip(sparse_x, sparse_y, sparse_error):
    if draw_labels:
        for k in eval_dict:
            ax1.text(eval_dict[k][0], eval_dict[k][1], k, color='black', fontsize=8, ha='left', va='top')

    # Add text or annotations to the second subplot (ax2)
    # label_str = "Coverage: {}%\nGlobal Error: {}{}\nStd. Dev: {}{}\n{}".format(
    #     metrics[4],metrics[2],units,metrics[3],units,note
    # )
    # ax2.text(0.01, 0.01, label_str, fontsize=12, color='black')
    # Remove axis labels and ticks for the second subplot
    # ax2.axis('off')
    # Adjust the layout spacing
    # plt.tight_layout()

    ax1.set_title(title)
    ax2.set_title("Connection Scale Factor Histogram")
    ax2.set_xlabel("Scale Factor")
    ax2.set_ylabel("Count")
    if save_file is not None:
        # plt.show()
        plt.savefig(save_file)
        print("Saved: {}".format(save_file))
        plt.close()
    else:
        plt.show()

    return [scale_avg,scale_stddev,scale_stddev/scale_avg]

# FOR TESTING
def modify_dict(point_dict, x=0):
    print("Modding...")
    mod_dict = point_dict.copy()
    dict_mp = get_dict_midpoint(mod_dict)
    delta = x
    dx = random.randint(-50,50)
    dy = random.randint(-50,50)
    mod_dict['RA1'][0] = mod_dict['RA1'][0] + dx
    mod_dict['RA1'][1] = mod_dict['RA1'][1] + dy
    dx = random.randint(-50,50)
    dy = random.randint(-50,50)
    mod_dict['LA3'][0] = mod_dict['LA3'][0] + dx
    mod_dict['LA3'][1] = mod_dict['LA3'][1] + dy
    # for k in mod_dict.keys():
    #     if k[0] == 'R':
    #         print (k)
    #         # Shift
    #         mod_dict[k][0] = mod_dict[k][0] - 0
    #         # Rotate
    #         angle_rad = math.radians(delta)
    #         # rx, ry = rotate(dict_mp, mod_dict[k], angle_rad)
    #         rx, ry = rotate(mod_dict["RC9"], mod_dict[k], angle_rad)
    #         mod_dict[k] = [float(rx),float(ry)]


    return mod_dict

def log_to_csv(log_dir, log_name, data_values):
    log_path = os.path.join(log_dir, log_name)
    header = ['Timestamp', 'Map Name', 'Note',
             'global error', 'global err std dev', 'coverage',
             'scale mean', 'scale std dev', 'normalized scale std dev']  # Adjust the header columns as needed
    
    # Check if the directory exists, create it if it does not
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    file_exists = os.path.isfile(log_path)
    
    # Open the file in append mode
    with open(log_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # If the file does not exist, write the header first
        if not file_exists:
            print("LOG CREATED...")
            writer.writerow(header)
        
        # Append the data values
        writer.writerow(data_values)
        print("LOG DATA ADDED...")

def main(argv):
    global position_changed, marker_position, pts_selection, pts1, pts2
    blank = np.zeros([1000,1000,3],dtype=np.uint8)
    blank.fill(255)
    img = blank.copy()
    pts1_color = (0,255,0)
    pts2_color = (255,0,255)

    uni_scale = 50
    pts1 = import_reference_pts("D:/map_accuracy_eval/example/nerve_basement_omron_sr.pts")
    pts1 = scale_dict(pts1,uni_scale)
    # bounds = get_dict_bounds(pts1)
    # pts1 = translate_dict(pts1,[0,-bounds[1][0]])
    # export_reference_pts(pts1,"nerve_basement_omron_scaled","D:/map_accuracy_eval/example")
    # pts2 = import_reference_pts("D:/map_accuracy_eval/example/nerve_basement_spot_pixelcoord.txt")
    # pts2 = scale_dict(pts2,uni_scale)
    # pts2 = translate_dict(pts2,[0,1200])

    pts2 = import_reference_pts("D:/map_accuracy_eval/example/nerve_basement_spot_pixelcoord.txt")
    # pts2 = scale_dict(pts2,uni_scale)
    # pts2 = translate_dict(pts2,[0,-bounds[1][0]])
    # pts2 = modify_dict(pts2)

    cv.namedWindow("Point Matching",cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback("Point Matching", click_event)
    font = cv.FONT_HERSHEY_PLAIN
    running = True
    while running:

        if position_changed:
            img = blank.copy()
            for k in pts1:
                # print(pts1[k])
                img = cv.circle(img, [int(pts1[k][0]),int(pts1[k][1])], 2, pts1_color, 2)
                img = cv.putText(img,str(k),[int(pts1[k][0]),int(pts1[k][1])],font,1,(0,0,0),1,cv.LINE_AA)
            for k in pts2:
                # print(pts2[k])
                img = cv.circle(img, [int(pts2[k][0]),int(pts2[k][1])], 2, pts2_color, 2)
                img = cv.putText(img,str(k),[int(pts2[k][0]),int(pts2[k][1])],font,1,(0,0,0),1,cv.LINE_AA)
            
            img_text = "Edit Pts:"+str(pts_selection)+"/2   Pos:"+str(marker_position)+"(Press 3 to match)"
            org = (5,25)
            img = cv.putText(img,img_text,org,font,2,(0,0,0),2,cv.LINE_AA)

            position_changed = False

        # KEYBOARD CONTROLS
        k = cv.waitKey(100) & 0xFF
        if k == ord('1'): # Edit pts1
            pts_selection = 1
            marker_position = 1
            position_changed = True
        if k == ord('2'): # Edit pts2
            pts_selection = 2
            marker_position = 1
            position_changed = True
        # if k == ord('3'): # Match points
        #     # cv.namedWindow("Error",cv.WINDOW_AUTOSIZE)
        #     # error_px, error_img = calc_error(pts1, pts2, blank.copy())
        #     scaled_dict, scale, error_img = auto_scale(pts1, pts2, blank.copy())
        #     cv.imshow("Point Matching", error_img)
        #     cv.waitKey()
        #     rot_dict, min_rot, rot_img = auto_rotate(pts1, scaled_dict, blank.copy())
        #     cv.imshow("Point Matching", rot_img)
        #     cv.waitKey()
        if k == ord('4'): # TESTING
            mtd, med, coverage = calc_coverage(pts1,pts2)
            # print(mtd, med, coverage)
            point_errors, global_error, error_std = calc_error(mtd, med)
            # print(point_errors, global_error, error_std)
            # pts2, min_scale, _ = auto_scale(mtd,med)
            mtd, min_scale, _ = auto_scale(med,mtd)
            # print(pts2, min_scale)
            # pts2, offset = auto_align(mtd,pts2)
            position_changed = True

            point_errors, global_error, error_std = calc_error(mtd, pts2)
            generate_pointerror_contour_plot(mtd,pts2,point_errors,"D:/map_accuracy_eval/example/spot_map_image.png")
            # generate_contour_plot(mtd,pts2,point_errors)
        if k == ord('5'): 
            pts1 = translate_dict(pts1,[100,100])
            position_changed = True
        if k == ord('6'): 
            pts2, offset = auto_align(pts1,pts2)
            position_changed = True
        if k == ord('q'): # Exit program
            running = False
        
        
        # print(help(rotate))
        cv.imshow("Point Matching", img)
        

if __name__ == "__main__":
    # main(sys.argv[1:])
    eval_dir = "C:/Users/nullp/Documents/MapEvaluations/"
    eval_files = [f for f in listdir(eval_dir) if isfile(join(eval_dir, f))] 
    mapnum = 0
    # Set Parameters
    # save_dir = "C:/Users/nullp/Projects/map_accuracy_eval/output"
    save_dir = "C:/Users/nullp/Documents/MapEvaluations/output"
    truth_pts_file = "C:/Users/nullp/Projects/map_accuracy_eval/input/devens-f15-gps-gt.txt"
    # eval_pts_file = "C:/Users/nullp/Projects/map_accuracy_eval/input/devens-f15-gps-gt.txt"
    eval_pts_file = "C:/Users/nullp/Documents/MapEvaluations/teal-react-photo.csv"
    # eval_pts_file = os.path.join(eval_dir,eval_files[mapnum])
    # map_name = "Devens F15 Teal Eval2"
    map_name = get_map_label(eval_pts_file)
    map_name = "teal-react-photo"
    fig_note = ""
    excluded_std = 0
    metrics = []
    scale_metrics = []

    # Setup Reference Points
    pts1_scale = 1000000
    pts1 = import_reference_pts(truth_pts_file)
    if pts1_scale is not None:
        pts1 = scale_dict(pts1,pts1_scale)
    # bounds = get_dict_bounds(pts1)
    # pts1 = translate_dict(pts1,[0,-bounds[1][0]])

    pts2_scale = None
    pts2 = import_reference_pts(eval_pts_file)
    if pts2_scale is not None:
        pts2 = scale_dict(pts2,pts2_scale)
    # pts2 = scale_dict(pts2,1/100)
    # pts2 = translate_dict(pts2,[0,-bounds[1][0]])
    # pts2 = modify_dict(pts2)


    # Load map background image if applicable
    map_image = None
    map_img_scale = None
    if map_image is not None:
        map_image = cv.imread("C:/Users/nullp/Projects/map_accuracy_eval/input/420-rough-raster-inv.png")
        if map_img_scale is not None:
            map_image = scale_image_with_aspect_ratio(map_image,map_img_scale)


    # Calculate metrics for pts1 (truth) and pts2 (eval)
    print(f"\nPerforming Evaluation: {map_name}")
    mtd, med, coverage = calc_coverage(pts1,pts2)
    metrics = calc_error(mtd, pts2,precision=12)
    metrics.append(coverage)
    # NOTE: metrics = [point error dict, scale factors dict, global error, global err std dev, coverage]

    # Create Scale Factor Plot
    title = "{} Scale Factors - Excluding +/- {:.2f}stddev".format(map_name,excluded_std)
    filename = "{}_scalefactor_exclude{:02d}".format(map_name,int(excluded_std*10)) 
    save_file = os.path.join(save_dir,filename)
    scale_metrics = generate_scalefactor_plot(mtd,pts2,metrics,excludestd=excluded_std,title=title,save_file=save_file,image=map_image,
                        units="px",note=fig_note)
    # NOTE: scale_metrics = [mean, std dev, normed std dev]
    
    # Create Contour Plot
    filename = "{}_globalerrorcontour".format(map_name)  
    save_file = os.path.join(save_dir,filename)
    title = "{} Global Error Contour Plot".format(map_name)
    # generate_pointerror_contour_plot(mtd,pts2,metrics,image=map_image,title=title,save_file=save_file)
   
    # generate_scalefactor_plot(mtd,pts2,metrics,excludestd=excluded_std,title=title,#image=map_image,
    #                     units="px",note=fig_note)

    log_results = True
    if log_results:
        log_dir = "C:/Users/nullp/Projects/map_accuracy_eval"
        log_name = "map_results.csv"
        log_data = []
        now = datetime.datetime.now()
        now.strftime('%Y-%m-%d-%H:%M:%S')
        log_data = [now, map_name, fig_note] + metrics[2:] + scale_metrics
        log_to_csv(log_dir,log_name,log_data)



