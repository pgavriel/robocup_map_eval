import os
import argparse
import cv2
# import point_matching as pm 
import utilities as util

# # HAZMAT CORRESPONDENCE LABELS (QR NUMBER: HAZMAT LABEL)
# hazmat_dict = {
#     1 : "",
#     2: "",
#     3: "Radioactive",
#     4: "Corrosive",
#     5: "Explosives",
#     6: "",
#     7: "",
#     8: "",
#     9: "Fuel_Oil",
#     10: "Flammable Solid_2",
#     11: "",
#     12: "",
#     13: "Inhalation Hazard",
#     14: "",
#     15: "Spontaneously Combustable",
#     16: "?_2",
#     17: "",
#     18: "Flammable Gas_2",
#     19: "",
#     20: "",
#     21: "",
#     # 22: "Organic Peroxide_1",
#     23: "Oxidizer",
#     24: "",
#     25: "",
#     26: "",
#     27: "",
#     28: "",
#     29: "",
#     30: "Non-Flammable Gas_2",
#     31: "Organic Peroxide",
#     32: "",
#     33: "",
#     34: "",
#     35: "",
#     36: "",
#     37: "Poison",
#     38: "Blasting Agent",
#     39: "Dangerous_when_Wet",
#     40: "",
#     41: "",
#     42: "Oxygen",
#     # 43: "Flammable Gas_4"
# }

def import_pts(pts_file):
    print("Importing "+str(pts_file)+" ...")
    try:
        point_dict = dict()
        f = open(pts_file,"r")
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            data = line.split(',')
            # print(data)
            if data[0].isnumeric():
                marker = int(data[0])
            else:
                marker = data[0]
            x = float(data[1])
            y = float(data[2][:-1])
            coords = [x,y]
            point_dict[marker] = coords
        print("Imported point dict:")
        print(point_dict)
        return point_dict
    except:
        print("PTS IMPORT FAILED")
        return None
    
def export(dir,filename,ref_dict):
    try:
        pts_file = filename+".pts"
        f = open(os.path.join(dir,pts_file),"w+")
        f.write(filename+"\n")
        for p in sorted(ref_dict.keys()):
            write_str = str(p)+","+str(ref_dict[p][0])+","+str(ref_dict[p][1])+"\n"
            f.write(write_str)
        f.close()
        print("POINT DICT EXPORTED TO " + filename)
    except:
        print("Unable to export points")

def create_hazmat_dict(dir,filename,qr_dict):
    try:
        pts_file = filename+".pts"
        f = open(os.path.join(dir,pts_file),"w+")
        f.write(filename+"\n")
        for p in sorted(qr_dict.keys()):
            qr_int = int(p.split('_')[-1])
            write_str = util.hazmat_dict[qr_int]+","+str(qr_dict[p][0])+","+str(qr_dict[p][1])+"\n"
            f.write(write_str)
        f.close()
        print("POINT DICT EXPORTED TO " + filename)
    except:
        print("Unable to export points")

# def draw_textline(image,string,org=[5,30],scale=1.0,c=(0,0,0),c2=(255,255,255),t=2,border=True):
#     font = cv2.FONT_HERSHEY_PLAIN
#     if border: image = cv2.putText(image,string,org,font,scale,c2,t+1,cv2.LINE_AA)
#     image = cv2.putText(image,string,org,font,scale,c,t,cv2.LINE_AA)

# def draw_ref_pt(image,pos,label=None,color=(0,0,255)):
#     # print("Drawing Point \"{}\" at {}".format(label,pos))
#     pos[0] = int(pos[0])
#     pos[1] = int(pos[1])
#     # cv.circle(im, center, 3, (0, 25, 255), 2)
#     d = 10
#     t = 1
#     cv2.line(image,(pos[0],pos[1]-d),(pos[0],pos[1]+d),color,t)
#     cv2.line(image,(pos[0]-d,pos[1]),(pos[0]+d,pos[1]),color,t)
#     if label is not None:
#         font = cv2.FONT_HERSHEY_PLAIN
#         font_scale = 1
#         thickness = 1
#         lbl_offset = [2,-5]
#         org = (pos[0]+lbl_offset[0],pos[1]+lbl_offset[1])
#         image = cv2.putText(image,str(label),org,font,font_scale,(255,255,255),thickness+1,cv2.LINE_AA)
#         image = cv2.putText(image,str(label),org,font,font_scale,color,thickness,cv2.LINE_AA)
    
#     return image

def dict_apply_flip(input_dict, flip_x=False, flip_y=False):
    for p in input_dict:
        x = input_dict[p] 
        if flip_x and flip_y: input_dict[p] = [x[0] * -1,x[1] * -1]
        elif flip_x: input_dict[p] = [x[0] * -1,x[1]]
        elif flip_y: input_dict[p] = [x[0],x[1] * -1]
    return input_dict


# OpenCV Callbacks ======================================================
# Callback function for the trackbars
def update_scale(val):
    print(f"Scale: {val}")

def update_rotation(val):
    print(f"Rotation: {val}")

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, click_flag
    if event == cv2.EVENT_LBUTTONUP:
        mouse_x, mouse_y = x, y
        click_flag = True
        print(f"Mouse released at ({mouse_x}, {mouse_y})")

def main(args):
    #Variables =====================================================
    global mouse_x, mouse_y, click_flag
    mouse_x, mouse_y = 0, 0
    click_flag = False
    eval_pt_scale = 100
    eval_pt_offset = [567, 335]
    eval_pt_rotation = 90 
    flip_x = True
    flip_y = False
    # Eval point modes (Show all, show marker at current_pos[ition], hide points)
    modes = ["All","Evaluate","Hide"]
    current_mode = 0 # Index position within modes list
    current_pos = 0 # Index position within eval points
    # Flags
    running = True
    update = True


    # GROUND TRUTH POINTS & MAP IMAGE ======================================
    gt_file = "LabGroundTruth.pts"
    qr_pts = import_pts(os.path.join(args.gt_dir,gt_file))
    create_hazmat_dict(args.gt_dir,"LabGroundTruth_hazmat",qr_pts)
    haz_pts = import_pts(os.path.join(args.gt_dir,"LabGroundTruth_hazmat.pts"))
    obj_pts = import_pts(os.path.join(args.gt_dir,"LabGroundTruth_objects.pts"))
    real_pts = import_pts(os.path.join(args.gt_dir,"LabGroundTruth_real.pts"))
    ptslist_position = 0
    pts_list = [qr_pts, haz_pts, obj_pts, real_pts]
    lbls_list = ["QR", "Hazmat", "Objects (Images)", "Objects (Real)"]
    # Ground Truth Map
    map_img = cv2.imread(os.path.join(args.gt_dir,args.gt_map))
    height, width, _ = map_img.shape
    map_img_center = [width//2, height//2]

    #EVALUATION POINTS FILE =================================================
    eval_pts = import_pts(os.path.join(args.eval_dir,args.eval_file))
    eval_pts = dict_apply_flip(eval_pts,flip_x,flip_y)
    eval_pts = util.move_dict_to(eval_pts,map_img_center)
    new_eval_pts = eval_pts.copy()

    # CV Window ==============================================================
    cv2.namedWindow('Robocup Map Eval')
    cv2.createTrackbar('Scale', 'Robocup Map Eval', eval_pt_scale, 1000, update_scale)  # scale from 0.0 to 10.0
    cv2.createTrackbar('Rotation', 'Robocup Map Eval', eval_pt_rotation, 360, update_rotation)  # rotation from 0 to 360 degrees
    cv2.setMouseCallback('Robocup Map Eval', mouse_callback)
    new_eval_pts = util.scale_dict(new_eval_pts,eval_pt_scale)
    new_eval_pts = util.rotate_dict(new_eval_pts,eval_pt_rotation)

    # Running loop
    while running:
        # When image is clicked, move center point of eval points to mouse position
        if click_flag:
            eval_pts = util.move_dict_to(eval_pts,[mouse_x,mouse_y])
            new_eval_pts = util.move_dict_to(new_eval_pts,[mouse_x,mouse_y])
            update = True
            click_flag = False
        
        # Check whether to update eval points scaling
        tb_scale = cv2.getTrackbarPos('Scale', 'Robocup Map Eval')
        if tb_scale != eval_pt_scale:
            eval_pt_scale = tb_scale
            new_eval_pts = eval_pts.copy()
            new_eval_pts = util.scale_dict(new_eval_pts,eval_pt_scale)
            new_eval_pts = util.rotate_dict(new_eval_pts,eval_pt_rotation)
            update = True

        # Check whether to update eval points rotation     
        tb_rot = cv2.getTrackbarPos('Rotation', 'Robocup Map Eval')
        if tb_rot != eval_pt_rotation:
            print(f"Dif:{eval_pt_rotation-tb_rot}")
            new_eval_pts = util.rotate_dict(new_eval_pts,tb_rot-eval_pt_rotation)
            eval_pt_rotation = tb_rot
            update = True

        # Update flag
        if update:
            display_img = map_img.copy()
            # Draw Ground Truth Points
            for p in pts_list[ptslist_position]: 
                util.draw_ref_pt(display_img,pts_list[ptslist_position][p],p)
            s = "Showing "+lbls_list[ptslist_position]+" Ground Truth"
            util.draw_textline(display_img,s,t=1,scale=2.0)


            # Decide how to draw Eval Points based on current_mode 
            if current_mode == 0:
                for p in new_eval_pts: 
                    util.draw_ref_pt(display_img,new_eval_pts[p],p,color=(32,50,1))
                s = "[Show All]"
                util.draw_textline(display_img,s,org=[5,60],t=1,c=(32,50,1),scale=1.5)
            elif current_mode == 1:
                keys = list(new_eval_pts.keys())
                util.draw_ref_pt(display_img,new_eval_pts[keys[current_pos]],keys[current_pos],color=(32,50,1))
                s = f"[Evaluating] Position {current_pos+1}/{len(new_eval_pts.keys())}: {keys[current_pos]} @ {new_eval_pts[keys[current_pos]]}"
                util.draw_textline(display_img,s,org=[5,60],t=1,c=(32,50,1),scale=1.5)
            elif current_mode == 2:
                s = "[Hide Eval Points]"
                util.draw_textline(display_img,s,org=[5,60],t=1,c=(32,50,1),scale=1.5)
            update = False

        # Display the image 
        cv2.imshow("Robocup Map Eval",display_img)

        #Define Keyboard Controls ==========================================================
        k = cv2.waitKey(100) & 0xFF
        # Q - Quit
        if k == ord('q'):
            running = False
        # E - Cycle through eval point modes
        if k == ord('e'):
            current_mode = ( current_mode + 1 ) % len(modes)
            print(f"MODE: {current_mode} : {modes[current_mode]}")
            update = True
        # R - Previous point position
        if k == ord('r'):
            keys = list(new_eval_pts.keys())
            current_pos = ( current_pos - 1 ) % len(keys)
            print(f"Position {current_pos}/{len(new_eval_pts.keys())}: {keys[current_pos]} @ {new_eval_pts[keys[current_pos]]}")
            update = True
        # T - Next point position 
        if k == ord('t'):
            keys = list(new_eval_pts.keys())
            current_pos = ( current_pos + 1 ) % len(keys)
            print(f"Position {current_pos}/{len(new_eval_pts.keys())}: {keys[current_pos]} @ {new_eval_pts[keys[current_pos]]}")
            update = True
        if k == ord('1'):
            ptslist_position = 0
            update = True
        if k == ord('2'):
            ptslist_position = 1
            update = True
        if k == ord('3'):
            ptslist_position = 2
            update = True
        if k == ord('4'):
            ptslist_position = 3
            update = True

if __name__ == "__main__":
    # PARAMETER DEFAULT VALUES
    gt_base_dir = "./data/ground_truth/pts"         # Ground Truth Data Base Directory
    gt_map_image_file = "../LabGroundTruth.png"     # Ground Truth Image Filename
    eval_base_dir = "./data/eval/example"           # Evaluation Data Base Directory
    eval_file = "eval.pts"                          # Evaluation Data Filename

    parser = argparse.ArgumentParser(description="Robocup '24 Labyrinth Map Evaluation Tool")
    parser.add_argument('--gt_dir', dest='gt_dir', type=str, default=gt_base_dir,
                        help='Ground Truth Data Base Directory')
    parser.add_argument('-g', '--gt_map', dest='gt_map', type=str, default=gt_map_image_file,
                        help='Ground Truth Image Filename')
    parser.add_argument('--eval_dir', dest='eval_dir', type=str, default=eval_base_dir,
                        help='Evaluation Data Base Directory')
    parser.add_argument('-e', '--eval_file', dest='eval_file', type=str, default=eval_file,
                        help='Evaluation Data Filename (.pts file)')
    
    args = parser.parse_args()
    main(args)