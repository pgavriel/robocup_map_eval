import os
import argparse
import cv2
import utilities as util

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
        print(f"Mouse clicked at ({mouse_x}, {mouse_y})")

def main(args):
    #Variables =====================================================
    eval_pt_scale = args.scale
    eval_pt_rotation = args.rotation
    flip_x = args.flip_x
    flip_y = args.flip_y
    # Eval point modes (Show all, show marker at current_pos[ition], hide points)
    modes = ["All","Evaluate","Hide"]
    current_mode = 0 # Index position within modes list
    current_pos = 0 # Index position within eval points
    # Flags
    global mouse_x, mouse_y, click_flag
    mouse_x, mouse_y = 0, 0
    click_flag = False
    running = True
    update = True

    # GROUND TRUTH POINTS & MAP IMAGE ======================================
    gt_file = "LabGroundTruth.pts"
    qr_pts = util.import_pts(os.path.join(args.gt_dir,gt_file))
    util.create_hazmat_dict(args.gt_dir,"LabGroundTruth_hazmat",qr_pts)
    haz_pts = util.import_pts(os.path.join(args.gt_dir,"LabGroundTruth_hazmat.pts"))
    obj_pts = util.import_pts(os.path.join(args.gt_dir,"LabGroundTruth_objects.pts"))
    real_pts = util.import_pts(os.path.join(args.gt_dir,"LabGroundTruth_real.pts"))
    ptslist_position = 0
    pts_list = [qr_pts, haz_pts, obj_pts, real_pts]
    lbls_list = ["QR", "Hazmat", "Objects (Images)", "Objects (Real)"]
    # Ground Truth Map
    map_img = cv2.imread(os.path.join(args.gt_dir,args.gt_map))
    height, width, _ = map_img.shape
    map_img_center = [width//2, height//2]

    #EVALUATION POINTS FILE =================================================
    eval_pts = util.import_pts(os.path.join(args.eval_dir,args.eval_file))
    eval_label = util.get_map_label(os.path.join(args.eval_dir,args.eval_file))
    eval_pts = util.flip_dict(eval_pts,flip_x,flip_y)
    eval_pts = util.move_dict_to(eval_pts,map_img_center)
    new_eval_pts = eval_pts.copy()

    # CV Window ==============================================================
    cv2.namedWindow('Robocup Map Eval')
    cv2.createTrackbar('Scale', 'Robocup Map Eval', eval_pt_scale, 1000, update_scale)  # scale from 0.0 to 10.0
    cv2.createTrackbar('Rotation', 'Robocup Map Eval', eval_pt_rotation, 360, update_rotation)  # rotation from 0 to 360 degrees
    cv2.setMouseCallback('Robocup Map Eval', mouse_callback)
    new_eval_pts = util.scale_dict(new_eval_pts,eval_pt_scale)
    new_eval_pts = util.rotate_dict(new_eval_pts,eval_pt_rotation)

    # Running loop ============================================================
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
            s = f"[{eval_label}] Showing {lbls_list[ptslist_position]} Ground Truth"
            util.draw_textline(display_img,s,t=1,scale=1.5)

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
        # R - Previous point Index
        if k == ord('r'):
            keys = list(new_eval_pts.keys())
            current_pos = ( current_pos - 1 ) % len(keys)
            print(f"Position {current_pos}/{len(new_eval_pts.keys())}: {keys[current_pos]} @ {new_eval_pts[keys[current_pos]]}")
            update = True
        # T - Next point Index 
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
        if k == ord('s'):
            util.save_screenshot(display_img,args.output_dir)

if __name__ == "__main__":
    # PARAMETER DEFAULT VALUES
    gt_base_dir = "./data/ground_truth/pts"         # Ground Truth Data Base Directory
    gt_map_image_file = "../LabGroundTruth.png"     # Ground Truth Image Filename
    eval_base_dir = "./data/eval/example"           # Evaluation Data Base Directory
    eval_file = "eval.pts"                          # Evaluation Data Filename
    output_dir = "./output"                         # Directory for saving images
    initial_scale = 100                             # Initial Scale (100 = m -> cm)
    initial_rotation = 90                           # Initial Rotation
    flip_x = True
    flip_y = False

    parser = argparse.ArgumentParser(description="Robocup '24 Labyrinth Map Evaluation Tool")
    parser.add_argument('--gt_dir', dest='gt_dir', type=str, default=gt_base_dir,
                        help='Ground Truth Data Base Directory')
    parser.add_argument('-g', '--gt_map', dest='gt_map', type=str, default=gt_map_image_file,
                        help='Ground Truth Image Filename')
    parser.add_argument('--eval_dir', dest='eval_dir', type=str, default=eval_base_dir,
                        help='Evaluation Data Base Directory')
    parser.add_argument('-e', '--eval_file', dest='eval_file', type=str, default=eval_file,
                        help='Evaluation Data Filename (.pts file)')
    parser.add_argument('-o', '--output_dir', dest='output_dir', type=str, default=output_dir,
                        help='Directory for saving output images')
    parser.add_argument('--scale', dest='scale', type=int, default=initial_scale,
                        help='Initial scale to apply to points')
    parser.add_argument('--rotation', dest='rotation', type=int, default=initial_rotation,
                        help='Initial rotation to apply to points')
    parser.add_argument('--flip-x', dest='flip_x', type=bool, default=flip_x,
                        help='Whether to flip the x values of eval points')
    parser.add_argument('--flip-y', dest='flip_y', type=bool, default=flip_y,
                        help='Whether to flip the y values of eval points')
    args = parser.parse_args()
    main(args)