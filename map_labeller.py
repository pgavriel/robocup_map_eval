#!/usr/bin/env python3
import os
from os import listdir
from os.path import isfile, join
import cv2
import tkinter as tk
from tkinter import filedialog
import utilities as util

position = None
position_changed = False

def click_event(event,x,y,flags,param):
    global position, position_changed, start_position
    # start_position = []
    if event == cv2.EVENT_LBUTTONDOWN:
        start_position = [x, y]
    if event == cv2.EVENT_LBUTTONUP:
        if (start_position == []):
            position = [x, y]
        else:
            position = [(start_position[0]+x)//2,(start_position[1]+y)//2]
        position_changed = True

def center_window(win, width, height):
    screen_width = win.winfo_screenwidth()
    screen_height = win.winfo_screenheight()

    x = (screen_width - width) // 2
    y = (screen_height - height) // 2

    win.geometry(f"{width}x{height}+{x}+{y}")

def on_enter(event=None):
    global text
    text = entry.get()
    print("Label Named:", text)
    popup_window.destroy()
    root.destroy()

def show_popup(default_text=""):
    global popup_window,entry,root
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    popup_window = tk.Toplevel(takefocus=True)
    popup_window.title("Enter Reference Point Label")
    popup_window.focus_force()
    entry = tk.Entry(popup_window)
    entry.insert(0, default_text)  # Default text in the entry widget
    entry.select_range(0, 'end')     # Highlight all text
    entry.focus_set()                # Set focus to the entry widget
    entry.focus_force()
    entry.bind('<Return>', on_enter)

    enter_button = tk.Button(popup_window, text="Enter", command=on_enter)

    entry.pack(padx=10, pady=10)
    enter_button.pack(pady=10)
    center_window(popup_window,200,100)
    # while not submitted:
    tk.mainloop()
    return text

def draw_ref_pt(image,pos,label=None):
    print("Drawing Point \"{}\" at {}".format(label,pos))
    pos[0] = int(pos[0])
    pos[1] = int(pos[1])
    # cv.circle(im, center, 3, (0, 25, 255), 2)
    d = 10
    t = 1
    cv2.line(image,(pos[0],pos[1]-d),(pos[0],pos[1]+d),(0,0,255),t)
    cv2.line(image,(pos[0]-d,pos[1]),(pos[0]+d,pos[1]),(0,0,255),t)
    if label is not None:
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1
        thickness = 1
        lbl_offset = [2,-5]
        org = (pos[0]+lbl_offset[0],pos[1]+lbl_offset[1])
        image = cv2.putText(image,str(label),org,font,font_scale,(255,255,255),thickness+1,cv2.LINE_AA)
        image = cv2.putText(image,str(label),org,font,font_scale,(0,0,0),thickness,cv2.LINE_AA)
    
    return image


# Create a simple GUI to select an image file
root = tk.Tk()
root.withdraw()  # Hide the main window
# center_window(root,1920,1080)
root.geometry(f"{1920}x{1080}+{0}+{0}")
# Ask the user to select an image file using a file dialog
initial_dir = '.'
file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff *.tif *.pgm *.ppm *.pnm")], initialdir=initial_dir)
file_name = os.path.basename(file_path)
file_dir = os.path.dirname(file_path)
running = True
update = True
root.destroy()
if file_path:
    # Load the selected image using OpenCV
    image = cv2.imread(file_path)
    draw_img = image.copy()
    #CV Window
    cv2.namedWindow("Map Image",cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Map Image", click_event)
    # cv2.setWindowProperty("Map Image", cv2.WND_PROP_TOPMOST, 1)
    if image is None:
        print("Failed to load the image.")
        running = False

    point_dict = dict()
    pt_label = None
    lbl_index = 0
    while running:

        if position_changed:
            if pt_label is None:
                pt_label = show_popup(str(len(point_dict)+1))

            print("Label: {}     Pos: {}".format(pt_label,position))
            point_dict[pt_label] = position
            # Redraw map image
            update = True
            position_changed = False
        if update:
            draw_img = image.copy()
            for p in point_dict:
                draw_img = draw_ref_pt(draw_img,point_dict[p],p)

            #Draw Status Text:
            keys = list(point_dict.keys())
            if len(keys) <= lbl_index:
                current_key = None
                current_val = None
            else:
                current_key = keys[lbl_index]
                current_val = point_dict[current_key]
            status_str = "{}/{} - {}: {}".format(lbl_index+1,len(point_dict),current_key,current_val)
            util.draw_textline(draw_img,status_str,scale = 2)
            update = False

        # Display the image 
        cv2.imshow("Map Image",draw_img)

        #Keyboard Controls
        k = cv2.waitKey(100) & 0xFF
        if k == ord('q'):
            running = False
        if k == ord('r'):
            pt_label = show_popup(str(len(point_dict)+1))
        if k == ord('s'): # Save image
            f, extension = os.path.splitext(file_name)
            filename = "pts\\"+f+"_pts.jpg"
            cv2.imwrite(os.path.join(file_dir,filename),draw_img)
            print("Saved {}".format(os.path.join(file_dir,filename)))
        if k == ord('z'): # Import pts (should be in pts/[image_name].pts)
            f, extension = os.path.splitext(file_name)
            pts_file = "pts\\"+f+".pts"
            pts_file = os.path.join(file_dir,pts_file)
            point_dict = util.import_pts(pts_file)
            update = True
        if k == ord('x'): # Export pts (to file_dir/pts/[image_name].pts)
            f, extension = os.path.splitext(file_name)
            filename = "pts\\"+f+"_pts.jpg"
            util.export(os.path.join(file_dir,"pts"),f,point_dict)
            cv2.imwrite(os.path.join(file_dir,filename),draw_img)
            print("Saved {}".format(os.path.join(file_dir,filename)))
        if k == ord('1'): # PREVIOUS REF POINT
            lbl_index = (lbl_index-1)%(len(point_dict)+1)
            keys = list(point_dict.keys())
            if len(keys) <= lbl_index:
                pt_label = None
            else:
                pt_label = keys[lbl_index]
            update = True
            print("Index:",lbl_index)
        if k == ord('2'): # NEXT REF POINT
            lbl_index = (lbl_index+1)%(len(point_dict)+1)
            keys = list(point_dict.keys())
            if len(keys) <= lbl_index:
                pt_label = None
            else:
                pt_label = keys[lbl_index]
            update = True
            print("Index:",lbl_index,"\tLabel:",pt_label)
        if k == ord('3'): # REMOVE REF POINT
            keys = list(point_dict.keys())
            if len(keys) <= lbl_index:
                pass
                print("Nothing removed.")
            else:
                point_dict.pop(keys[lbl_index], None)
                update = True

else:
    print("No image file selected.")

# Close the GUI
# root.destroy()
