import os
import datetime
import cv2
from math import radians, sin, cos
# I/O FUNCTIONS ==============================
def generate_unique_filename(label="capture",format="%y-%m-%d-%H-%M-%S"):
    # Get the current time
    now = datetime.datetime.now()
    # Format the time string 
    time_str = now.strftime(format)
    # Create the filename with the desired suffix
    filename = f"{time_str}-{label}.png"
    return filename

def save_screenshot(image, save_dir='./output'):
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Generate the file name and path
    im_name = generate_unique_filename()
    save_path = os.path.join(save_dir,im_name)
    # Save the image to the specified file path
    cv2.imwrite(save_path, image)
    print(f"Screenshot saved: {save_path}")


# POINT I/O FUNCTIONS ==========================================
def get_map_label(pts_file):
    '''
    Loads a csv textfile containing a maps reference points and
    returns the first line of the file as the "label" for the map
    '''
    print("Getting label for "+str(pts_file)+" ...")
    label = "None"
    try:
        f = open(pts_file,"r")
        lines = f.readlines()
        label = lines[0].split(',')[0]
    
        print(f"Map Label: {label}")
    except:
        print("GET LABEL FAILED")
    
    return label

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

# HAZMAT CORRESPONDENCE LABELS (QR NUMBER: HAZMAT LABEL)
hazmat_dict = {
    1 : "",
    2: "",
    3: "Radioactive",
    4: "Corrosive",
    5: "Explosives",
    6: "",
    7: "",
    8: "",
    9: "Fuel_Oil",
    10: "Flammable Solid_2",
    11: "",
    12: "",
    13: "Inhalation Hazard",
    14: "",
    15: "Spontaneously Combustable",
    16: "?_2",
    17: "",
    18: "Flammable Gas_2",
    19: "",
    20: "",
    21: "",
    # 22: "Organic Peroxide_1",
    23: "Oxidizer",
    24: "",
    25: "",
    26: "",
    27: "",
    28: "",
    29: "",
    30: "Non-Flammable Gas_2",
    31: "Organic Peroxide",
    32: "",
    33: "",
    34: "",
    35: "",
    36: "",
    37: "Poison",
    38: "Blasting Agent",
    39: "Dangerous_when_Wet",
    40: "",
    41: "",
    42: "Oxygen",
    # 43: "Flammable Gas_4"
}

def create_hazmat_dict(dir,filename,qr_dict):
    try:
        pts_file = filename+".pts"
        f = open(os.path.join(dir,pts_file),"w+")
        f.write(filename+"\n")
        for p in sorted(qr_dict.keys()):
            qr_int = int(p.split('_')[-1])
            write_str = hazmat_dict[qr_int]+","+str(qr_dict[p][0])+","+str(qr_dict[p][1])+"\n"
            f.write(write_str)
        f.close()
        print("POINT DICT EXPORTED TO " + filename)
    except:
        print("Unable to export points")


# POINT MANIPULATION FUNCTIONS ================================================================
def get_dict_midpoint(test_dict):
    test_midpoint = [0, 0]
    for k in test_dict:
        test_midpoint = [test_midpoint[0]+test_dict[k][0],test_midpoint[1]+test_dict[k][1]]
    # test_midpoint = [test_midpoint[0]//len(test_dict),test_midpoint[1]//len(test_dict)]
    test_midpoint = [test_midpoint[0]/len(test_dict),test_midpoint[1]/len(test_dict)]
    return test_midpoint

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

def scale_dict(pts_dict, scale_factor, midpoint=None, verbose=False):
    if midpoint is None:
        midpoint = get_dict_midpoint(pts_dict)
    # Scale Test Dict
    print("Scaling Point Dictionary by factor of {}".format(scale_factor))
    scaled_dict = pts_dict.copy()
    for m in scaled_dict:
        scaled_x = midpoint[0] + scale_factor * (scaled_dict[m][0] - midpoint[0])
        scaled_y = midpoint[1] + scale_factor * (scaled_dict[m][1] - midpoint[1])
        scaled_dict[m] = [scaled_x, scaled_y]
        if verbose:
            print("{} -> {}".format(pts_dict[m],scaled_dict[m]))
    return scaled_dict

def rotate(origin, point, angle):
    '''
    Rotate a point counterclockwise by a given angle around a given origin. 
    The angle should be given in radians.'''
    ox, oy = origin
    px, py = point

    qx = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy)
    qy = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy)
    return qx, qy

def rotate_dict(point_dict, degrees):
    rot_dict = point_dict.copy()
    test_mp = get_dict_midpoint(rot_dict)
    angle_rad = radians(degrees)
    for k in rot_dict:
            rx, ry = rotate(test_mp, rot_dict[k], angle_rad)
            rot_dict[k] = [float(rx),float(ry)]
    return rot_dict 

def flip_dict(input_dict, flip_x=False, flip_y=False):
    for p in input_dict:
        x = input_dict[p] 
        if flip_x and flip_y: input_dict[p] = [x[0] * -1,x[1] * -1]
        elif flip_x: input_dict[p] = [x[0] * -1,x[1]]
        elif flip_y: input_dict[p] = [x[0],x[1] * -1]
    return input_dict

# DRAWING FUNCTIONS ============================================================================
def draw_textline(image,string,org=[5,30],scale=1.0,c=(0,0,0),c2=(255,255,255),t=2,border=True):
    font = cv2.FONT_HERSHEY_PLAIN
    if border: image = cv2.putText(image,string,org,font,scale,c2,t+1,cv2.LINE_AA)
    image = cv2.putText(image,string,org,font,scale,c,t,cv2.LINE_AA)

def draw_ref_pt(image,pos,label=None,color=(0,0,255)):
    # print("Drawing Point \"{}\" at {}".format(label,pos))
    pos[0] = int(pos[0])
    pos[1] = int(pos[1])
    # cv.circle(im, center, 3, (0, 25, 255), 2)
    d = 10
    t = 1
    cv2.line(image,(pos[0],pos[1]-d),(pos[0],pos[1]+d),color,t)
    cv2.line(image,(pos[0]-d,pos[1]),(pos[0]+d,pos[1]),color,t)
    if label is not None:
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1
        thickness = 1
        lbl_offset = [2,-5]
        org = (pos[0]+lbl_offset[0],pos[1]+lbl_offset[1])
        image = cv2.putText(image,str(label),org,font,font_scale,(255,255,255),thickness+1,cv2.LINE_AA)
        image = cv2.putText(image,str(label),org,font,font_scale,color,thickness,cv2.LINE_AA)
    
    return image
