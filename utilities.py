import cv2
from math import radians, sin, cos

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

def rotate(origin, point, angle):
    '''
    Rotate a point counterclockwise by a given angle around a given origin. 
    The angle should be given in radians.'''
    ox, oy = origin
    px, py = point

    qx = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy)
    qy = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy)
    return qx, qy

def rotate_dict(point_dict,degrees):
    rot_dict = point_dict.copy()
    test_mp = get_dict_midpoint(rot_dict)
    angle_rad = radians(degrees)
    for k in rot_dict:
            rx, ry = rotate(test_mp, rot_dict[k], angle_rad)
            rot_dict[k] = [float(rx),float(ry)]
    return rot_dict 

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