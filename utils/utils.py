import pandas as pd
import extcolors
from colormap import rgb2hex
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def hex2rgb(hex_value):
    h = hex_value.strip("#") 
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return rgb

def rgb2hsv(r, g, b):
    # Normalize R, G, B values
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    # h, s, v = hue, saturation, value
    max_rgb = max(r, g, b)    
    min_rgb = min(r, g, b)   
    difference = max_rgb-min_rgb 
    # if max_rgb and max_rgb are equal then h = 0
    if max_rgb == min_rgb:
        h = 0
    # if max_rgb==r then h is computed as follows
    elif max_rgb == r:
        h = (60 * ((g - b) / difference) + 360) % 360
    # if max_rgb==g then compute h as follows
    elif max_rgb == g:
        h = (60 * ((b - r) / difference) + 120) % 360
    # if max_rgb=b then compute h
    elif max_rgb == b:
        h = (60 * ((r - g) / difference) + 240) % 360
    # if max_rgb==zero then s=0
    if max_rgb == 0:
        s = 0
    else:
        s = (difference / max_rgb) * 100
    # compute v
    v = max_rgb * 100
    # return rounded values of H, S and V
    return tuple(map(round, (h, s, v)))

def plot_color_palette(outpath, zoom, list_precent, text_c, list_color, input_image, ):
    bg = 'bg.png'
    fig, ax = plt.subplots(figsize=(192,108),dpi=10)
    fig.set_facecolor('white')
    os.makedirs(outpath, exist_ok=True)
    plt.savefig(outpath + bg)
    plt.close(fig)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(160,120), dpi = 10)
        
    #donut plot
    wedges, text = ax1.pie(list_precent,
                           labels = text_c,
                           labeldistance = 1.05,
                           colors =list_color,
                           textprops = {'fontsize': 150, 'color':'black'})
    plt.setp(wedges, width=0.3)

    #add image in the center of donut plot
    img = mpimg.imread(input_image)
    imagebox = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(imagebox, (0, 0))
    ax1.add_artist(ab)
        
    #color palette
    x_posi, y_posi, y_posi2 = 160, -170, -170
    for c in list_color:
        if list_color.index(c) <= 5:
            y_posi += 180
            rect = patches.Rectangle((x_posi, y_posi), 360, 160, facecolor = c)
            ax2.add_patch(rect)
            ax2.text(x = x_posi+400, y = y_posi+100, s = c, fontdict={'fontsize': 190})
        else:
            y_posi2 += 180
            rect = patches.Rectangle((x_posi + 1000, y_posi2), 360, 160, facecolor = c)
            ax2.add_artist(rect)
            ax2.text(x = x_posi+1400, y = y_posi2+100, s = c, fontdict={'fontsize': 190})

    fig.set_facecolor('white')        
    ax2.axis('off')
    bg = plt.imread(outpath + bg)
    plt.imshow(bg)       
    plt.tight_layout()
    plt.savefig(outpath + input_image.split('/')[-1])
    plt.close(fig)

def cat_from_hue(hues, saturations, values, 
                 colors_list=['Red', 'Yellow', 'Green', 'Cyan', 'Blue', 'Purple', 'Magenta', 'BW'],
                 color_bins=[0, 35, 70, 150, 200, 260, 345, 360]):
    if 'BW' in colors_list:
        if color_bins is None:
            color_bins = [(x) * 360 / (len(colors_list) - 1) for x in range(len(colors_list))]
            
        y_cat, bb = pd.cut(hues, 
                               bins=color_bins,
                               labels=colors_list[:-1], #BW in last position
                               include_lowest=True,
                               ordered=True,
                               retbins=True,
                            )
        y_cat = y_cat.add_categories('BW')
                
        y_cat[hues <= 15] = colors_list[-2]
        y_cat[saturations <= 5] = 'BW'
        y_cat[values <= 5] = 'BW'
    else:
        if color_bins is None:
            color_bins = [(x) * 360 / (len(colors_list)) for x in range(len(colors_list) + 1)]
                
        y_cat, bb = pd.cut(hues, 
                           bins=color_bins,
                           labels=colors_list,
                           include_lowest=True,
                           ordered=True, # True when not repeating colors
                           retbins=True,
                          )
            
    return y_cat
            
def color2range(color, colors_list, color_bins):
    if color == 'BW':
        range_col = {'h': None, 's':[0, 10], 'v':[0, 10]}
    elif color == 'Red':
        range_col = {'h': [340, 10], 's':[10, 100], 'v':None}
    else:
        try:
            idx = colors_list.index(color)
            x = color_bins[idx]
            y = color_bins[idx + 1]
            range_col = {'h': [max(x,10), min(y, 340)], 's':[10, 100], 'v':None}
        except Exception as e:
            print(e)
            range_col = {'h':None, 's':None, 'v':None}
    print(color, range_col)
    return range_col
    
def range2color(h, s, v, colors_list, color_bins):
    if (s >= 0 and s < 10) or (v >= 0 and v < 10):
        color = 'BW'
    elif (h >= 340 or h < 10):
        color = 'Red'
    else:
        for idx, col in enumerate(colors_list[:-1]):
            x = color_bins[idx]
            y = color_bins[idx + 1]
            if (h >= x and h < y):
                color = col
    return color
    
def range2continuous(val, range_col_h):
    if range_col_h is None:
        print('Value is None, not implemented yet')
        return 
    if val >= range_col_h[0] and val <= range_col_h[1]:
        cont_value = 180
    elif range_col_h[0] == 340:
        if (val >= range_col_h[0] or val <= range_col_h[1]):
            cont_value = 180
        elif val < range_col_h[0] and val >= range_col_h[0] - 180:
            cont_value = (180 - np.abs(range_col_h[0] - val)) 
        elif val > range_col_h[1] and val <= range_col_h[1] + 180:
            cont_value = (180 - np.abs(range_col_h[1] - val)) 
        else:
            print('Not sure what this case is', val, range_col_h)
            return    
    else:
        if val > range_col_h[1] and val <= range_col_h[1] + 180:
            cont_value = (180 - np.abs(range_col_h[1] - val)) 
        elif val > range_col_h[1] and val >= range_col_h[1] + 180:
            remainder = 360 - val
            cont_value = (180 - np.abs(range_col_h[0] + remainder))
        elif val < range_col_h[0] and val <= np.abs(range_col_h[0] - 180):
            cont_value = (180 - np.abs(val - range_col_h[0])) 
        elif val < range_col_h[0] and val >= np.abs(range_col_h[0] - 180):
            remainder = val
            cont_value = (180 - np.abs((360 - range_col_h[1]) + remainder))
        else:
            print('Not sure what this case is', val, range_col_h)
            return
    #print(range_col_h, val, cont_value)
    return np.abs(cont_value)

