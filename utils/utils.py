
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
