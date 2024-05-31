from itertools import combinations
import numpy as np
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1994, delta_e_cie1976
import numpy as np
import colorsys
import sys
from sklearn.cluster import KMeans


sys.path.append('../utils/')
from utils import *

def closest_color(target_rgb, color_list, lum_strength=0):
    # Convert the target color to LAB space
    target_color = sRGBColor(*target_rgb, is_upscaled=True)
    target_lab = convert_color(target_color, LabColor)

    min_distance = float('inf')
    closest_color = None

    for i, rgb in enumerate(color_list):
        # Convert each color in the list to LAB space
        color = sRGBColor(*rgb, is_upscaled=True)
        color_lab = convert_color(color, LabColor)

        # Calculate the distance between the target color and this color
        # distance = delta_e_cie1994(target_lab, color_lab, K_1=0.048, K_2=0.014, K_L=2)
        distance = delta_e_cie1976(target_lab, color_lab)

        # Update the closest color if this distance is smaller
        if distance < min_distance:
            min_distance = distance
            closest_color = rgb
            closest_color_idx = i

    return closest_color, closest_color_idx

def rgb_to_hsv(rgb):
    # Convert RGB to HSV using colorsys (expects RGB values to be in [0, 1] range)
    r, g, b = [x / 255.0 for x in rgb]
    return colorsys.rgb_to_hsv(r, g, b)

def closest_color_hue(target_rgb, color_list, lum_strength=0.2):
    target_hsv = rgb_to_hsv(target_rgb)
    target_hue = target_hsv[0]

    min_distance = float('inf')
    closest_color = None

    for i, rgb in enumerate(color_list):
        color_hsv = rgb_to_hsv(rgb)
        color_hue = color_hsv[0]

        # Calculate the distance between the target hue and this color's hue
        distance = abs(target_hue - color_hue)

        # Handle circular nature of hue
        if distance > 0.5:
            distance = 1 - distance

        distance += lum_strength * (np.abs(target_hsv[1] - color_hsv[1]) + np.abs(target_hsv[2] - color_hsv[2]))
        
        # Update the closest color if this distance is smaller
        if distance < min_distance:
            min_distance = distance
            closest_color = rgb
            closest_color_idx = i

    return closest_color, closest_color_idx

def get_color(target_rgb, colors_list, colors_names, method=closest_color_hue, lum_strength=0.2):
    color, color_idx = method(target_rgb, colors_list, lum_strength)
    color_name = colors_names[color_idx]
    return color_name

def adaptive_clustering_2(image_path, K=8, max_iterations=100, convergence_threshold=0.01):
    # Load the image in RGB
    if type(image_path) == str:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = np.array(image_path)
    
    original_shape = image.shape
    # Flatten the image to a 2D array of pixels (3 columns for R, G, B)
    pixels = image.reshape(-1, 3)
    if np.isnan(pixels).any() or (pixels == 0).all():
        print("Data contains NaN values")
        print(image_path)
        return np.array([(0,0,0)]*K), [len(pixels)]*K
    
    # Initial K-means to segment the image
    kmeans = KMeans(n_clusters=K, random_state=0)
    labels = kmeans.fit_predict(pixels)
    centroids = kmeans.cluster_centers_

    old_labels = labels

    # Adaptive iteration
    n_iterations = 0
    while n_iterations < max_iterations:
        # Recalculate centroids from labels
        centroids = np.array([pixels[labels == i].mean(axis=0) for i in range(K)])

        # Re-cluster using updated centroids
        kmeans = KMeans(n_clusters=K, init=centroids, n_init=1)
        labels = kmeans.fit_predict(pixels)

        # Check for convergence
        if np.sum(labels != old_labels) < convergence_threshold * len(pixels):
            break

        old_labels = labels
        n_iterations += 1

    cnts = np.array([len(pixels[labels == i]) for i in range(K)])
    
    return np.array(centroids) / 255, cnts

def extract_palette(image_path, K, method=adaptive_clustering_2):
    #extcolors.extract_from_path(im, tolerance=K, limit=13)
    return method(image_path, K)

# Function to check if the colors are monochromatic
def is_monochromatic(hues):
    confidence = np.mean([abs(hue - hues[0]) for hue in hues])
    corrects = sum([1 if abs(hue - hues[0]) < 10  else 0 for hue in hues])
    correct = True if corrects >= (len(hues)) else False
    return correct,  np.round(confidence)

# Function to check if the colors are analogous
def is_analogous(hues):
    confidence = np.mean([abs(hue - hues[0]) for hue in hues])
    corrects = sum([1 if abs(hue - hues[0]) <= 60  else 0 for hue in hues])
    correct = True if corrects >= (len(hues)) else False
    return correct,  np.round(confidence)

# Function to check if the colors are complementary
def is_complementary(hues):
    if len(hues) > 1:
        confidence = min([abs((hue - other_hue) % 360 - 180) for hue, other_hue in combinations(hues, 2)])
    else:
        return False, 0
    return any(abs((hue - other_hue) % 360 - 180) < 15 for hue, other_hue in combinations(hues, 2)), np.round(confidence)

# Function to check if the colors are triadic
def is_triadic(hues):
    for hue, other_hue, third_hue in combinations(hues, 3):
        if abs((hue - other_hue) % 360 - 120) < 15 or abs((third_hue - other_hue) % 360 - 120) < 15:
            if abs((third_hue - other_hue) % 360 - 120) < 15 or abs((third_hue - hue) % 360 - 120) < 15:
                if abs((other_hue - hue) % 360 - 120) < 15 or abs((other_hue - third_hue) % 360 - 120) < 15:
                    return True, min(abs((third_hue - other_hue) % 360 - 120), abs((third_hue - hue) % 360 - 120), abs((hue - other_hue) % 360 - 120))
    return False, 0

# Function to check if the colors are split complementary
def is_split_complementary(hues):
    for hue, other_hue, third_hue in combinations(hues, 3):
        if (abs((hue - other_hue) % 360 - 150) < 15 or abs((hue - other_hue) % 360 - 210) < 15) or (
            abs((hue - third_hue) % 360 - 150) < 15 or abs((hue - third_hue) % 360 - 210) < 15):
            if (abs((other_hue - hue) % 360 - 150) < 15 or abs((other_hue - hue) % 360 - 210) < 15) or (
                abs((other_hue - third_hue) % 360 - 150) < 15 or abs((other_hue - third_hue) % 360 - 210) < 15):
                if (abs((third_hue - hue) % 360 - 150) < 15 or abs((third_hue - hue) % 360 - 210) < 15) or (
                abs((third_hue - other_hue) % 360 - 150) < 15 or abs((third_hue - other_hue) % 360 - 210) < 15):
                    return True, min(min(abs((hue - other_hue) % 360 - 150), abs((hue - other_hue) % 360 - 210)), 
                                 min(abs((hue - third_hue) % 360 - 150), abs((hue - third_hue) % 360 - 210)), 
                                 min(abs((third_hue - other_hue) % 360 - 150), abs((third_hue - other_hue) % 360 - 210)), )
    return False, 0

# Function to check if the colors are double complementary
def is_double_complementary(hues):
    for hues_4 in combinations(hues, 4):
        count = 0
        amount = 0
        for hues_2 in combinations(hues_4, 2):
            if is_complementary(hues_2)[0]:
                count += 1
                amount += is_complementary(hues_2)[1]
                
        if count >= 2: 
            return True, np.round(amount / count)
    return False, 0

# Main function to determine the color harmony scheme
def extract_harmonies(hues):
    harmnies = []
    harmnies.append(int(is_monochromatic(hues)[0]))
    harmnies.append(int(is_analogous(hues)[0]))
    harmnies.append(int(is_complementary(hues)[0]))
    harmnies.append(int(is_triadic(hues)[0]))
    harmnies.append(int(is_split_complementary(hues)[0]))
    harmnies.append(int(is_double_complementary(hues)[0]))
    return harmnies

def color_to_df(input_, hex_=False, adaptive=True):
    if adaptive:
        df_rgb = input_[0]
        if hex_:
            df_color_up = [rgb2hex(*i, normalised=True) for i in df_rgb]
        else:
            df_color_up = [rgb2hsv(*i, normalised=True) for i in df_rgb]
        
        df_color_up = [[x*360, y*100,z*100] for [x,y,z] in df_color_up]
        df_percent = input_[1]
    else:
        colors_pre_list = str(input_).replace('([(','').split(', (')[0:-1]
        df_rgb = [i.split('), ')[0] + ')' for i in colors_pre_list]
        df_percent = [i.split('), ')[1].replace(')','') for i in colors_pre_list]
        if hex_:
            #convert RGB to HEX code
            df_color_up = [rgb2hex(int(i.split(", ")[0].replace("(","")),
                                int(i.split(", ")[1]),
                                int(i.split(", ")[2].replace(")",""))) for i in df_rgb]
        else:
            df_color_up = [rgb2hsv(int(i.split(", ")[0].replace("(","")),
                                int(i.split(", ")[1]),
                                int(i.split(", ")[2].replace(")",""))) for i in df_rgb]
    df_rgb = [tuple(i) for i in df_rgb]
    df = pd.DataFrame(zip(df_color_up, df_percent, df_rgb), columns = ['c_code','occurence', 'rgb'])
    return df
    
def plot_color_palette(df_color, input_image, outpath='color_palettes/'):
    #annotate text
    list_color = list(df_color['rgb'])
    list_precent = [int(i) for i in list(df_color['occurence'])]
    text_c = [str(c) + ' ' + str(round(p*100/sum(list_precent),1)) +'%' for c, p in zip(list_color, list_precent)]
    plot_color_palette(outpath, 1, list_precent, text_c, list_color, input_image,)
    colors = list(df_color['c_code'])    
    return colors
