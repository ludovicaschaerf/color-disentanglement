from itertools import combinations
import numpy as np


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
