#Created by: Steven Weller and Setu Marath during the 2025 Western Engineering 
#Automotive Competition, thank you for your permission to use this code.
#Need vedo in addition to yaml file to run this code
#This code is used to visualize the lidar data and the bounding boxes of the objects in the data

import math

import vedo
import numpy as np

plotter = vedo.Plotter()

def load_bin_file(file_path:str):
    # extract all the binary data
    data = np.fromfile(file_path, dtype=np.float32)

    # X, Y, Z
    points = data.reshape(-1, 5)
    # returns list of points

    x_data = points[:, 0]
    y_data = points[:, 1]
    z_data = points[:, 2]

    # need to convert to numpy stack
    # (convert into columns of x, y, z)
    return np.vstack((x_data, y_data, z_data)).T

def generate_points(filepath:str):
    # load point data
    points = load_bin_file(filepath)

    # get distance of each point from origin
    distances = np.linalg.norm(points, axis=1)

    # put distance in range from 0 to 1 for each distance
    min_dist, max_dist = distances.min(), distances.max()
    normalized_distances = (distances - min_dist) / (max_dist - min_dist)

    # create colors based on distance (closer = bright, farther = dark)
    # colors in range from 0 to 1
    colors = [(1-d, 1-d, 1-d) for d in normalized_distances]

    # create a point cloud with distance-based shading
    point_cloud = vedo.Points(points)
    point_cloud.cmap("gray", normalized_distances)

    return point_cloud

def generate_bounding_boxes(filepath:str, label_color:dict=None, additional_label_text:str="", wireframe_on:bool = False, text_color:str="orange"):
    if label_color is None:
        label_color = {'human': "red", 'motorcycle': "green", 'bicycle': "blue"}

    boxes = []
    labels = []

    for line in open(filepath):
        l = line.strip("\n").split(" ")
        label = l[0]
        l = l[1:-1]

        for i in range(len(l)):
            l[i] = float(l[i])

        for k in label_color:
            if k in label.lower():

                if wireframe_on:
                    box = vedo.Box(pos=(l[0], l[1], l[2]),
                                   size=(l[3], l[4], l[5]),
                                   c=label_color[k]).wireframe()
                else:
                    box = vedo.Box(pos=(l[0], l[1], l[2]),
                                   size=(l[3], l[4], l[5]),
                                   c=label_color[k])

                box.rotate(l[-1], (0, 0, 1), (l[0], l[1], l[2]), True)

                boxes.append(
                    box
                )

                text = vedo.Text3D(additional_label_text.capitalize() + ": " + k.capitalize(),
                                   pos=(l[0] - l[3]/2, l[1], l[2] + l[5]),
                                   s=0.5,
                                   c=text_color,
                                   depth=0.5,
                                   justify="centered")

                text.rotate(l[-1], (0, 0, 1), (l[0], l[1], l[2]), True)

                labels.append(
                    text
                )

    return boxes, labels

# which file to load
question_value = input("Which file number do you want to try and use? [0, 19497]: ")

while int(question_value) < 0 or int(question_value) > 19497:
    print("Please enter a valid number")
    test_value = input("What value do you want to try and use? [0, 19497]: ")

question_value = (6 - len(question_value)) * "0" + question_value

ask_compare_solution = input("Do you want to compare a solution file? y/n: ").lower()

while ask_compare_solution != "y" and ask_compare_solution != "n":
    print("Please enter either 'y' or 'n'")
    ask_compare_solution = input("Do you want to compare your solution? y/n: ").lower()

solution_path_found = False
solution_file_path = ""

while not solution_path_found and ask_compare_solution.lower() == "y":
    if ask_compare_solution == "y":
        solution_file_path = input("Enter file path: ")

        if solution_file_path.lower() == "n":
            ask_compare_solution = "n"
            break

        if '.txt' not in solution_file_path:
            print("Please enter a .txt file")

        try:
            open(solution_file_path, "r")
            solution_path_found = True
        except:
            print("File not found, try again. If you want to break out press: 'n'")

answer_boxes, answer_labels = generate_bounding_boxes("data_lidar/data/labels/" + question_value + ".txt",
                                                      additional_label_text="Given", text_color="orange")

point_cloud = generate_points("data_lidar/data/scans/" + question_value + ".bin")

if solution_path_found:
    solution_boxes, solution_labels = generate_bounding_boxes(solution_file_path,
                                                              {'human': "orange",
                                                               'motorcycle': "yellow",
                                                               'bicycle': "purple"},
                                                              "Yours", text_color="brown")

    vedo.show(point_cloud,
              solution_boxes,
              solution_labels,
              answer_boxes,
              answer_labels)
else:
    vedo.show(point_cloud,
              answer_boxes,
              answer_labels)


