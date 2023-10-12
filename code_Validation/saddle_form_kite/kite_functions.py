import numpy as np


def extract_bridle_connectivity():
    ##TODO: fix hardcoding
    """hardcoded connectivity of the bridle lines"""
    bridle_ci_TE = [
        0,
        21,
        21,
        21,
        22,
        22,
        23,
        23,
        27,
        27,
        24,
        24,
        28,
        28,
        25,
        25,
        29,
        29,
        26,
        26,
        30,
        30,
    ]
    bridle_cj_TE = [
        21,
        22,
        23,
        27,
        24,
        28,
        24,
        1,
        28,
        10,
        25,
        26,
        29,
        30,
        18,
        17,
        11,
        12,
        16,
        15,
        13,
        14,
    ]
    bridle_ci_LE = [
        0,
        0,
        31,
        31,
        34,
        34,
        32,
        32,
        35,
        35,
        33,
        33,
        36,
        36,
        24,
        28,
        31,
        34,
    ]
    bridle_cj_LE = [31, 34, 32, 33, 35, 36, 2, 3, 9, 8, 4, 5, 7, 6, 19, 20, 19, 20]

    bridle_ci = np.append(bridle_ci_TE, bridle_ci_LE)
    bridle_cj = np.append(bridle_cj_TE, bridle_cj_LE)

    return bridle_ci, bridle_cj


def extract_te_line_indices(plate_point_indices, wing_ci, wing_cj):
    """Extracts a list of TE_indices from the plate_point_indices
    using its order: [left_LE, right_LE, right_TE,left_TE]"""
    TE_point_indices = []
    for plate in plate_point_indices:
        TE_point_indices.append(plate[2])
        TE_point_indices.append(plate[3])

    TE_line_indices, tube_line_indices = [], []
    for idx, (ci, cj) in enumerate(zip(wing_ci, wing_cj)):
        if ci in TE_point_indices and cj in TE_point_indices:
            TE_line_indices.append(idx)
    # return [2,8,14,20,26,32,38,44,50] #old hardcoded V3
    return np.array(list(set(TE_line_indices)))


def extract_tube_line_indices():
    ##TODO: fix hardcoding
    """hardcoded V3 indices of the inflatable tubes
    when looping over conn_i, the index that corresponds to a tube"""
    return np.array(
        [
            0,
            1,
            3,
            6,
            7,
            9,
            12,
            13,
            15,
            18,
            19,
            21,
            24,
            25,
            27,
            30,
            31,
            33,
            36,
            37,
            39,
            42,
            43,
            45,
            48,
            49,
            51,
        ]
    )


def extract_plate_point_indices():
    ##TODO: fix hardcoding
    """hardcoded indices of the kite plates"""
    ### Plate connectivity
    plate_1 = [19, 2, 18, 1]
    plate_2 = [2, 3, 17, 18]
    plate_3 = [3, 4, 16, 17]
    plate_4 = [4, 5, 15, 16]
    plate_5 = [5, 6, 14, 15]
    plate_6 = [6, 7, 13, 14]
    plate_7 = [7, 8, 12, 13]
    plate_8 = [8, 9, 11, 12]
    plate_9 = [9, 20, 10, 11]
    plate_point_indices = [
        plate_1,
        plate_2,
        plate_3,
        plate_4,
        plate_5,
        plate_6,
        plate_7,
        plate_8,
        plate_9,
    ]

    return np.array(plate_point_indices)


def extract_wing_connectivity(plate_point_indices):
    """hardcoded connectivity of the wing,
    based on the plate_point_indices"""

    wing_ci, wing_cj = [], []
    for i in np.arange(0, len(plate_point_indices)):
        # The 4 lines describing the tubular frame
        wing_ci.append(plate_point_indices[i][0])  # LE
        wing_cj.append(plate_point_indices[i][1])  # LE

        wing_ci.append(plate_point_indices[i][1])  # Strut right
        wing_cj.append(plate_point_indices[i][2])  # Strut right

        wing_ci.append(plate_point_indices[i][2])  # TE
        wing_cj.append(plate_point_indices[i][3])  # TE

        wing_ci.append(plate_point_indices[i][3])  # Strut left
        wing_cj.append(plate_point_indices[i][0])  # Strut left

        # Them diagonals
        wing_ci.append(plate_point_indices[i][0])
        wing_cj.append(plate_point_indices[i][2])

        wing_ci.append(plate_point_indices[i][1])
        wing_cj.append(plate_point_indices[i][3])

    wing_ci = np.reshape(wing_ci, len(wing_ci))
    wing_cj = np.reshape(wing_cj, len(wing_cj))

    return wing_ci, wing_cj

def calculate_force_aero_plate(plate_point_indices,pos,vel_app,A_projected,rho,equal_boolean=False):
	# Specifying tubular frame + plate diagonals.
	# Reasons for incorporating plate diagonals is that otherwise the quadrilaterals can deform into other shapes
	# This phenomena is best explained by looking at a rectangle that turns into a diamond/kite shape


    aero_force = np.zeros(pos.shape)

    for i in np.arange(0,len(plate_point_indices)): #looping through each panel

        ### Calculating the area of the plate ##TODO: this could be part of a class
        # calculating the sides
        side_1 = np.linalg.norm(pos[plate_point_indices[i][0]]-pos[plate_point_indices[i][1]])
        side_2 = np.linalg.norm(pos[plate_point_indices[i][1]]-pos[plate_point_indices[i][2]])
        side_3 = np.linalg.norm(pos[plate_point_indices[i][2]]-pos[plate_point_indices[i][3]])
        side_4 = np.linalg.norm(pos[plate_point_indices[i][3]]-pos[plate_point_indices[i][0]])

        # Calculating the semi-perimeter (s) of the given quadilateral
        s = (side_1 + side_2 + side_3 + side_4) / 2

        # Applying Brahmagupta's formula to #https://en.wikipedia.org/wiki/Brahmagupta%27s_formula
        # get maximum area of quadrilateral
        area_p =  np.sqrt((s - side_1) *(s - side_2) * (s - side_3) * (s - side_4))
        # a,b,c,d = side_1,side_2,side_3,side_4
        # area_p_other = (1/4) * np.sqrt((-a+b+c+d)*(a-b+c+d)*(a+b-c+d)*(a+b+c-d))

        ### Calculating the angle of attack
        middle_LE_point = (np.subtract(pos[plate_point_indices[i][1]], pos[plate_point_indices[i][0]]) / 2) + pos[plate_point_indices[i][0]]
        middle_TE_point = (np.subtract(pos[plate_point_indices[i][3]], pos[plate_point_indices[i][2]]) / 2) + pos[plate_point_indices[i][2]]
        middle_vec = np.subtract(middle_TE_point,middle_LE_point)
        middle_vec_unit = middle_vec / np.linalg.norm(middle_vec)

        Va_norm = np.linalg.norm(vel_app)
        vw_vec_unit = vel_app / Va_norm
        aoa_p = np.arccos(np.dot(middle_vec_unit,vw_vec_unit)/(np.linalg.norm(middle_vec_unit)*np.linalg.norm(vw_vec_unit)))

        ###  Define the perpendicular unit vector to each segment
        # Defining 2 vectors tangential to the plane, by using the diagonals
        diagonal_1 = np.subtract(pos[plate_point_indices[i][0]], pos[plate_point_indices[i][2]])
        diagonal_2 = np.subtract(pos[plate_point_indices[i][1]], pos[plate_point_indices[i][3]])

        # Finding the cross product of these two vectors, should give a perpendicular vector / two solutions actually.
        perpendicular_vector_1 = np.cross(diagonal_1,diagonal_2)
        unit_perp_vec = perpendicular_vector_1 / np.linalg.norm(perpendicular_vector_1)

        ### Find the direction of lift and drag
        L_vec_unit = unit_perp_vec

        # correcting the angle of attack for the orientation of the plate
        if unit_perp_vec[0] >0: #the plate is tilted backwards
            aoa_p = aoa_p
        elif unit_perp_vec[0] <0: #the plate is tilted forwards
            aoa_p = -aoa_p

        # AoA is equal to aoa_p
        Cl_p = 2*np.pi*np.sin(aoa_p)
        L_p = 0.5*rho*(Va_norm**2)*Cl_p*(area_p)

        ## Splitting lift 
        Fx_p = L_p*L_vec_unit[0] 
        Fy_p = L_p*L_vec_unit[1] # side-force
        Fz_p = L_p*L_vec_unit[2] 

        ## Apply 1/4 of perpendicular unit vector to each node of the respective panel
        for k,j in enumerate(plate_point_indices[i]): #loop Through all the indexes for the defined plate
            if equal_boolean == True:
                aero_force[j, 0] += 0.25*Fx_p  # Drag only works parallel to the wind
                aero_force[j, 1] += 0.25*Fy_p
                aero_force[j, 2] += 0.25*Fz_p				
            else:
                if k == 0 or k == 1: #LEADING EDGE
                    aero_force[j, 0] += 0.25*Fx_p  # Drag only works parallel to the wind
                    aero_force[j, 1] += 0.5*0.75*Fy_p
                    aero_force[j, 2] += 0.5*0.75*Fz_p
                elif k == 2 or k == 3: #TRAILING EDGE
                    aero_force[j, 0] += 0.25*Fx_p  # Drag only works parallel to the wind
                    aero_force[j, 1] += 0.5*0.25*Fy_p
                    aero_force[j, 2] += 0.5*0.25*Fz_p				
			
    return np.array(aero_force)


def extract_pulley_connectivity(
    points, bridle_ci, bridle_cj, PULLEY_POINT_INDICES, NUMBER_OF_PULLEYS_IN_BACK_LINES
):
    # PULLEY_POINT_INDICES = PULLEY_DATA["POINT_INDICES"]
    # NUMBER_OF_PULLEYS_IN_BACK_LINES = PULLEY_DATA["NUMBER_OF_PULLEYS_IN_BACK_LINES"]

    # print(f'PULLEY_POINT_INDICES {len(PULLEY_POINT_INDICES)}, NUMBER_OF_PULLEYS_IN_BACK_LINES: {NUMBER_OF_PULLEYS_IN_BACK_LINES}')
    # Sorting pulleys based on z-coordinate
    sorted_pulley_point_indices = sorted(
        PULLEY_POINT_INDICES, key=lambda index: points[index][2]
    )

    # TODO: this is where you input the number of pulleys in the back-lines, that have a different feature
    # for pulleys where the line is below the pulley point
    pulley_point_indices_line_below = sorted_pulley_point_indices[
        :NUMBER_OF_PULLEYS_IN_BACK_LINES
    ]
    if (
        len(PULLEY_POINT_INDICES) > NUMBER_OF_PULLEYS_IN_BACK_LINES
    ):  # if there are pulleys in the front lines
        pulley_point_indices_line_above = sorted_pulley_point_indices[
            NUMBER_OF_PULLEYS_IN_BACK_LINES:
        ]
    else:
        pulley_point_indices_line_above = []  # this should be the case for the V3.25

    all_possible_line_indices = []

    for idx, (idx_bridle_node_i, idx_bridle_node_j) in enumerate(
        zip(bridle_ci, bridle_cj)
    ):  # loop through each bridle line
        # if the current line its index i OR j  is a pulley point
        # AND the other line is LOWER than the pulley_point, i.e. pulley line is BELOW the pulley
        if (
            idx_bridle_node_i in pulley_point_indices_line_below
            and points[idx_bridle_node_i][2] > points[idx_bridle_node_j][2]
        ) or (
            idx_bridle_node_j in pulley_point_indices_line_below
            and points[idx_bridle_node_j][2] > points[idx_bridle_node_i][2]
        ):
            all_possible_line_indices.append(idx)

        # if the current line its index i OR j  is a pulley point
        # AND the other line is HIGHER than the pulley_point, i.e. pulley line is ABOVE the pulley
        if (
            idx_bridle_node_i in pulley_point_indices_line_above
            and points[idx_bridle_node_i][2] < points[idx_bridle_node_j][2]
        ) or (
            idx_bridle_node_j in pulley_point_indices_line_above
            and points[idx_bridle_node_j][2] < points[idx_bridle_node_i][2]
        ):
            all_possible_line_indices.append(idx)

    # loop through all_possible_line_indices, twice to try and find the matching line pair
    pulley_line_pair_indices = {}
    pulley_line_indices = []

    for line_index_1 in all_possible_line_indices:
        for line_index_2 in all_possible_line_indices:
            # break if the same line is compared
            if line_index_1 == line_index_2:
                break

            # list the point_indices of the two lines
            point_indices = [
                bridle_ci[line_index_1],
                bridle_cj[line_index_1],
                bridle_ci[line_index_2],
                bridle_cj[line_index_2],
            ]
            # sort the point_indices based on the z-coordinate of the points[index]
            sorted_indices = sorted(point_indices, key=lambda index: points[index][2])

            # IF the pulley line is BELOW the pulley point, i.e. IF LAST two indices correspond to the same point
            # AND the index is in the pulley_index list (i.e. it is actually a pulley)
            # AND the line is not already used, i.e. not in the used_pulley_line_indices set
            if (
                sorted_indices[2] == sorted_indices[3]
                and sorted_indices[3] in pulley_point_indices_line_below
                and line_index_1 not in pulley_line_indices
                and line_index_2 not in pulley_line_indices
            ):
                # Append the line indices, to the pulley_line_indices
                pulley_line_indices.append(line_index_1)
                pulley_line_indices.append(line_index_2)
                # Make new key indices for the pulley line pair indices
                pulley_line_pair_indices[str(line_index_1)] = line_index_2

            # IF the pulley line is ABOVE the pulley point, i.e. IF FIRST indices correspond to the same point
            # AND the index is in the pulley_index list (i.e. it is actually a pulley)
            # AND the line is not already used, i.e. not in the used_pulley_line_indices set
            elif (
                sorted_indices[0] == sorted_indices[1]
                and sorted_indices[0] in pulley_point_indices_line_above
                and line_index_1 not in pulley_line_indices
                and line_index_2 not in pulley_line_indices
            ):
                # Append the line indices, to the pulley_line_indices
                pulley_line_indices.append(line_index_1)
                pulley_line_indices.append(line_index_2)
                # Make new key indices for the pulley line pair indices
                pulley_line_pair_indices[str(line_index_1)] = line_index_2

    # i know that line key and line value make up a pulley
    # so i want to plot both line key and line value
    # extract them separately and append
    pulley_ci_key = [
        bridle_ci[int(key_index)] for key_index in pulley_line_pair_indices.keys()
    ]
    pulley_cj_key = [
        bridle_cj[int(key_index)] for key_index in pulley_line_pair_indices.keys()
    ]
    pulley_ci_value = [
        bridle_ci[int(value_index)] for value_index in pulley_line_pair_indices.values()
    ]
    pulley_cj_value = [
        bridle_cj[int(value_index)] for value_index in pulley_line_pair_indices.values()
    ]
    pulley_ci = np.concatenate((pulley_ci_key, pulley_ci_value))
    pulley_cj = np.concatenate((pulley_cj_key, pulley_cj_value))

    pulley_data = {}
    pulley_data["line_indices"] = np.ndarray.flatten(np.array(pulley_line_indices))
    pulley_data["line_pair_indices"] = pulley_line_pair_indices
    pulley_data["ci"] = pulley_ci
    pulley_data["cj"] = pulley_cj

    ##TODO: added for implementing Alex's framework
    additional_dict = {}
    for i in range(0,len(pulley_data['line_indices']),2):

        # line 1, key: line 1 and value: line 2 data
        line_1_key = str(pulley_data["line_indices"][i])
        line_1_idx_p3 = bridle_ci[pulley_data["line_indices"][i+1]]
        line_1_idx_p4 = bridle_cj[pulley_data["line_indices"][i+1]]
        line_1_rest_length_p3p4 = np.linalg.norm(points[line_1_idx_p3]-points[line_1_idx_p4])
        additional_dict[line_1_key] = np.array([line_1_idx_p3,line_1_idx_p4,line_1_rest_length_p3p4])

        # line 2, key: line 2 and value: line 1 data
        line_2_key = str(pulley_data["line_indices"][i+1])
        line_2_idx_p3 = bridle_ci[pulley_data["line_indices"][i]]
        line_2_idx_p4 = bridle_cj[pulley_data["line_indices"][i]]
        line_2_rest_length_p3p4 = np.linalg.norm(points[line_2_idx_p3]-points[line_2_idx_p4])
        additional_dict[line_2_key] = np.array([line_2_idx_p3,line_2_idx_p4,line_2_rest_length_p3p4])

    pulley_data['other_line_pair'] = additional_dict

    return pulley_data