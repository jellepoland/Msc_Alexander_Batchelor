#%% First defining some functions
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


def extract_points_and_connectivity(simulation_directory, surfplan_file):
    """Extracts the points and connectivity of the V3
    This should be done from the surfplan_file, but is instead harcoded
    """
    points_struc = extract_points_struc(simulation_directory)
    bridle_ci, bridle_cj = extract_bridle_connectivity()
    plate_point_indices = extract_plate_point_indices()
    wing_ci, wing_cj = extract_wing_connectivity(plate_point_indices)
    te_line_indices = extract_te_line_indices(plate_point_indices, wing_ci, wing_cj)
    tube_line_indices = extract_tube_line_indices()

    return (
        np.array(points_struc),
        bridle_ci,
        bridle_cj,
        plate_point_indices,
        wing_ci,
        wing_cj,
        te_line_indices,
        tube_line_indices,
    )

#%% Calling those functions

points_ini = np.load('/home/jellepoland/surfdrive/phd/code/Msc_Alexander_Batchelor/code_Validation/saddle_form_kite/surfplan_points_V3.npy')

bridle_ci, bridle_cj = extract_bridle_connectivity()
plate_point_indices = extract_plate_point_indices()
wing_ci, wing_cj = extract_wing_connectivity(plate_point_indices)

wing_connectivity = np.column_stack((wing_ci, wing_cj))
bridle_connectivity = np.column_stack((bridle_ci, bridle_cj))
connections_kite = np.vstack((wing_connectivity, bridle_connectivity))

# structural_mesher.calculate_edge_lengths(
#     config.kite.connectivity.bridle_ci, config.kite.connectivity.bridle_cj, variables.points_ini)

# %%
