import numpy as np

def deg2rad(degrees):
    return degrees * np.pi / 180

def rotation_x(angle_deg):
    angle = deg2rad(angle_deg)
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle),  np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def rotation_y(angle_deg):
    angle = deg2rad(angle_deg)
    return np.array([
        [np.cos(angle), 0, np.sin(angle), 0],
        [0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [0, 0, 0, 1]
    ])

def rotation_z(angle_deg):
    angle = deg2rad(angle_deg)
    return np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle),  np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def translation_x(units):
    return np.array([
        [1, 0, 0, units],
        [0, 1, 0, 0    ],
        [0, 0, 1, 0    ],
        [0, 0, 0, 1    ]
    ])

# Apply transformations: roll -> translate -> yaw
R_roll = rotation_z(30)
T = translation_x(10)
R_yaw = rotation_y(10)

# Final transformation
combined_matrix = R_yaw @ T @ R_roll
print(combined_matrix)
