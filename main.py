import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import skvideo.io
from scipy.spatial.transform import Rotation as R
# from tqdm import tqdm

# Makeup world coordinates of the board
BOARD_W = 5
BOARD_H = 7
SQUARE_LENGTH = 0.03  # meters
SPACE_LENGTH = 0.006  # meters

def detect_markers(image):
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        image, aruco_dict, parameters=parameters
    )
    return corners, ids


def get_board_world_coords_straight():
    # Makeup world coordinates of the board
    BOARD_W = 5
    BOARD_H = 7
    SQUARE_LENGTH = 0.03  # meters
    SPACE_LENGTH = 0.006  # meters
    BOARD_WORLD_COORDS = np.zeros((BOARD_W * BOARD_H, 4, 3), dtype=np.float32)
    for i in range(BOARD_H):
        for j in range(BOARD_W):
            x = j * (SQUARE_LENGTH + SPACE_LENGTH)
            y = i * (SQUARE_LENGTH + SPACE_LENGTH)
            BOARD_WORLD_COORDS[i * BOARD_W + j] = np.array(
                [
                    [x, y, 0],
                    [x + SQUARE_LENGTH, y, 0],
                    [x + SQUARE_LENGTH, y + SQUARE_LENGTH, 0],
                    [x, y + SQUARE_LENGTH, 0],
                ],
                dtype=np.float32,
            )
    return BOARD_WORLD_COORDS


def get_board_world_coords():
    BOARD_WORLD_COORDS = np.zeros((BOARD_W * BOARD_H, 4, 3), dtype=np.float32)
    for i in range(BOARD_H):
        for j in range(BOARD_W):
            id = (BOARD_H - 1 - i) * BOARD_W + (BOARD_W - 1 - j)
            plane_x = j * (SQUARE_LENGTH + SPACE_LENGTH)
            plane_y = i * (SQUARE_LENGTH + SPACE_LENGTH)

            if id < 20:  # lower plane
                x = plane_y - 3 * (SQUARE_LENGTH + SPACE_LENGTH)
                y = plane_x
                z = 0
                BOARD_WORLD_COORDS[id] = np.array(
                    [
                        [x + SQUARE_LENGTH, y + SQUARE_LENGTH, z],
                        [x + SQUARE_LENGTH, y, z],
                        [x, y, z],
                        [x, y + SQUARE_LENGTH, z],
                    ]
                )
            else:
                x = 0
                y = plane_x
                z = 3 * (SQUARE_LENGTH + SPACE_LENGTH) - plane_y
                BOARD_WORLD_COORDS[id] = np.array(
                    [
                        [x, y + SQUARE_LENGTH, z - SQUARE_LENGTH],
                        [x, y, z - SQUARE_LENGTH],
                        [x, y, z],
                        [x, y + SQUARE_LENGTH, z],
                    ]
                )
    # visualize the points in 3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(
    #     BOARD_WORLD_COORDS[:, :, 0],
    #     BOARD_WORLD_COORDS[:, :, 1],
    #     BOARD_WORLD_COORDS[:, :, 2],
    # )
    # # Show tooltip as the index of the point
    # for i in range(BOARD_W * BOARD_H):
    #     vis_id = 1
    #     ax.text(
    #         BOARD_WORLD_COORDS[i, vis_id, 0],
    #         BOARD_WORLD_COORDS[i, vis_id, 1],
    #         BOARD_WORLD_COORDS[i, vis_id, 2],
    #         str(i),
    #     )
    #     # Hightlight the texted point
    #     ax.scatter(
    #         BOARD_WORLD_COORDS[i, vis_id, 0],
    #         BOARD_WORLD_COORDS[i, vis_id, 1],
    #         BOARD_WORLD_COORDS[i, vis_id, 2],
    #         c="r",
    #     )
    # plt.show()
    return BOARD_WORLD_COORDS.astype(np.float32)

def get_camera_pose(image):
    corners, ids = detect_markers(image)

    # # Draw detected markers
    # frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    # # Show the frame
    # plt.figure()
    # plt.imshow(frame_markers)
    # plt.show()

    
    # Calculate camera intrinsics
    # board_world_coords = get_board_world_coords_straight()[ids.flatten()]
    # print(image.shape[1::-1])
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    #     board_world_coords.reshape(1, 35 * 4, 3), np.array(corners).reshape(1, 35 * 4, 2), image.shape[1::-1], None, None
    # )
    # rvecs = rvecs[0]
    # tvecs = tvecs[0]

    # Load camera intrinsics
    # board_world_coords = get_board_world_coords()[ids.flatten()]
    camera_params = np.load("/Users/qingchengzhao/Nutstore Files/Nutstore/课程/Term-V/CS180/Final/AR/data/intrinsic.npy", allow_pickle=True).item()
    initial_mtx = camera_params["mtx"]
    initial_dist = camera_params["dist"]
    mtx = initial_mtx
    dist = initial_dist
    
    # Calculate camera intrinsics
    # ids_mask = ids < 100
    board_world_coords = get_board_world_coords()[ids]
    # ret, mtx, dist, _, _ = cv2.calibrateCamera(
    #     board_world_coords[ids_mask].reshape(1, -1, 3), np.array(corners)[ids_mask].reshape(1, -1, 2), image.shape[1::-1], initial_mtx, initial_dist,
    #     flags=cv2.CALIB_USE_INTRINSIC_GUESS
    # )
    # print("Camera matrix:")
    # print(mtx)
    # np.save("/Users/qingchengzhao/Nutstore Files/Nutstore/课程/Term-V/CS180/Final/AR/data/intrinsic.npy", {"mtx": mtx, "dist": dist})

    # board_world_coords = get_board_world_coords()[ids]


    # Get camera pose
    ids = ids.flatten()
    ret, rvecs, tvecs = cv2.solvePnP(
        (board_world_coords.reshape(-1, 3)).copy(),
        (np.array(corners).reshape(-1, 2)).copy(),
        mtx,
        dist
    )

    # Cast world coordinates to camera coordinates to check if they are correct
    # (they should be the same as the detected marker points)
    # camera_world_coords = cv2.projectPoints(
    #     board_world_coords.reshape(-1, 3), rvecs, tvecs, mtx, dist
    # )[0].reshape(-1, 1, 4, 2)
    # Draw camera world coordinates
    # frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), camera_world_coords, ids)
    # Show the frame
    # plt.figure()
    # plt.imshow(frame_markers)
    # plt.show()

    rotation = R.from_rotvec(rvecs.reshape(1, 3))

    # Get Camera Pose in World Coordinate System
    camera_location = tvecs.reshape(3, 1)

    return mtx, dist, rvecs, tvecs, camera_location, rotation


def plot_cube_in_frame(frame, cube_uv):
    plt.figure()
    plt.imshow(frame)
    plt.scatter(cube_uv[:, 0], cube_uv[:, 1])
    # Link points of the cube
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        plt.plot(
            [cube_uv[i, 0], cube_uv[j, 0]],
            [cube_uv[i, 1], cube_uv[j, 1]],
        )
    for i, j in [(4, 5), (5, 6), (6, 7), (7, 4)]:
        plt.plot(
            [cube_uv[i, 0], cube_uv[j, 0]],
            [cube_uv[i, 1], cube_uv[j, 1]],
        )
    for i, j in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        plt.plot(
            [cube_uv[i, 0], cube_uv[j, 0]],
            [cube_uv[i, 1], cube_uv[j, 1]],
        )

def create_bbox(corners, width=6):
    # Create bounding box for corner points
    # corners: N x 2
    radius = width // 2
    return np.concatenate([corners - radius, np.ones_like(corners) * width], axis=1)


def track_markers(video_data):
    start_corners, start_ids = detect_markers(video_data[0])
    start_corners = np.array(start_corners).reshape(-1, 2)
    start_corners = create_bbox(start_corners, width=12)
    # Show bounding box
    plt.figure()
    plt.imshow(video_data[0])
    for corner in start_corners:
        plt.gca().add_patch(Rectangle(corner[:2], corner[2], corner[3], linewidth=1, edgecolor="r", facecolor="none"))
    plt.show()
    print(start_corners.shape)
        
    # Track markers
    tracker = cv2.legacy.MultiTracker_create()
    for corner in start_corners:
        tracker.add(cv2.legacy.TrackerCSRT_create(), video_data[0], tuple(corner))

    # Track markers in each frame
    for index, frame in tqdm(enumerate(video_data)):
        ret, corners = tracker.update(frame)
        corners = np.array(corners).reshape(-1, 2)
        corners = create_bbox(corners)
        # Show bounding box
        plt.figure()
        plt.imshow(frame)
        for corner in corners:
            plt.gca().add_patch(Rectangle(corner[:2], corner[2], corner[3], linewidth=1, edgecolor="r", facecolor="none"))
        plt.savefig(
            f"/Users/qingchengzhao/Nutstore Files/Nutstore/课程/Term-V/CS180/Final/AR/ouput/track_{index:06d}.png"
        )
        plt.close()
        print(corners.shape)

# Load video
video_data = skvideo.io.vread(
    "/Users/qingchengzhao/Nutstore Files/Nutstore/课程/Term-V/CS180/Final/AR/data/big.mp4"
)
# np.save("data/big.npy", video_data)
# exit(0)
# video_data = np.load(
#     "/Users/qingchengzhao/Nutstore Files/Nutstore/课程/Term-V/CS180/Final/AR/data/big.npy"
# )
frame_num, height, width, _ = video_data.shape
# for index, frame in tqdm(enumerate(video_data)):
#     # corners, ids = detect_markers(frame[:, :, ::-1])
#     # frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
#     # cv2.imwrite(
#     #     f"/Users/qingchengzhao/Nutstore Files/Nutstore/课程/Term-V/CS180/Final/AR/ouput/{index:06d}.png",
#     #     frame_markers,
#     # )
#     # if index == 10:
#     #     break
#     # continue
#     mtx, dist, rvecs, tvecs, cam_loc, cam_rot = get_camera_pose(frame)

#     # Cast world coordinates of a unit cube to camera coordinates
#     cube_world_coords = (
#         np.array(
#             [
#                 [0, 0, 0],
#                 [1, 0, 0],
#                 [1, 1, 0],
#                 [0, 1, 0],
#                 [0, 0, 1],
#                 [1, 0, 1],
#                 [1, 1, 1],
#                 [0, 1, 1],
#             ],
#             dtype=np.float32,
#         )
#         * (0.036)
#         * 2
#     )

#     cube_uv = cv2.projectPoints(cube_world_coords, rvecs, tvecs, mtx, np.zeros(5))[
#         0
#     ].reshape(8, 2)

#     # Draw points of the cube
#     plot_cube_in_frame(frame, cube_uv)
#     # plt.show()
#     # Save to file
#     plt.savefig(
#         f"/Users/qingchengzhao/Nutstore Files/Nutstore/课程/Term-V/CS180/Final/AR/ouput/cube_{index:06d}.png"
#     )
#     plt.close()
#     # if index == 20:
#     #     break
# exit(0)

cam_loc_list = []
cam_rot_list = []
for frame in video_data:
    mtx, dist, rvecs, tvecs, cam_loc, cam_rot = get_camera_pose(frame)
    cam_loc_list.append(cam_loc)
    cam_rot_list.append(cam_rot)

# Cast world coordinates to camera coordinates to check if they are correct
# (they should be the same as the detected marker points)
# camera_world_coords = cv2.projectPoints(
#     board_world_coords, rvecs[0], tvecs[0], mtx, dist
# )[0].reshape(35, 1, 4, 2)

# # Draw camera world coordinates
# frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), camera_world_coords, ids)
# # Show the frame
# plt.figure()
# plt.imshow(frame_markers)
# plt.show()

# Cast world coordinates of a unit cube to camera coordinates
# cube_world_coords = np.array(
#     [
#         [0, 0, 0],
#         [1, 0, 0],
#         [1, 1, 0],
#         [0, 1, 0],
#         [0, 0, 1],
#         [1, 0, 1],
#         [1, 1, 1],
#         [0, 1, 1],
#     ],
#     dtype=np.float32,
# ) * (square_length + space_length) * 7

# cube_camera_coords = cv2.projectPoints(
#     cube_world_coords, rvecs[0], tvecs[0], mtx, np.zeros(5)
# )[0].reshape(8, 2)
# print("Cube camera coordinates:")
# print(cube_camera_coords)

# # Draw points of the cube
# plt.figure()
# plt.imshow(frame)
# plt.scatter(cube_camera_coords[:, 0], cube_camera_coords[:, 1])
# # Link points of the cube
# for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
#     plt.plot(
#         [cube_camera_coords[i, 0], cube_camera_coords[j, 0]],
#         [cube_camera_coords[i, 1], cube_camera_coords[j, 1]],
#     )
# for i, j in [(4, 5), (5, 6), (6, 7), (7, 4)]:
#     plt.plot(
#         [cube_camera_coords[i, 0], cube_camera_coords[j, 0]],
#         [cube_camera_coords[i, 1], cube_camera_coords[j, 1]],
#     )
# for i, j in [(0, 4), (1, 5), (2, 6), (3, 7)]:
#     plt.plot(
#         [cube_camera_coords[i, 0], cube_camera_coords[j, 0]],
#         [cube_camera_coords[i, 1], cube_camera_coords[j, 1]],
#     )
# plt.show()

try:
    import bpy
except ImportError:
    print("This script must be run from Blender")
    quit()


def set_camera_params(camera, mtx):
    # Set camera intrinsics
    camera.data.lens_unit = "FOV"
    camera.data.angle = 2 * np.arctan(width / (2 * mtx[0, 0]))

    camera.data.shift_x = mtx[0, 2] / width - 0.5
    camera.data.shift_y = mtx[1, 2] / height - 0.5
    camera.data.sensor_width = mtx[0, 0]
    camera.data.sensor_height = mtx[1, 1]
    camera.data.clip_start = 0.1
    camera.data.clip_end = 1000


def set_camera_pose(camera, loc, rot):
    # Process rotation
    to_blender_mtx = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    camera_euler = R.from_matrix(rot.as_matrix().reshape(3, 3).T @ to_blender_mtx).as_euler("xyz")
    # Process translation
    camera_loc = -rot.as_matrix().reshape(3, 3).T @ loc.reshape(3, 1)

    # Set camera rotation
    camera.rotation_mode = "XYZ"
    camera.rotation_euler[0] = camera_euler[0]
    camera.rotation_euler[1] = camera_euler[1]
    camera.rotation_euler[2] = camera_euler[2]

    # Set camera translation
    camera.location[0] = camera_loc[0]
    camera.location[1] = camera_loc[1]
    camera.location[2] = camera_loc[2]


camera = bpy.data.objects["Camera"]

set_camera_params(camera, mtx)

# Clear all keyframes
camera.animation_data_clear()

# Set camera pose for each frame
for i in range(frame_num):
    set_camera_pose(camera, cam_loc_list[i], cam_rot_list[i])
    camera.keyframe_insert(data_path="location", frame=i)
    camera.keyframe_insert(data_path="rotation_euler", frame=i)

# Set frame range
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = frame_num - 1