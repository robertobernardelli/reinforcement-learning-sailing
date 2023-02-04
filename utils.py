import numpy as np
import random
import cv2
import cvzone
import time
from scipy import interpolate


def initialize_speed_config():
    """
    Initializes speed configuration
    """

    data = [
        [0, 0],
        [30, 4],
        [60, 7],
        [70, 7.5],
        [90, 7],
        [120, 6],
        [150, 5.2],
        [180, 5],
        [210, 5.2],
        [240, 6],
        [270, 7],
        [290, 7.5],
        [300, 7],
        [330, 4],
        [360, 0],
    ]

    x, y = zip(*data)

    tck = interpolate.splrep(x, y, s=0)

    return tck


def get_boat_speed(boat_heading, wind_heading, tck):
    """
    Returns the boat speed given the boat heading and wind heading
    """
    x = (((180 - boat_heading) % 360) - wind_heading - 180) % 360
    boat_speed = interpolate.splev(x, tck, der=0)

    return boat_speed


def degrees_to_radians(degrees):
    return degrees * np.pi / 180


def reposition_target(target_position):
    """
    After the target is reached, it is repositioned randomly on the map
    """
    target_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
    return target_position


def collision_with_boundaries(boat_position):
    if (
        boat_position[0] >= 500
        or boat_position[0] < 0
        or boat_position[1] >= 500
        or boat_position[1] < 0
    ):
        return 1
    else:
        return 0


def euclidean_distance(point1, point2):
    """
    Returns the euclidean distance between two points
    """
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_line_params(wind_direction):
    """
    Returns the start and end points of a line centered at (50, 50)
    with lenght 30 that represents the wind direction as tuples of integers
    """
    return (70, 40), (
        70 + int(60 * np.sin(np.deg2rad(wind_direction))),
        50 - int(60 * np.cos(np.deg2rad(wind_direction))),
    )


def draw_boat_rectangle(img, boat_position):
    """
    Draws a green point on the image, given the boat position. Useful for debugging.
    """
    cv2.rectangle(
        img,
        (int(boat_position[0]), int(boat_position[1])),
        (int(boat_position[0]) + 1, int(boat_position[1]) + 1),
        (0, 255, 0),
        3,
    )
    return img


def draw_boat(img, boat_position, boat_heading):
    """
    Draws the boat on the image, given the boat position and heading.
    It works also if the boat is near the boundaries of the image.
    """

    overlay_img = cv2.imread("boat_sprite.png", -1)
    boat_sprite_size = (50, 50)
    overlay_img = cv2.resize(overlay_img, boat_sprite_size)

    cols, rows = overlay_img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), (boat_heading + 180) % 360, 1)
    overlay_img = cv2.warpAffine(overlay_img, M, (cols, rows))

    # Get the size of the background image
    bg_height, bg_width = img.shape[:2]

    # Get the size of the overlay image
    overlay_height, overlay_width = overlay_img.shape[:2]

    # Calculate the top-left and bottom-right coordinates of the overlay image
    x, y = boat_position
    x -= overlay_width // 2
    y -= overlay_height // 2
    x2, y2 = x + overlay_width, y + overlay_height

    # Convert the coordinates to integers
    x, y, x2, y2 = map(int, (x, y, x2, y2))

    # Crop the overlay image if it is outside the bounds of the background image
    if x < 0:
        x_offset = -x
        x = 0
        overlay_img = overlay_img[:, x_offset:]
    if y < 0:
        y_offset = -y
        y = 0
        overlay_img = overlay_img[y_offset:, :]
    if x2 > bg_width:
        x2_offset = x2 - bg_width
        x2 = bg_width
        overlay_img = overlay_img[:, :-x2_offset]
    if y2 > bg_height:
        y2_offset = y2 - bg_height
        y2 = bg_height
        overlay_img = overlay_img[:-y2_offset:, :]

    img = cvzone.overlayPNG(img, overlay_img, [x, y])

    return img


def draw_sea_and_dashboard(wind_direction, boat_heading, boat_speed, total_reward):
    img = np.zeros((500, 700, 3), dtype="uint8")
    img[:, :] = (102, 0, 0)  # blue sea

    # Dashboard
    # img[500:600,0:500] = (192,192,192) # silver
    # img[500:600,0:100] = (0,0,0) # black

    # Display wind direction
    start_point, end_point = get_line_params(wind_direction)
    img = cv2.arrowedLine(
        img, start_point, end_point, (255, 255, 255), 1, tipLength=0.2
    )

    label_wind = f"wind: {int(wind_direction)} deg"
    img = cv2.putText(
        img,
        label_wind,
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    label_boat = (
        f"boat: {round(float(boat_speed), 1)} knts, {(180-boat_heading)%360} deg"
    )
    img = cv2.putText(
        img,
        label_boat,
        (150, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    label_reward = f"reward: {round(float(total_reward),2)}"
    img = cv2.putText(
        img,
        label_reward,
        (370, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return img


def draw_target(img, target_position):
    x, y = target_position
    cv2.rectangle(img, (x, y), (x + 1, y + 1), (0, 0, 255), 3)

    cv2.circle(img, target_position, 30, (0, 0, 255), 1)

    return img


def step():
    # Takes step after fixed time
    t_end = time.time() + 0.05
    k = -1
    while time.time() < t_end:
        if k == -1:
            k = cv2.waitKey(1)
        else:
            continue


def draw_boat_path(img, boat_path):
    """
    Draw the boat path
    """

    for i, position in enumerate(boat_path):
        x, y = position
        x, y = int(x), int(y)
        img = cv2.rectangle(img, (x, y), (x + 1, y + 1), (211, 211, 211))

    return img
