def normalize_vectors(x: pd.Series, y: pd.Series, z: pd.Series, eps: float = 1e-9):
    V = np.vstack([x.values, y.values, z.values]).T.astype(float)
    norms = np.linalg.norm(V, axis=1)
    safe_norms = np.where(norms > eps, norms, np.nan)
    U = V / safe_norms[:, None]
    return U[:, 0], U[:, 1], U[:, 2], norms

def read_imu_txt(path: str):
    # skip bad lines
    imu_data = pd.read_csv(path, sep=',', header=None, on_bad_lines='skip')

    # print("DataFrame shape:", imu_data.shape)
    imu_data = imu_data.dropna()
    # print(f"\nAfter cleaning: {imu_data.shape[0]} valid rows")

    if imu_data.shape[1] < 7:
        raise ValueError(f"Expected at least 7 columns, got {imu_data.shape[1]}.")

    return imu_data

def df_to_rpy(df: pd.DataFrame, deg: bool = False):
    Accx = df.iloc[:, 1]
    Accy = df.iloc[:, 2]
    Accz = df.iloc[:, 3]
    Magx = df.iloc[:, 4]
    Magy = df.iloc[:, 5]
    Magz = df.iloc[:, 6]
    acc_unit_x, acc_unit_y, acc_unit_z, _ = normalize_vectors(Accx, Accy, Accz)
    mag_unit_x, mag_unit_y, mag_unit_z, _ = normalize_vectors(Magx, Magy, Magz)

    pitch = np.arcsin(-acc_unit_x)
    roll = np.arctan2(acc_unit_y, acc_unit_z)

    sx, sy, sz = mag_unit_x, mag_unit_y, mag_unit_z
    theta = pitch
    phi   = roll

    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_p, cos_p = np.sin(phi),   np.cos(phi)

    y_comp = sy * cos_p + sz * sin_p
    x_comp = sx * cos_t + sy * sin_p * sin_t - sz * cos_p * sin_t

    yaw = np.arctan2(y_comp, x_comp)

    
    # find average of roll, pitch, yaw
    roll_avg = np.nanmean(roll) 
    pitch_avg = np.nanmean(pitch)
    yaw_avg = np.nanmean(yaw)
    # to degree
    if deg:
        roll_avg_deg = np.degrees(roll_avg)
        pitch_avg_deg = np.degrees(pitch_avg)
        yaw_avg_deg = np.degrees(yaw_avg)
        return roll_avg_deg, pitch_avg_deg, yaw_avg_deg


    return roll_avg, pitch_avg, yaw_avg


def rpy_to_Rot(r, p, y):
    # R = R_yaw * R_pitch * R_roll
    # R*x = (R_yaw * (R_pitch * (R_roll * x)))

    R_yaw = np.array([[np.cos(y), -np.sin(y), 0],
                    [np.sin(y), np.cos(y), 0],
                    [0, 0, 1]])

    R_pitch = np.array([[np.cos(p), 0, np.sin(p)],
                        [0, 1, 0],
                        [-np.sin(p), 0, np.cos(p)]])

    R_roll = np.array([[1, 0, 0],
                    [0, np.cos(r), -np.sin(r)],
                    [0, np.sin(r), np.cos(r)]])

    Rot = R_yaw @ R_pitch @ R_roll

    print(Rot)