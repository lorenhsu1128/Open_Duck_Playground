

# import jax
# import jax.numpy as np

import numpy as np


# Tracking rewards.
def reward_tracking_lin_vel(commands, local_vel, tracking_sigma):
    # lin_vel_error = np.sum(np.square(commands[:2] - local_vel[:2]))
    # return np.nan_to_num(np.exp(-lin_vel_error / self._config.reward_config.tracking_sigma))
    y_tol = 0.1
    error_x = np.square(commands[0] - local_vel[0])
    error_y = np.clip(np.abs(local_vel[1] - commands[1]) - y_tol, 0.0, None)
    lin_vel_error = error_x + np.square(error_y)
    return np.nan_to_num(np.exp(-lin_vel_error / tracking_sigma))


def reward_tracking_ang_vel(commands, ang_vel, tracking_sigma):
    ang_vel_error = np.square(commands[2] - ang_vel[2])
    return np.nan_to_num(np.exp(-ang_vel_error / tracking_sigma))


# Base-related rewards.


def cost_lin_vel_z(global_linvel):
    return np.nan_to_num(np.square(global_linvel[2]))


def cost_ang_vel_xy(global_angvel):
    return np.nan_to_num(np.sum(np.square(global_angvel[:2])))


def cost_orientation(torso_zaxis):
    return np.nan_to_num(np.sum(np.square(torso_zaxis[:2])))


def cost_base_height(base_height, base_height_target):
    return np.nan_to_num(np.square(base_height - base_height_target))


def reward_base_y_swing(base_y_speed, freq, amplitude, t, tracking_sigma):
    target_y_speed = amplitude * np.sin(2 * np.pi * freq * t)
    y_speed_error = np.square(target_y_speed - base_y_speed)
    return np.nan_to_num(np.exp(-y_speed_error / tracking_sigma))


# Energy related rewards.


def cost_torques(torques):
    return np.nan_to_num(np.sum(np.square(torques)))
    # return np.nan_to_num(np.sum(np.abs(torques)))


def cost_energy(qvel, qfrc_actuator):
    return np.nan_to_num(np.sum(np.abs(qvel) * np.abs(qfrc_actuator)))


def cost_action_rate(act, last_act):
    c1 = np.nan_to_num(np.sum(np.square(act - last_act)))
    return c1


# Other rewards.


def cost_joint_pos_limits(qpos, soft_lowers, soft_uppers):
    out_of_limits = -np.clip(qpos - soft_lowers, None, 0.0)
    out_of_limits += np.clip(qpos - soft_uppers, 0.0, None)
    return np.nan_to_num(np.sum(out_of_limits))


def cost_stand_still(commands, qpos, qvel, default_pose, ignore_head=False):
    # TODO no hard coded slices
    cmd_norm = np.linalg.norm(commands[:3])
    if not ignore_head:
        pose_cost = np.sum(np.abs(qpos - default_pose))
        vel_cost = np.sum(np.abs(qvel))
    else:
        left_leg_pos = qpos[:5]
        right_leg_pos = qpos[9:]
        left_leg_vel = qvel[:5]
        right_leg_vel = qvel[9:]
        left_leg_default = default_pose[:5]
        right_leg_default = default_pose[9:]
        pose_cost = np.sum(np.abs(left_leg_pos - left_leg_default)) + np.sum(
            np.abs(right_leg_pos - right_leg_default)
        )
        vel_cost = np.sum(np.abs(left_leg_vel)) + np.sum(np.abs(right_leg_vel))

    return np.nan_to_num(pose_cost + vel_cost) * (cmd_norm < 0.01)


def cost_termination(done):
    return done


def reward_alive():
    return np.array(1.0)


# Pose-related rewards.


def cost_head_pos(joints_qpos, joints_qvel, cmd):
    move_cmd_norm = np.linalg.norm(cmd[:3])
    head_cmd = cmd[3:]
    head_pos = joints_qpos[5:9]
    # head_vel = joints_qvel[5:9]

    # target_head_qvel = np.zeros_like(head_cmd)

    head_pos_error = np.sum(np.square(head_pos - head_cmd))

    # head_vel_error = np.sum(np.square(head_vel - target_head_qvel))

    return np.nan_to_num(head_pos_error) * (move_cmd_norm > 0.01)
    # return np.nan_to_num(head_pos_error + head_vel_error)


# FIXME
def cost_joint_deviation_hip(qpos, cmd, hip_indices, default_pose):
    cost = np.sum(np.abs(qpos[hip_indices] - default_pose[hip_indices]))
    cost *= np.abs(cmd[1]) > 0.1
    return np.nan_to_num(cost)


# FIXME
def cost_joint_deviation_knee(qpos, knee_indices, default_pose):
    return np.nan_to_num(
        np.sum(np.abs(qpos[knee_indices] - default_pose[knee_indices]))
    )


# FIXME
def cost_pose(qpos, default_pose, weights):
    return np.nan_to_num(np.sum(np.square(qpos - default_pose) * weights))


# Feet related rewards.


# FIXME
def cost_feet_slip(contact, global_linvel):
    body_vel = global_linvel[:2]
    reward = np.sum(np.linalg.norm(body_vel, axis=-1) * contact)
    return np.nan_to_num(reward)


# FIXME
def cost_feet_clearance(feet_vel, foot_pos, max_foot_height):
    # feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = np.sqrt(np.linalg.norm(vel_xy, axis=-1))
    # foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = np.abs(foot_z - max_foot_height)
    return np.nan_to_num(np.sum(delta * vel_norm))


# FIXME
def cost_feet_height(swing_peak, first_contact, max_foot_height):
    error = swing_peak / max_foot_height - 1.0
    return np.nan_to_num(np.sum(np.square(error) * first_contact))


# FIXME
def reward_feet_air_time(
    air_time, first_contact, commands, threshold_min=0.1, threshold_max=0.5  # 0.2
):
    cmd_norm = np.linalg.norm(commands[:3])
    air_time = (air_time - threshold_min) * first_contact
    air_time = np.clip(air_time, max=threshold_max - threshold_min)
    reward = np.sum(air_time)
    reward *= cmd_norm > 0.01  # No reward for zero commands.
    return np.nan_to_num(reward)


# FIXME
def reward_feet_phase(foot_pos, rz):
    # Reward for tracking the desired foot height.
    # foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    # rz = gait.get_rz(phase, swing_height=foot_height)
    error = np.sum(np.square(foot_z - rz))
    reward = np.exp(-error / 0.01)
    # TODO(kevin): Ensure no movement at 0 command.
    # cmd_norm = np.linalg.norm(commands)
    # reward *= cmd_norm > 0.1  # No reward for zero commands.
    return np.nan_to_num(reward)

# 新增雙腳不平行成本項，希望讓左右 leg_hip_pitch 儘量對稱
def cost_legs_asymmetry(joints_qpos, left_hip_pitch_idx, right_hip_pitch_idx, left_knee_idx, right_knee_idx):
    """Calculates the asymmetry cost between the left and right leg joints."""
    
    # Hip pitch joints should be opposite to each other, so their sum should be close to 0.
    hip_pitch_asymmetry = np.square(joints_qpos[left_hip_pitch_idx] + joints_qpos[right_hip_pitch_idx])
    
    # Knee joints should be roughly equal, so their difference should be close to 0.
    knee_asymmetry = np.square(joints_qpos[left_knee_idx] - joints_qpos[right_knee_idx])
    
    # You can also add other joints like hip_roll, hip_yaw, ankle if needed.
    
    return np.nan_to_num(hip_pitch_asymmetry + knee_asymmetry)