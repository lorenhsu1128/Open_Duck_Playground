# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Open Duck Mini V2. (based on Berkeley Humanoid)"""

from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from . import constants
from . import base as open_duck_mini_v2_base

# from playground.common.utils import LowPassActionFilter
from playground.common.poly_reference_motion import PolyReferenceMotion
from playground.common.rewards import (
    reward_tracking_lin_vel,
    reward_tracking_ang_vel,
    cost_torques,
    cost_action_rate,
    cost_stand_still,
    reward_alive,
    # cost_legs_asymmetry,  # <-- 左右腳對稱站立成本
    # reward_symmetry,  # <-- 左右腳對稱站立獎勵
    # reward_head_roll_zero,
    reward_head_orientation_walking,
    reward_head_straight_standing,
    # reward_hip_pitch_symmetry_standing,
    # reward_hips_straight_standing,
    reward_ideal_standing_hips,
    # reward_hip_pitch_in_turn,
    reward_asymmetric_turning_gait,
    reward_head_forward_tilt,
    head_straight_standing,
    ideal_standing_hips,
    reward_balance_with_head_motion,
)
from playground.open_duck_mini_v2.custom_rewards import reward_imitation

# if set to false, won't require the reference data to be present and won't compute the reference motions polynoms for nothing
# 模仿獎勵
USE_IMITATION_REWARD = True
USE_MOTOR_SPEED_LIMITS = True
# 隨機推力
USE_PUSH = True


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.25,
        dof_vel_scale=0.05,
        history_len=0,
        soft_joint_pos_limit_factor=0.95,
        max_motor_velocity=5.24,  # rad/s
        # 提高 entropy_cost 會在獎勵系統中加入一個新的目標：「保持你決策的不確定性」。
        # 這會直接懲罰那些過於自信、單一的策略，強迫 AI 在整個訓練過程中都保持探索的熱情，從而更容易跳出局部最優解。
        # 明確地定義 entropy_cost，覆蓋掉預設值
        # 建議值: 0.01 或 0.015
        entropy_cost=0.03, #0.02

        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            action_min_delay=0,  # env steps
            action_max_delay=3,  # env steps
            imu_min_delay=0,  # env steps
            imu_max_delay=3,  # env steps
            scales=config_dict.create(
                hip_pos=0.03,  # rad, for each hip joint
                knee_pos=0.05,  # rad, for each knee joint
                ankle_pos=0.08,  # rad, for each ankle joint
                joint_vel=2.5,  # rad/s # Was 1.5
                gravity=0.1,
                linvel=0.1,
                gyro=0.1,
                accelerometer=0.05,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                # tracking_lin_vel=2.5,
                # tracking_ang_vel=6.0,
                tracking_lin_vel=4.0,
                tracking_ang_vel=10.0,
                torques=-1.0e-3,
                action_rate=-0.5,  # was -1.5
                stand_still=-0.2,  # was -1.0 TODO try to relax this a bit ?
                alive=20.0,
                imitation=1.5, # 1.5 穩定版為2.0
                # symmetry=4.5, # 雙腳平行就加分，有效果，可以修正雙腳站立時不正的狀況 5.0站姿歪斜
                # legs_asymmetry=-1.5,  # <-- 雙腳不平行就扣分 第一次-0.5無效果
                # head_roll_zero=4.0, #頭部roll維持0度的獎勵
                head_orientation_walking=5.0, #走路時頭部儘量不動的獎勵
                head_straight_standing=5.5, #站立時頭部回正的獎勵 4.0 +- 0.5 穩定版為5.5
                # hip_pitch_symmetry_standing=2.0, #站立時髖部回正的獎勵
                # hips_straight_standing=3.0,
                ideal_standing_hips=6.0, #站立時理想站姿的獎勵 穩定版為4.0
                # hip_pitch_in_turn=2.0, # 轉彎時的抬腳獎勵，可以從 1.0 ~ 2.0 開始嘗試
                asymmetric_turning_gait=0.0, #2.5
                # 為頭部前傾設定一個權重
                head_forward_tilt=5.0,
                balance_with_head_motion=10.0,
            ),
            tracking_sigma=0.01,  # was working at 0.01
        ),
        push_config=config_dict.create(
            enable=USE_PUSH,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        # 將 x 軸線速度的範圍從 [-0.15, 0.15] 擴大
        # lin_vel_x=[-0.15, 0.15], 
        lin_vel_x=[-0.20, 0.20], 
        lin_vel_y=[-0.2, 0.2],
        ang_vel_yaw=[-1.5, 1.5],  # [-1.0, 1.0]
        neck_pitch_range=[-0.34, 1.1],
        head_pitch_range=[-0.78, 0.78],
        head_yaw_range=[-1.5, 1.5],
        head_roll_range=[-0.5, 0.5],  #原本數值-0.5, 0.5
        head_range_factor=1.0,  # to make it easier
    )


class Joystick(open_duck_mini_v2_base.OpenDuckMiniV2Env):
    """Track a joystick command."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=constants.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

        

    def _post_init(self) -> None:

        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_actuator = self._mj_model.keyframe(
            "home"
        ).ctrl  # ctrl of all the actual joints (no floating base and no backlash)

        if USE_IMITATION_REWARD:
            self.PRM = PolyReferenceMotion(
                "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
            )

        # Note: First joint is freejoint.
        # get the range of the joints
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        # weights for computing the cost of each joints compared to a reference pose
        self._weights = jp.array(
            [
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,  # left leg.
                # 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, #head
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,  # right leg.
            ]
        )

        # 新增以下程式碼來獲取關節索引
        self._left_hip_pitch_idx = constants.JOINTS_ORDER_NO_HEAD.index('left_hip_pitch')
        self._right_hip_pitch_idx = constants.JOINTS_ORDER_NO_HEAD.index('right_hip_pitch')
        self._left_knee_idx = constants.JOINTS_ORDER_NO_HEAD.index('left_knee')
        self._right_knee_idx = constants.JOINTS_ORDER_NO_HEAD.index('right_knee')

        self._njoints = self._mj_model.njnt  # number of joints
        self._actuators = self._mj_model.nu  # number of actuators

        self._torso_body_id = self._mj_model.body(constants.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._site_id = self._mj_model.site("imu").id

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in constants.FEET_SITES]
        )

        # self._floor_geom_id = self._mj_model.geom("floor").id

        # 新的、更靈活的程式碼
        all_geom_names = [self._mj_model.geom(i).name for i in range(self._mj_model.ngeom)]
        floor_names = [name for name in all_geom_names if "floor" in name]
        self._floor_geom_ids = jp.array([self._mj_model.geom(name).id for name in floor_names])

        self._feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in constants.FEET_GEOMS]
        )

        foot_linvel_sensor_adr = []
        for site in constants.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        # # noise in the simu?
        qpos_noise_scale = np.zeros(self._actuators)

        hip_ids = [
            idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_hip" in j
        ]
        knee_ids = [
            idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_knee" in j
        ]
        ankle_ids = [
            idx for idx, j in enumerate(constants.JOINTS_ORDER_NO_HEAD) if "_ankle" in j
        ]

        qpos_noise_scale[hip_ids] = self._config.noise_config.scales.hip_pos
        qpos_noise_scale[knee_ids] = self._config.noise_config.scales.knee_pos
        qpos_noise_scale[ankle_ids] = self._config.noise_config.scales.ankle_pos
        # qpos_noise_scale[faa_ids] = self._config.noise_config.scales.faa_pos
        self._qpos_noise_scale = jp.array(qpos_noise_scale)

        # self.action_filter = LowPassActionFilter(
        #     1 / self._config.ctrl_dt, cutoff_frequency=37.5
        # )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q  # the complete qpos
        # print(f'DEBUG0 init qpos: {qpos}')
        qvel = jp.zeros(self.mjx_model.nv)

        # init position/orientation in environment
        # x=+U(-0.05, 0.05), y=+U(-0.05, 0.05), yaw=U(-3.14, 3.14).
        rng, key = jax.random.split(rng)
        # dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)

        # # ----- 新增的程式碼 -----
        # # 隨機選擇出生區域：0 代表平坦地面，1 代表崎嶇地面
        # terrain_choice = jax.random.randint(key, shape=(), minval=0, maxval=2)
        # # 根據選擇，設定 X 軸的基礎出生點
        # start_x = jax.lax.cond(terrain_choice == 0,
        #                     lambda: 0.0,   # 平坦地面的中心 X 座標
        #                     lambda: 20.0)  # 崎嶇地面的中心 X 座標
        # # ----- 新增結束 -----

        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)

        # floating base
        base_qpos = self.get_floating_base_qpos(qpos)
        base_qpos = base_qpos.at[0:2].set(
            qpos[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 2]
            + dxy
        )  # x y noise

        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
        quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        new_quat = math.quat_mul(
            qpos[self._floating_base_qpos_addr + 3 : self._floating_base_qpos_addr + 7],
            quat,
        )  # yaw noise

        base_qpos = base_qpos.at[3:7].set(new_quat)

        qpos = self.set_floating_base_qpos(base_qpos, qpos)
        # print(f'DEBUG1 base qpos: {qpos}')
        # init joint position
        # qpos[7:]=*U(0.0, 0.1)
        rng, key = jax.random.split(rng)

        # multiply actual joints with noise (excluding floating base and backlash)
        qpos_j = self.get_actuator_joints_qpos(qpos) * jax.random.uniform(
            key, (self._actuators,), minval=0.5, maxval=1.5
        )
        qpos = self.set_actuator_joints_qpos(qpos_j, qpos)
        # print(f'DEBUG2 joint qpos: {qpos}')
        # init joint vel
        # d(xyzrpy)=U(-0.05, 0.05)
        rng, key = jax.random.split(rng)
        # qvel = qvel.at[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 6].set(
        #     jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
        # )

        qvel = self.set_floating_base_qvel(
            jax.random.uniform(key, (6,), minval=-0.05, maxval=0.05), qvel
        )
        # print(f'DEBUG3 base qvel: {qvel}')
        ctrl = self.get_actuator_joints_qpos(qpos)
        # print(f'DEBUG4 ctrl: {ctrl}')
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)
        rng, cmd_rng = jax.random.split(rng)
        cmd = self.sample_command(cmd_rng)

        # Sample push interval.
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        if USE_IMITATION_REWARD:
            current_reference_motion = self.PRM.get_reference_motion(
                cmd[0], cmd[1], cmd[2], 0
            )
        else:
            current_reference_motion = jp.zeros(0)

        info = {
            "rng": rng,
            "step": 0,
            "command": cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "last_last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": self._default_actuator,
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            # Push related.
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": push_interval_steps,
            # History related.
            "action_history": jp.zeros(
                self._config.noise_config.action_max_delay * self._actuators
            ),
            "imu_history": jp.zeros(self._config.noise_config.imu_max_delay * 3),
            # imitation related
            "imitation_i": 0,
            "current_reference_motion": current_reference_motion,
            "imitation_phase": jp.zeros(2),
        }

        metrics = {}
        for k, v in self._config.reward_config.scales.items():
            if v != 0:
                if v > 0:
                    metrics[f"reward/{k}"] = jp.zeros(())
                else:
                    metrics[f"cost/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        # contact = jp.array(
        #     [
        #         geoms_colliding(data, geom_id, self._floor_geom_id)
        #         for geom_id in self._feet_geom_id
        #     ]
        # )
        # obs = self._get_obs(data, info, contact)

        # 新的、修正後的接觸偵測邏輯
        contact = jp.array([
            # 檢查一隻腳 (foot_geom_id) 是否與任何一個地面 (floor_id) 發生碰撞
            jp.any(jp.array([
                geoms_colliding(data, foot_geom_id, floor_id)
                for floor_id in self._floor_geom_ids
            ]))
            for foot_geom_id in self._feet_geom_id
        ])
        obs = self._get_obs(data, info, contact)

        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        if USE_IMITATION_REWARD:
            state.info["imitation_i"] += 1
            state.info["imitation_i"] = (
                state.info["imitation_i"] % self.PRM.nb_steps_in_period
            )  # not critical, is already moduloed in get_reference_motion
            state.info["imitation_phase"] = jp.array(
                [
                    jp.cos(
                        (state.info["imitation_i"] / self.PRM.nb_steps_in_period)
                        * 2
                        * jp.pi
                    ),
                    jp.sin(
                        (state.info["imitation_i"] / self.PRM.nb_steps_in_period)
                        * 2
                        * jp.pi
                    ),
                ]
            )
        else:
            state.info["imitation_i"] = 0

        if USE_IMITATION_REWARD:
            state.info["current_reference_motion"] = self.PRM.get_reference_motion(
                state.info["command"][0],
                state.info["command"][1],
                state.info["command"][2],
                state.info["imitation_i"],
            )
        else:
            state.info["current_reference_motion"] = jp.zeros(0)

        state.info["rng"], push1_rng, push2_rng, action_delay_rng = jax.random.split(
            state.info["rng"], 4
        )

        # Handle action delay
        action_history = (
            jp.roll(state.info["action_history"], self._actuators)
            .at[: self._actuators]
            .set(action)
        )
        state.info["action_history"] = action_history
        action_idx = jax.random.randint(
            action_delay_rng,
            (1,),
            minval=self._config.noise_config.action_min_delay,
            maxval=self._config.noise_config.action_max_delay,
        )
        action_w_delay = action_history.reshape((-1, self._actuators))[
            action_idx[0]
        ]  # action with delay

        # self.action_filter.push(action_w_delay)
        # action_w_delay = self.action_filter.get_filtered_action()

        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= (
            jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
        )
        push *= self._config.push_config.enable
        qvel = state.data.qvel
        qvel = qvel.at[
            self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2
        ].set(
            push * push_magnitude
            + qvel[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2]
        )  # floating base x,y
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        ####

        motor_targets = (
            self._default_actuator + action_w_delay * self._config.action_scale
        )

        if USE_MOTOR_SPEED_LIMITS:
            prev_motor_targets = state.info["motor_targets"]

            motor_targets = jp.clip(
                motor_targets,
                prev_motor_targets
                - self._config.max_motor_velocity * self.dt,  # control dt
                prev_motor_targets
                + self._config.max_motor_velocity * self.dt,  # control dt
            )

        # motor_targets.at[5:9].set(state.info["command"][3:])  # head joints
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

        state.info["motor_targets"] = motor_targets

        # contact = jp.array(
        #     [
        #         geoms_colliding(data, geom_id, self._floor_geom_id)
        #         for geom_id in self._feet_geom_id
        #     ]
        # )
        # 新的、修正後的接觸偵測邏輯
        contact = jp.array([
            # 檢查一隻腳 (foot_geom_id) 是否與任何一個地面 (floor_id) 發生碰撞
            jp.any(jp.array([
                geoms_colliding(data, foot_geom_id, floor_id)
                for floor_id in self._floor_geom_ids
            ]))
            for foot_geom_id in self._feet_geom_id
        ])
        
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt
        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        obs = self._get_obs(data, state.info, contact)
        done = self._get_termination(data)

        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        
        # # FIXME
        # rewards = {
        #     k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        # }
        # reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # 請將其修改為:
        scaled_rewards = {}
        for k, v in rewards.items():
            # 在 jp.sum 外層加上 jp.asarray()，強制將 v 轉換為 JAX array
            scaled_rewards[k] = jp.sum(jp.asarray(v)) * self._config.reward_config.scales[k]
        reward = jp.clip(sum(scaled_rewards.values()) * self.dt, 0.0, 10000.0)


        # jax.debug.print('STEP REWARD: {}',reward)
        state.info["push"] = push
        state.info["step"] += 1
        state.info["push_step"] += 1
        state.info["last_last_last_act"] = state.info["last_last_act"]
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action  # was
        # state.info["last_act"] = motor_targets  # became
        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
        state.info["command"] = jp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        state.info["step"] = jp.where(
            done | (state.info["step"] > 500),
            0,
            state.info["step"],
        )
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact
        # for k, v in rewards.items():
        #     rew_scale = self._config.reward_config.scales[k]
        #     if rew_scale != 0:
        #         if rew_scale > 0:
        #             state.metrics[f"reward/{k}"] = v
        #         else:
        #             state.metrics[f"cost/{k}"] = -v
        # 請將其修改為:
        for k, v in rewards.items():
            rew_scale = self._config.reward_config.scales[k]
            # 先用 asarray 和 sum 確保獎勵值是一個純數值
            summed_v = jp.sum(jp.asarray(v))
            if rew_scale != 0:
                if rew_scale > 0:
                    state.metrics[f"reward/{k}"] = summed_v
                else:
                    state.metrics[f"cost/{k}"] = -summed_v

        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_gravity(data)[-1] < 0.0
        return fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> mjx_env.Observation:

        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        accelerometer = self.get_accelerometer(data)
        # accelerometer[0] += 1.3 # TODO testing
        accelerometer.at[0].set(accelerometer[0] + 1.3)

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_accelerometer = (
            accelerometer
            + (2 * jax.random.uniform(noise_rng, shape=accelerometer.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.accelerometer
        )

        gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        # Handle IMU delay
        imu_history = jp.roll(info["imu_history"], 3).at[:3].set(noisy_gravity)
        info["imu_history"] = imu_history
        imu_idx = jax.random.randint(
            noise_rng,
            (1,),
            minval=self._config.noise_config.imu_min_delay,
            maxval=self._config.noise_config.imu_max_delay,
        )
        noisy_gravity = imu_history.reshape((-1, 3))[imu_idx[0]]

        # joint_angles = data.qpos[7:]

        # Handling backlash
        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_backlash = self.get_actuator_backlash_qpos(data.qpos)

        for i in self.backlash_idx_to_add:
            joint_backlash = jp.insert(joint_backlash, i, 0)

        joint_angles = joint_angles + joint_backlash

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1.0)
            * self._config.noise_config.level
            * self._qpos_noise_scale
        )

        # joint_vel = data.qvel[6:]
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1.0)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        linvel = self.get_local_linvel(data)
        # info["rng"], noise_rng = jax.random.split(info["rng"])
        # noisy_linvel = (
        #     linvel
        #     + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        #     * self._config.noise_config.level
        #     * self._config.noise_config.scales.linvel
        # )

        state = jp.hstack(
            [
                # noisy_linvel,  # 3
                # noisy_gyro,  # 3
                # noisy_gravity,  # 3
                noisy_gyro,  # 3
                noisy_accelerometer,  # 3
                info["command"],  # 3
                noisy_joint_angles - self._default_actuator,  # 10
                noisy_joint_vel * self._config.dof_vel_scale,  # 10
                info["last_act"],  # 10
                info["last_last_act"],  # 10
                info["last_last_last_act"],  # 10
                info["motor_targets"],  # 10
                contact,  # 2
                # info["current_reference_motion"],
                # info["imitation_i"],
                info["imitation_phase"],
            ]
        )

        accelerometer = self.get_accelerometer(data)
        global_angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        root_height = data.qpos[self._floating_base_qpos_addr + 2]

        privileged_state = jp.hstack(
            [
                state,
                gyro,  # 3
                accelerometer,  # 3
                gravity,  # 3
                linvel,  # 3
                global_angvel,  # 3
                joint_angles - self._default_actuator,
                joint_vel,
                root_height,  # 1
                data.actuator_force,  # 10
                contact,  # 2
                feet_vel,  # 4*3
                info["feet_air_time"],  # 2
                info["current_reference_motion"],
                info["imitation_i"],
                info["imitation_phase"],
            ]
        )

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics  # Unused.

        # 取得所有關節
        qpos = self.get_actuator_joints_qpos(data.qpos)

        # 左右腳數值
        left_qpos = qpos[:5]  # Assuming first 5 are left side
        right_qpos = qpos[9:14] # Assuming next 5 are right side

        # 頭部roll
        head_roll_angle = qpos[8] # 取得 head_roll 的角度 (索引為 8)
        # 取得 head_roll 的指令 (在 command 陣列中的索引為 6)
        head_roll_command = info["command"][6]


        # head_roll的獎勵計算
        # head_roll_zero_reward = reward_head_roll_zero(head_roll_angle),

        # # 只有當 head_roll 的指令為 0 時(無頭部控制模式)，才計算「保持為零」的獎勵
        # # 否則，獎勵值為 0，不產生任何影響
        # head_roll_zero_reward = jp.where(
        #     head_roll_command == 0.0,
        #     reward_head_roll_zero(head_roll_angle),
        #     0.0
        # )

        qpos_actuators = self.get_actuator_joints_qpos(data.qpos)

        # 獲取頭部角度
        head_yaw_angle = qpos[7]   # 根據 actuator 順序，head_yaw 索引為 7
        head_roll_angle = qpos[8]  # 根據 actuator 順序，head_roll 索引為 8
        forward_velocity = self.get_local_linvel(data)[0] # 取得 x 軸(前進)速度

        # 獲取當前的速度指令
        lin_vel_command = info["command"][0:2] # 線速度指令 (x, y)
        ang_vel_command = info["command"][2]   # 角速度指令 (yaw)

        
        head_yaw_angle = qpos[7]
        head_roll_angle = qpos[8]
        
        # 獲取髖部 pitch 角度
        # 根據 actuator 順序: left_hip_pitch=2, right_hip_pitch=11
        left_hip_pitch_angle = qpos[2]
        right_hip_pitch_angle = qpos[11]

        # 獲取轉彎指令的絕對值
        ang_vel_command_abs = jp.abs(info["command"][2])
        # 獲取設定的最大轉彎速度
        max_ang_vel = self._config.ang_vel_yaw[1]
        
        # 根據轉彎指令的大小，動態計算模仿獎勵的折扣率
        # 轉彎指令為 0 時，折扣率為 1.0 (完全模仿)
        # 轉彎指令達到最大時，折扣率最低降至 0.1 (10%模仿)
        imitation_discount = 1.0 - (ang_vel_command_abs / max_ang_vel) * 0.9
        
        # 計算原始的模仿獎勵
        imitation_reward = reward_imitation(
            self.get_floating_base_qpos(data.qpos),
            self.get_floating_base_qvel(data.qvel),
            self.get_actuator_joints_qpos(data.qpos),
            self.get_actuator_joints_qvel(data.qvel),
            contact,
            info["current_reference_motion"],
            info["command"],
            USE_IMITATION_REWARD,
        )
        
        # 應用動態折扣，得到最終的模仿獎勵
        final_imitation_reward = imitation_reward * imitation_discount


        left_hip_pitch = qpos_actuators[2]
        right_hip_pitch = qpos_actuators[11]
        ang_vel_command = info["command"][2]

        head_commands = info["command"][3:]
        
        # 獲取頭部和頸部角度
        # 根據 actuator 順序: neck_pitch=5, head_pitch=6
        neck_pitch_angle = qpos_actuators[5]
        head_pitch_angle = qpos_actuators[6]

        
        # # 1. 計算基礎的角速度追蹤獎勵
        # tracking_ang_vel_reward = reward_tracking_ang_vel(
        #         info["command"],
        #         self.get_gyro(data),
        #         self._config.reward_config.tracking_sigma,
        #     )
        
        # # 2. 判斷指令是否為右轉 (ang_vel_command > 0)
        # is_turning_right_command = ang_vel_command > 0
        
        # # 3. 如果是右轉指令，給予一個額外的獎勵加成 (例如 1.5 代表 50% 的加成)
        # right_turn_bonus = 1.5 
        
        # # 4. 應用加成
        # final_tracking_ang_vel_reward = jp.where(
        #     is_turning_right_command,
        #     tracking_ang_vel_reward * right_turn_bonus,
        #     tracking_ang_vel_reward
        # )

        ret = {
            "tracking_lin_vel": reward_tracking_lin_vel(
                info["command"],
                self.get_local_linvel(data),
                self._config.reward_config.tracking_sigma,
            ),
            "tracking_ang_vel": reward_tracking_ang_vel(
                info["command"],
                self.get_gyro(data),
                self._config.reward_config.tracking_sigma,
            ),
            # "orientation": cost_orientation(self.get_gravity(data)),
            "torques": cost_torques(data.actuator_force),
            "action_rate": cost_action_rate(action, info["last_act"]),
            "alive": reward_alive(),
            # "imitation": reward_imitation(  # FIXME, this reward is so adhoc...
            #     self.get_floating_base_qpos(data.qpos),  # floating base qpos
            #     self.get_floating_base_qvel(data.qvel),  # floating base qvel
            #     self.get_actuator_joints_qpos(data.qpos),
            #     self.get_actuator_joints_qvel(data.qvel),
            #     contact,
            #     info["current_reference_motion"],
            #     info["command"],
            #     USE_IMITATION_REWARD,
            # ),
            "imitation": final_imitation_reward, # <-- 使用我們計算出的最終值
            "stand_still": cost_stand_still(
                # info["command"], data.qpos[7:], data.qvel[6:], self._default_pose
                info["command"],
                self.get_actuator_joints_qpos(data.qpos),
                self.get_actuator_joints_qvel(data.qvel),
                self._default_actuator,
                ignore_head=False,
            ),
            # 新增雙腳不平行成本項
            # "legs_asymmetry": cost_legs_asymmetry(
            #     self.get_actuator_joints_qpos(data.qpos),
            #     self._left_hip_pitch_idx,
            #     self._right_hip_pitch_idx,
            #     self._left_knee_idx,
            #     self._right_knee_idx,
            # ),
            # 新增雙腳平行獎勵
            # "symmetry": reward_symmetry(left_qpos, right_qpos),
            
            # 新增head_roll角度回正獎勵
            # "head_roll_zero": head_roll_zero_reward,

            # 新增走路時，頭部角度回正獎勵
            "head_orientation_walking": reward_head_orientation_walking(
                head_roll_angle=head_roll_angle,
                head_yaw_angle=head_yaw_angle,
                forward_velocity=forward_velocity,
            ),

            # 新增站立時，頭部角度回正獎勵
            # "head_straight_standing": reward_head_straight_standing(
            #     head_yaw_angle=head_yaw_angle,
            #     head_roll_angle=head_roll_angle,
            #     lin_vel_command=lin_vel_command,
            #     ang_vel_command=ang_vel_command,
            # ),
            # 新增站立時，髖部角度回正獎勵
            # "hip_pitch_symmetry_standing": reward_hip_pitch_symmetry_standing(
            #     left_hip_pitch_angle=left_hip_pitch_angle,
            #     right_hip_pitch_angle=right_hip_pitch_angle,
            #     lin_vel_command=lin_vel_command,
            #     ang_vel_command=ang_vel_command,
            # ),
            # 新增站立時，髖部角度回零獎勵
            # "hips_straight_standing": reward_hips_straight_standing(
            #     left_hip_pitch_angle=left_hip_pitch_angle,
            #     right_hip_pitch_angle=right_hip_pitch_angle,
            #     lin_vel_command=lin_vel_command,
            #     ang_vel_command=ang_vel_command,
            # ),
            # 新增站立時，理想站姿獎勵
            # "ideal_standing_hips": reward_ideal_standing_hips(
            #     qpos_actuators=qpos_actuators,
            #     lin_vel_command=lin_vel_command,
            #     ang_vel_command=ang_vel_command,
            # ),
            # 新增轉彎時，抬高腳獎勵
            # "hip_pitch_in_turn": reward_hip_pitch_in_turn(
            #     left_hip_pitch_angle=left_hip_pitch,
            #     right_hip_pitch_angle=right_hip_pitch,
            #     ang_vel_command=ang_vel_command,
            # ),
            # 非對稱轉彎步態獎勵
            "asymmetric_turning_gait": reward_asymmetric_turning_gait(
                left_hip_pitch_angle=left_hip_pitch,
                right_hip_pitch_angle=right_hip_pitch,
                ang_vel_command=ang_vel_command,
            ),
            # 獎勵頭部微微前傾
            "head_forward_tilt": reward_head_forward_tilt(
                neck_pitch_angle=neck_pitch_angle,
                head_pitch_angle=head_pitch_angle,
            ),
            # --- ▼▼▼ [核心修改] 使用更新後的獎勵函式 ▼▼▼ ---
            "head_straight_standing": head_straight_standing(
                head_yaw_angle=head_yaw_angle,
                head_roll_angle=head_roll_angle,
                lin_vel_command=lin_vel_command,
                ang_vel_command=ang_vel_command,
                head_commands=head_commands, # 傳入頭部指令
            ),
            "ideal_standing_hips": ideal_standing_hips(
                qpos_actuators=qpos_actuators,
                lin_vel_command=lin_vel_command,
                ang_vel_command=ang_vel_command,
                head_commands=head_commands, # 傳入頭部指令
            ),
            "balance_with_head_motion": reward_balance_with_head_motion(
                feet_contacts=contact,
                lin_vel_command=lin_vel_command,
                ang_vel_command=ang_vel_command,
                head_commands=head_commands,
            ),
        }

        return ret

    def sample_command(self, rng: jax.Array) -> jax.Array:
        # --- ▼▼▼ [核心修改] 創造三種訓練情境 ▼▼▼ ---
        key_case, key_move, key_head_move = jax.random.split(rng, 3)
        
        # --- 情境 A: 正常移動指令 (全身隨機運動) ---
        rng_vel_x, rng_vel_y, rng_ang_vel, rng_neck_p, rng_head_p, rng_head_y, rng_head_r = jax.random.split(key_move, 7)
        lin_vel_x = jax.random.uniform(rng_vel_x, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1])
        lin_vel_y = jax.random.uniform(rng_vel_y, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1])
        ang_vel_yaw = jax.random.uniform(rng_ang_vel, minval=self._config.ang_vel_yaw[0], maxval=self._config.ang_vel_yaw[1])
        neck_pitch = jax.random.uniform(rng_neck_p, minval=self._config.neck_pitch_range[0], maxval=self._config.neck_pitch_range[1])
        head_pitch = jax.random.uniform(rng_head_p, minval=self._config.head_pitch_range[0], maxval=self._config.head_pitch_range[1])
        head_yaw = jax.random.uniform(rng_head_y, minval=self._config.head_yaw_range[0], maxval=self._config.head_yaw_range[1])
        head_roll = jax.random.uniform(rng_head_r, minval=self._config.head_roll_range[0], maxval=self._config.head_roll_range[1])
        move_command = jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw, neck_pitch, head_pitch, head_yaw, head_roll])

        # --- 情境 B: 完全靜止指令 ---
        stand_command = jp.zeros(7)
        
        # --- 情境 C: 站立且僅頭部運動指令 ---
        rng_neck_p_new, rng_head_p_new, rng_head_y_new, rng_head_r_new = jax.random.split(key_head_move, 4)
        neck_pitch_new = jax.random.uniform(rng_neck_p_new, minval=self._config.neck_pitch_range[0], maxval=self._config.neck_pitch_range[1])
        head_pitch_new = jax.random.uniform(rng_head_p_new, minval=self._config.head_pitch_range[0], maxval=self._config.head_pitch_range[1])
        head_yaw_new = jax.random.uniform(rng_head_y_new, minval=self._config.head_yaw_range[0], maxval=self._config.head_yaw_range[1])
        head_roll_new = jax.random.uniform(rng_head_r_new, minval=self._config.head_roll_range[0], maxval=self._config.head_roll_range[1])
        stand_and_move_head_command = jp.hstack([0.0, 0.0, 0.0, neck_pitch_new, head_pitch_new, head_yaw_new, head_roll_new])

        # --- 使用隨機數在三種情境中選擇一種 ---
        # 60% 機率移動，20% 機率完全靜止，20% 機率站立動頭
        command = jax.lax.switch(
            jax.random.choice(key_case, 3, p=jp.array([0.6, 0.2, 0.2])),
            [lambda: move_command, lambda: stand_command, lambda: stand_and_move_head_command],
        )
        return command

    # def sample_command(self, rng: jax.Array) -> jax.Array:
    #     rng1, rng2, rng3, rng4, rng5, rng6, rng7, rng8 = jax.random.split(rng, 8) # 新增rng9給 無頭部控制模式
    #     # rng1, rng2, rng3, rng4, rng5, rng6, rng7, rng8, rng9 = jax.random.split(rng, 9) # 新增rng9給 無頭部控制模式

    #     lin_vel_x = jax.random.uniform(
    #         rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
    #     )
    #     lin_vel_y = jax.random.uniform(
    #         rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
    #     )
    #     ang_vel_yaw = jax.random.uniform(
    #         rng3,
    #         minval=self._config.ang_vel_yaw[0],
    #         maxval=self._config.ang_vel_yaw[1],
    #     )

    #     # --- ▼▼▼ 關鍵修改：融合隨機指令與固定目標 ▼▼▼ ---
        
    #     # 1. 像往常一樣，產生隨機的頭部姿態指令
    #     random_neck_pitch = jax.random.uniform(
    #         rng5,
    #         minval=self._config.neck_pitch_range[0] * self._config.head_range_factor,
    #         maxval=self._config.neck_pitch_range[1] * self._config.head_range_factor,
    #     )
    #     random_head_pitch = jax.random.uniform(
    #         rng6,
    #         minval=self._config.head_pitch_range[0] * self._config.head_range_factor,
    #         maxval=self._config.head_pitch_range[1] * self._config.head_range_factor,
    #     )

    #     # 2. 定義我們期望的固定前傾姿態
    #     # 我們將目標前傾角度 (0.2 弧度) 分配給 neck_pitch
    #     target_neck_pitch = 0.2
    #     target_head_pitch = 0.0

    #     # 3. 進行加權平均，產生一個融合後的指令
    #     # blending_weight 越接近 1.0，我們的固定目標影響就越大
    #     blending_weight = 0.5 
    #     neck_pitch = (
    #         random_neck_pitch * (1 - blending_weight)
    #         + target_neck_pitch * blending_weight
    #     )
    #     head_pitch = (
    #         random_head_pitch * (1 - blending_weight)
    #         + target_head_pitch * blending_weight
    #     )
    #     # --- ▲▲▲ 修改結束 ▲▲▲ ---

    #     # head_yaw 和 head_roll 保持不變
    #     head_yaw = jax.random.uniform(
    #         rng7,
    #         minval=self._config.head_yaw_range[0] * self._config.head_range_factor,
    #         maxval=self._config.head_yaw_range[1] * self._config.head_range_factor,
    #     )

    #     head_roll = jax.random.uniform(
    #         rng8,
    #         minval=self._config.head_roll_range[0] * self._config.head_range_factor,
    #         maxval=self._config.head_roll_range[1] * self._config.head_range_factor,
    #     )

    #     # 嘗試強制為0
    #     # head_roll = 0.0
    #     # head_yaw = 0.0

    #     # # 有 50% 的機率進入「無頭部控制模式」
    #     # no_head_control_mode = jax.random.bernoulli(rng9, p=0.5)

    #     # # 如果進入該模式，就將所有頭部指令覆寫為 0
    #     # # neck_pitch = jp.where(no_head_control_mode, 0.0, neck_pitch)
    #     # # head_pitch = jp.where(no_head_control_mode, 0.0, head_pitch)
    #     # head_yaw = jp.where(no_head_control_mode, 0.0, head_yaw)
    #     # head_roll = jp.where(no_head_control_mode, 0.0, head_roll)

    #     # With 10% chance, set everything to zero.
    #     return jp.where(
    #         jax.random.bernoulli(rng4, p=0.3),
    #         jp.zeros(7),
    #         jp.hstack(
    #             [
    #                 lin_vel_x,
    #                 lin_vel_y,
    #                 ang_vel_yaw,
    #                 neck_pitch,
    #                 head_pitch,
    #                 head_yaw,
    #                 head_roll,
    #             ]
    #         ),
    #     )
