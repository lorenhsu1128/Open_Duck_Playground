import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.common.utils import LowPassActionFilter

from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase

USE_MOTOR_SPEED_LIMITS = True


class MjInfer(MJInferBase):
    def __init__(
        self, model_path: str, reference_data: str, onnx_model_path: str, standing: bool, home_pos: list
    ):
        super().__init__(model_path, home_pos=home_pos)
        self.home_pos = home_pos

        self.standing = standing
        self.head_control_mode = self.standing
        
        # --- [新增] 手動姿態調整模式的開關 ---
        self.manual_pose_mode = False
        # ------------------------------------

        # Params
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 0.25

        self.action_filter = LowPassActionFilter(50, cutoff_frequency=37.5)

        if not self.standing:
            self.PRM = PolyReferenceMotion(reference_data)

        self.policy = OnnxInfer(onnx_model_path, awd=True)

        self.COMMANDS_RANGE_X = [-0.15, 0.15]
        self.COMMANDS_RANGE_Y = [-0.2, 0.2]
        self.COMMANDS_RANGE_THETA = [-1.0, 1.0]  # [-1.0, 1.0]

        self.NECK_PITCH_RANGE = [-0.34, 1.1]
        self.HEAD_PITCH_RANGE = [-0.78, 0.78]
        self.HEAD_YAW_RANGE = [-1.5, 1.5]
        self.HEAD_ROLL_RANGE = [-0.5, 0.5]

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.saved_obs = []

        self.max_motor_velocity = 5.24  # rad/s

        self.phase_frequency_factor = 1.0

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")
        print(f"backlash joint names: {self.backlash_joint_names}")
        # print(f"actual joints idx: {self.get_actual_joints_idx()}")

    def get_obs(
        self,
        data,
        command,  # , qvel_history, qpos_error_history, gravity_history
    ):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        accelerometer[0] += 1.3

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)

        contacts = self.get_feet_contacts(data)

        obs = np.concatenate(
            [
                gyro,
                accelerometer,
                command,
                joint_angles - self.default_actuator,
                joint_vel * self.dof_vel_scale,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                contacts,
                self.imitation_phase,
            ]
        )

        return obs

    def key_callback(self, keycode):
        print(f"key: {keycode}")
        
        # --- [新增] M 鍵用來切換control手動模式 ---
        if keycode == 77:  # M key
            self.manual_pose_mode = not self.manual_pose_mode
            if self.manual_pose_mode:
                print("\n" + "="*50)
                print("      MANUAL POSING MODE ACTIVATED")
                print("      AI control is PAUSED. You can now adjust joints.")
                print("="*50 + "\n")
            else:
                print("\n" + "="*50)
                print("      MANUAL POSING MODE DEACTIVATED")
                print("      AI control is RESUMED.")
                print("="*50 + "\n")
            return
        # ------------------------------------

        if keycode == 72:  # h 來切換頭部模式
            self.head_control_mode = not self.head_control_mode
            
        lin_vel_x = 0
        lin_vel_y = 0
        ang_vel = 0
        if not self.head_control_mode:
            if keycode == 265:  # arrow up 前進
                lin_vel_x = self.COMMANDS_RANGE_X[1]
            if keycode == 264:  # arrow down 後退
                lin_vel_x = self.COMMANDS_RANGE_X[0]
            if keycode == 263:  # arrow left 左平移
                lin_vel_y = self.COMMANDS_RANGE_Y[1]
            if keycode == 262:  # arrow right 右平移
                lin_vel_y = self.COMMANDS_RANGE_Y[0]
            if keycode == 81:  # q key 左彎
                ang_vel = self.COMMANDS_RANGE_THETA[1]
            if keycode == 69:  # e key 右彎
                ang_vel = self.COMMANDS_RANGE_THETA[0]
            if keycode == 80:  # p 增加跨步頻率
                self.phase_frequency_factor += 0.1
            if keycode == 59:  # m 減低跨步頻率
                self.phase_frequency_factor -= 0.1
        else:
            neck_pitch = 0
            head_pitch = 0
            head_yaw = 0
            head_roll = 0
            if keycode == 265:  # arrow up 抬頭
                head_pitch = self.NECK_PITCH_RANGE[1]
            if keycode == 264:  # arrow down 低頭
                head_pitch = self.NECK_PITCH_RANGE[0]
            if keycode == 263:  # arrow left 左轉頭
                head_yaw = self.HEAD_YAW_RANGE[1]
            if keycode == 262:  # arrow right 右轉頭
                head_yaw = self.HEAD_YAW_RANGE[0]
            if keycode == 81:  # q key 左晃
                head_roll = self.HEAD_ROLL_RANGE[1]
            if keycode == 69:  # e key 右晃
                head_roll = self.HEAD_ROLL_RANGE[0]

            # 按下s重置頭部角度
            if keycode == 83:  # S key
                print("Head pose reset.")
                neck_pitch = 0.0
                head_pitch = 0.0
                head_yaw = 0.0
                head_roll = 0.0

            self.commands[3] = neck_pitch
            self.commands[4] = head_pitch
            self.commands[5] = head_yaw
            self.commands[6] = head_roll

        self.commands[0] = lin_vel_x
        self.commands[1] = lin_vel_y
        self.commands[2] = ang_vel

    def run(self):
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=True,  # <-- 確保左側 UI 開啟
                show_right_ui=True, # <-- 確保右側 UI 開啟
                key_callback=self.key_callback,
            ) as viewer:
                
                counter = 0
                while viewer.is_running():

                    viewer.cam.lookat[0:2] = self.home_pos
                    
                    step_start = time.time()

                    mujoco.mj_step(self.model, self.data)

                    # --- [修改] 只有在非手動模式下，才執行 AI 控制 ---
                    if not self.manual_pose_mode:
                        counter += 1
                        if counter % self.decimation == 0:
                            if not self.standing:
                                self.imitation_i += 1.0 * self.phase_frequency_factor
                                self.imitation_i = (
                                    self.imitation_i % self.PRM.nb_steps_in_period
                                )
                                self.imitation_phase = np.array(
                                    [
                                        np.cos(
                                            self.imitation_i
                                            / self.PRM.nb_steps_in_period
                                            * 2
                                            * np.pi
                                        ),
                                        np.sin(
                                            self.imitation_i
                                            / self.PRM.nb_steps_in_period
                                            * 2
                                            * np.pi
                                        ),
                                    ]
                                )

                            # 獲取當前的轉彎指令和實際的轉彎速度
                            ang_vel_command = self.commands[2]
                            actual_ang_vel = self.get_gyro(self.data)[2] # Z 軸的角速度

                            # 獲取轉彎獎勵是否被觸發的狀態
                            # (我們可以直接複製 rewards.py 中的邏輯來判斷)
                            turn_threshold = 0.1 # <-- 在這裡設定您想測試的值
                            is_turning = abs(ang_vel_command) > turn_threshold
                            
                            # 每隔一段時間印出一次數據，避免刷屏
                            if counter % (self.decimation * 10) == 0:
                                print(
                                    f"轉彎指令: {ang_vel_command:+.2f} | "
                                    f"實際角速度: {actual_ang_vel:+.2f} | "
                                    f"轉彎獎勵觸發: {'是' if is_turning else '否'}"
                                )

                            obs = self.get_obs(
                                self.data,
                                self.commands,
                            )
                            self.saved_obs.append(obs)
                            action = self.policy.infer(obs)

                            # 當進入 H 模式時，頭部改為純手控
                            if self.head_control_mode:
                                # 當頭部手動控制模式開啟時，我們用 commands 的值
                                # 來強制覆寫 AI 輸出的 action 中對應頭部的部分。
                                
                                # 根據您檔案中 actuator 的順序，頭部關節在 action 陣列中的索引是:
                                # neck_pitch: 5
                                # head_pitch: 6
                                # head_yaw:   7
                                # head_roll:  8
                                
                                # 根據您檔案中 key_callback 的邏輯，手動指令在 commands 陣列中的索引是:
                                # (unused) neck_pitch: 3 
                                # head_pitch: 4
                                # head_yaw:   5
                                # head_roll:  6
                                

                                # 執行覆寫
                                action[5] = self.commands[4]  # neck_pitch 使用 head_pitch 的指令
                                action[6] = self.commands[4]  # head_pitch
                                action[7] = self.commands[5]  # head_yaw
                                action[8] = self.commands[6]  # head_roll

                            self.last_last_last_action = self.last_last_action.copy()
                            self.last_last_action = self.last_action.copy()
                            self.last_action = action.copy()

                            self.motor_targets = (
                                self.default_actuator + action * self.action_scale
                            )

                            if USE_MOTOR_SPEED_LIMITS:
                                self.motor_targets = np.clip(
                                    self.motor_targets,
                                    self.prev_motor_targets
                                    - self.max_motor_velocity
                                    * (self.sim_dt * self.decimation),
                                    self.prev_motor_targets
                                    + self.max_motor_velocity
                                    * (self.sim_dt * self.decimation),
                                )

                                self.prev_motor_targets = self.motor_targets.copy()

                            self.data.ctrl = self.motor_targets.copy()
                    # -----------------------------------------------

                    viewer.sync()

                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    # parser.add_argument("-k", action="store_true", default=False)
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml",
    )
    # ----- 在下方新增這個參數 -----
    parser.add_argument(
        "--home_pos",
        type=float,
        nargs=2,
        default=[0.0, 0.0], #平地的中心點是 (0.0, 0.0) 崎嶇地的中心點是 (20.0, 0.0) 交界處是 (10.0, 0.0)
        help="Specify the initial x, y position of the robot. Default: 0.0 0.0",
    )
    # -----------------------------
    parser.add_argument("--standing", action="store_true", default=False)

    args = parser.parse_args()

    mjinfer = MjInfer(
        args.model_path,
        args.reference_data,
        args.onnx_model_path,
        args.standing,
        home_pos=args.home_pos,
    )
    mjinfer.run()