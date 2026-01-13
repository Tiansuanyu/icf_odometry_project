import mujoco
import mujoco.viewer
import numpy as np
import time
import pygame  # 引入pygame处理手柄

# --- 配置 ---
XML_PATH = "scene.xml" 

import mujoco
import mujoco.viewer
import numpy as np
import time
import pygame

NORMAL_SPEED = 80.0    # 正常推杆到底的速度 (原50)
TURBO_SPEED  = 200.0   # 按住加速键后的速度
ROTATION_SPEED = 5.0   # 旋转灵敏度 (原3.0)

RAIL_MIN = 0.0
RAIL_MAX = 0.25        

OFFSETS = {
    'front_left':   -np.pi/4, 'front_right':  +np.pi/4,
    'rear_left':    -3*np.pi/4, 'rear_right':   +3*np.pi/4
}

WHEEL_MAP_CONFIG = {
    'front_left':   'RR', 'front_right':  'LR',  
    'rear_left':    'RF', 'rear_right':   'LF',  
}

WHEEL_GEOMETRY = {
    'front_left':   (-1.0,  1.0), 'front_right':  ( 1.0,  1.0), 
    'rear_left':    (-1.0, -1.0), 'rear_right':   ( 1.0, -1.0), 
}

class ChassisController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.init_joystick()
        
        # --- 状态变量 ---
        self.vx, self.vy, self.w = 0.0, 0.0, 0.0
        self.rail_height = 0.0
        self.current_max_speed = NORMAL_SPEED  # 当前生效的最大速度

        self.actuators = {}
        self.wheels = {}
        
        # 紧凑化初始化代码 (逻辑未变)
        for name in ['LF', 'RF', 'LR', 'RR']:
            s_n, d_n, r_n = f"{name}_steer", f"{name}_drive", f"{name}_rail"
            s_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, s_n)
            d_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, d_n)
            r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, r_n)
            
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, s_n)
            if j_id == -1: j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_steer_joint")
            q_adr = model.jnt_qposadr[j_id] if j_id != -1 else None

            self.actuators[f"{name}_data"] = {'s': s_id, 'd': d_id, 'r': r_id, 'q': q_adr}

        for logic, prefix in WHEEL_MAP_CONFIG.items():
            d = self.actuators[f"{prefix}_data"]
            self.wheels[logic] = {'steer': d['s'], 'drive': d['d'], 'rail': d['r'], 'q': d['q'], 'pos': WHEEL_GEOMETRY[logic]}

    def init_joystick(self):
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"手柄已连接: {self.joystick.get_name()}")

    def process_input(self):
        pygame.event.pump()
        self.vx, self.vy, self.w = 0.0, 0.0, 0.0
        
        if self.joystick:
            # 1. 读取模拟量 (-1.0 到 1.0) -> 这里就是推杆力度映射
            val_lx = self.joystick.get_axis(0) 
            val_ly = self.joystick.get_axis(1) 
            val_rx = self.joystick.get_axis(3) # 不同手柄可能是轴2或轴3

            # 死区
            if abs(val_lx) < 0.1: val_lx = 0
            if abs(val_ly) < 0.1: val_ly = 0
            if abs(val_rx) < 0.1: val_rx = 0
            
            self.vx = val_lx
            self.vy = -val_ly
            self.w  = -val_rx

            # 2. 加速逻辑 (Turbo Mode)
            # 假设 Button 5 是 RB/R1 (Xbox/Logitech)
            # Switch Pro 可能是别的，你可以乱按试试
            if self.joystick.get_button(5): 
                self.current_max_speed = TURBO_SPEED
            else:
                self.current_max_speed = NORMAL_SPEED

            if self.joystick.get_button(0): self.rail_height = RAIL_MIN
            if self.joystick.get_button(1): self.rail_height = RAIL_MAX
            
        self.rail_height = np.clip(self.rail_height, RAIL_MIN, RAIL_MAX)

    def optimize_module(self, current_angle, target_angle, target_speed):
        error = target_angle - current_angle
        error = np.arctan2(np.sin(error), np.cos(error))
        
        if abs(error) > (np.pi / 2):
            target_angle += np.pi
            target_speed = -target_speed
            error = target_angle - current_angle
            error = np.arctan2(np.sin(error), np.cos(error))

        scale_factor = np.cos(error)
        if scale_factor < 0.1: scale_factor = 0.0
        
        return np.arctan2(np.sin(target_angle), np.cos(target_angle)), target_speed * scale_factor

    def update(self):
        self.process_input()

        for name, wheel in self.wheels.items():
            if wheel['rail'] != -1: self.data.ctrl[wheel['rail']] = self.rail_height

            rx, ry = wheel['pos'] 
            wheel_vx = self.vx - (self.w * ROTATION_SPEED) * ry
            wheel_vy = self.vy + (self.w * ROTATION_SPEED) * rx
            
            # 这里算出来的是 0.0 ~ 1.414 的比例值
            raw_target_speed = np.sqrt(wheel_vx**2 + wheel_vy**2)
            
            if raw_target_speed < 0.05:
                self.data.ctrl[wheel['drive']] = 0.0
                continue

            raw_target_angle = np.arctan2(wheel_vy, wheel_vx) + OFFSETS[name]

            current_angle = 0.0
            if wheel['q'] is not None:
                raw_q = self.data.qpos[wheel['q']]
                current_angle = np.arctan2(np.sin(raw_q), np.cos(raw_q))
            
            opt_angle, opt_speed_factor = self.optimize_module(current_angle, raw_target_angle, raw_target_speed)

            self.data.ctrl[wheel['steer']] = opt_angle
            
            # --- 关键修改：用 factor 乘以 当前最大速度 ---
            # 最终输出 = (推杆比例 * 优化系数) * (80 或 200)
            self.data.ctrl[wheel['drive']] = opt_speed_factor * self.current_max_speed

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    controller = ChassisController(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)
        while viewer.is_running():
            controller.update()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()