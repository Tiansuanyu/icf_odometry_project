import mujoco
import mujoco.viewer
import numpy as np
import time

# --- 配置 ---
XML_PATH = "scene.xml" 

MAX_SPEED = 50.0       
ROTATION_SPEED = 3.0   
RAIL_MIN = 0.00
RAIL_MAX = 0.25        

# Offset (安装角度偏移)
OFFSETS = {
    'front_left':   -np.pi/4, 
    'front_right':  +np.pi/4,
    'rear_left':    -3*np.pi/4,
    'rear_right':   +3*np.pi/4
}

WHEEL_MAP_CONFIG = {
    'front_left':   'RR', 
    'front_right':  'LR',  
    'rear_left':    'RF',  
    'rear_right':   'LF',  
}

WHEEL_GEOMETRY = {
    'front_left':   (-1.0,  1.0), 
    'front_right':  ( 1.0,  1.0), 
    'rear_left':    (-1.0, -1.0), 
    'rear_right':   ( 1.0, -1.0), 
}

class ChassisController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        self.actuators = {}
        self.wheels = {}
        
        # 初始化查找 ID 和 Address
        for name in ['LF', 'RF', 'LR', 'RR']:
            # 这些是 XML 里的 Actuator 名字
            steer_act_name = f"{name}_steer"
            drive_act_name = f"{name}_drive"
            rail_act_name  = f"{name}_rail"
            
            # 1. 查找 Actuator ID (用于写控制量)
            steer_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, steer_act_name)
            drive_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, drive_act_name)
            rail_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, rail_act_name)

            # 2. 查找 Joint Address (用于读真实角度)
            target_joint_name = f"{name}_yaw_joint"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, target_joint_name)
            
            if joint_id != -1:
                qpos_adr = model.jnt_qposadr[joint_id]
            else:
                qpos_adr = None
            
            if qpos_adr is None:
                # 打印出它试图寻找的名字，方便调试
                print(f"[严重警告] XML中找不到关节: '{target_joint_name}'。无法读取角度，优化逻辑将失效！")

            self.actuators[f"{name}_data"] = {
                'steer_id': steer_id,
                'drive_id': drive_id,
                'rail_id': rail_id,
                'qpos_adr': qpos_adr
            }

        # 组装逻辑轮子 (保持不变)
        for logic_name, xml_prefix in WHEEL_MAP_CONFIG.items():
            comp_data = self.actuators[f"{xml_prefix}_data"]
            self.wheels[logic_name] = {
                'steer_id': comp_data['steer_id'],
                'drive_id': comp_data['drive_id'],
                'rail_id':  comp_data['rail_id'],
                'qpos_adr': comp_data['qpos_adr'],
                'pos': WHEEL_GEOMETRY[logic_name]
            }

        self.vx = 0.0 
        self.vy = 0.0 
        self.w  = 0.0
        self.rail_height = 0.0

    def key_callback(self, keycode):
        # --- 核心修复：每次按键前，先强制清除上一次的速度状态 ---
        self.vx = 0.0
        self.vy = 0.0
        self.w = 0.0
        
        # --- 移动控制 ---
        if keycode == 265:   # Up (箭头向上)
            self.vy = 1.0
        elif keycode == 264: # Down (箭头向下)
            self.vy = -1.0
        elif keycode == 263: # Left (箭头向左)
            self.vx = -1.0
        elif keycode == 262: # Right (箭头向右)
            self.vx = 1.0
        
        # --- 旋转控制 ---
        elif keycode == 81:  # Q (左旋)
            self.w = -1.0  
        elif keycode == 69:  # E (右旋)
            self.w = 1.0   

        # --- 悬挂/其他功能 ---
        elif keycode == 32: 
            pass 
            
        elif keycode == 93: # ]
            self.rail_height = RAIL_MAX 
        elif keycode == 91: # [
            self.rail_height = RAIL_MIN 
        
        self.rail_height = np.clip(self.rail_height, RAIL_MIN, RAIL_MAX)

    def optimize_module(self, current_angle, target_angle, target_speed):
        """
        学术/工程标准 Swerve 优化函数
        输入: 当前弧度, 目标弧度, 目标速度
        输出: 优化后的目标弧度, 优化后的速度
        """
        
        # 1. 计算误差，并归一化到 [-pi, pi]
        error = target_angle - current_angle
        error = np.arctan2(np.sin(error), np.cos(error)) # 最稳健的归一化写法

        # 2. 策略A：最短路径 (Vector Inversion)
        # 如果误差绝对值 > 90度，说明反着开更近
        if abs(error) > (np.pi / 2):
            target_angle = target_angle + np.pi
            target_speed = -target_speed
            # 重新计算误差供策略B使用
            error = target_angle - current_angle
            error = np.arctan2(np.sin(error), np.cos(error))

        # 3. 策略B：余弦缩放 (Cosine Scaling)
        # 误差越大，速度越慢；误差90度时速度为0。
        scale_factor = np.cos(error)
        
        # 如果你希望容忍度高一点，不想让速度降得太厉害，可以用 power 调整，例如：
        # scale_factor = np.sign(np.cos(error)) * (abs(np.cos(error)) ** 0.5)
        
        # 只有当 scale_factor 非常小时才强制为0，防止电流噪声
        if scale_factor < 0.1: 
            scale_factor = 0.0

        final_speed = target_speed * scale_factor
        
        # 再次归一化目标角度输出，防止数值溢出
        final_angle = np.arctan2(np.sin(target_angle), np.cos(target_angle))
        
        return final_angle, final_speed

    def update(self):
        for name, wheel in self.wheels.items():
            # 悬挂
            if wheel['rail_id'] != -1:
                self.data.ctrl[wheel['rail_id']] = self.rail_height

            # 运动学解算
            rx, ry = wheel['pos'] 
            wheel_vx = self.vx - (self.w * ROTATION_SPEED) * ry
            wheel_vy = self.vy + (self.w * ROTATION_SPEED) * rx
            
            raw_target_speed = np.sqrt(wheel_vx**2 + wheel_vy**2)
            
            # 死区处理：如果摇杆归中，不再旋转轮子，只停止驱动
            if raw_target_speed < 0.01:
                self.data.ctrl[wheel['drive_id']] = 0.0
                continue

            raw_target_angle = np.arctan2(wheel_vy, wheel_vx) + OFFSETS[name]

            # --- 获取真实角度 ---
            current_angle = 0.0
            if wheel['qpos_adr'] is not None:
                # 核心修正：使用 arctan2(sin, cos) 彻底解决多圈累积问题
                raw_q = self.data.qpos[wheel['qpos_adr']]
                current_angle = np.arctan2(np.sin(raw_q), np.cos(raw_q))
            
            # --- 执行优化逻辑 ---
            opt_angle, opt_speed = self.optimize_module(current_angle, raw_target_angle, raw_target_speed)

            # 输出控制
            self.data.ctrl[wheel['steer_id']] = opt_angle
            self.data.ctrl[wheel['drive_id']] = opt_speed * MAX_SPEED

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    controller = ChassisController(model, data)

    with mujoco.viewer.launch_passive(model, data, key_callback=controller.key_callback) as viewer:
        mujoco.mj_resetData(model, data)
        while viewer.is_running():
            controller.update()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()