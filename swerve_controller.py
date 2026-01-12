import mujoco
import mujoco.viewer
import numpy as np
import time

# --- 配置 ---
XML_PATH = "scene.xml" 

# 速度与控制参数
MAX_SPEED = 50.0       
ROTATION_SPEED = 3.0   
RAIL_MIN = 0.0
RAIL_MAX = 0.25        
RAIL_STEP = 0.02       # 每次按键调整高度的步长

# Offset
OFFSETS = {
    'front_left':   -np.pi/4, 
    'front_right':  +np.pi/4,
    'rear_left':    -3*np.pi/4,
    'rear_right':   +3*np.pi/4
}

# 映射关系
WHEEL_MAP_CONFIG = {
    'front_left':   'RR', 
    'front_right':  'LR',  
    'rear_left':    'RF',  
    'rear_right':   'LF',  
}

# 几何坐标 (X=右, Y=前)
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
        # 查找所有电机
        for name in ['LF', 'RF', 'LR', 'RR']:
            self.actuators[f"{name}_steer"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_steer")
            self.actuators[f"{name}_drive"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_drive")
            
            # 悬挂电机检查
            rail_name = f"{name}_rail"
            rail_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, rail_name)
            if rail_id != -1:
                self.actuators[rail_name] = rail_id
            else:
                print(f"[严重警告] XML中没找到 {rail_name}！按升降键将无效。请检查XML actuator部分。")

        self.wheels = {}
        for logic_name, xml_prefix in WHEEL_MAP_CONFIG.items():
            wheel_data = {
                'steer_id': self.actuators[f"{xml_prefix}_steer"],
                'drive_id': self.actuators[f"{xml_prefix}_drive"],
                'pos': WHEEL_GEOMETRY[logic_name]
            }
            if f"{xml_prefix}_rail" in self.actuators:
                wheel_data['rail_id'] = self.actuators[f"{xml_prefix}_rail"]
            
            self.wheels[logic_name] = wheel_data

        self.vx = 0.0 
        self.vy = 0.0 
        self.w  = 0.0
        self.rail_height = 0.0

    def key_callback(self, keycode):
        # --- 按键侦测器 ---
        # 如果你按键没反应，看控制台打印的数字，把那个数字填到下面的 if 里
        print(f"Debug: Key Pressed Code = {keycode}") 

        self.vx = 0.0
        self.vy = 0.0
        self.w = 0.0
        
        # 移动 (上下左右)
        if keycode == 265:   # Up
            self.vy = 1.0
        elif keycode == 264: # Down
            self.vy = -1.0
        elif keycode == 263: # Left
            self.vx = -1.0
        elif keycode == 262: # Right
            self.vx = 1.0
        
        # 旋转
        elif keycode == 81:  # Q
            self.w = -1.0  
        elif keycode == 69:  # E
            self.w = 1.0   

        # --- 悬挂控制 (使用中括号) ---
        # [ 键 (通常是 91) -> 降低
        # ] 键 (通常是 93) -> 升高
        elif keycode == 93: # ] 键
            self.rail_height = RAIL_MAX # 直接设为 0.25
            
        elif keycode == 91: # [ 键
            self.rail_height = RAIL_MIN # 直接设为 0.0
        
        # 限制范围
        self.rail_height = np.clip(self.rail_height, RAIL_MIN, RAIL_MAX)

        # 打印当前状态
        # print(f"Status -> Height: {self.rail_height:.2f}")

    def update(self):
        for name, wheel in self.wheels.items():
            # 1. 悬挂控制
            if 'rail_id' in wheel:
                self.data.ctrl[wheel['rail_id']] = self.rail_height

            # 2. 运动学
            rx, ry = wheel['pos'] 
            wheel_vx = self.vx - (self.w * ROTATION_SPEED) * ry
            wheel_vy = self.vy + (self.w * ROTATION_SPEED) * rx
            
            target_speed = np.sqrt(wheel_vx**2 + wheel_vy**2)
            
            if target_speed < 0.1:
                self.data.ctrl[wheel['drive_id']] = 0.0
                continue

            target_angle = np.arctan2(wheel_vy, wheel_vx)
            final_angle = target_angle + OFFSETS[name]
            final_angle = (final_angle + np.pi) % (2 * np.pi) - np.pi
            
            self.data.ctrl[wheel['steer_id']] = final_angle
            self.data.ctrl[wheel['drive_id']] = target_speed * MAX_SPEED

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    controller = ChassisController(model, data)

    with mujoco.viewer.launch_passive(model, data, key_callback=controller.key_callback) as viewer:
        mujoco.mj_resetData(model, data)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE 

        while viewer.is_running():
            step_start = time.time()
            controller.update()
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()