import mujoco
import mujoco.viewer
import numpy as np
import time
import pygame
import matplotlib.pyplot as plt

# ================= 配置区域 =================
XML_PATH = "scene.xml"
WHEEL_RADIUS = 0.03

# 速度配置
NORMAL_SPEED = 80.0    
TURBO_SPEED  = 200.0   
ROTATION_SPEED = 5.0   
RAIL_MIN = 0.00
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

# ================= 滤波器类 (DLPF) =================
class ImuProcessor:
    def __init__(self, sample_rate=500.0, cutoff_freq=10.0):
        dt = 1.0 / sample_rate
        rc = 1.0 / (2 * np.pi * cutoff_freq)
        self.alpha = dt / (dt + rc)
        
        self.last_acc = None
        self.last_gyro = None

        # 模拟零偏 (Bias)
        self.acc_bias = np.array([0.02, -0.02, 0.05]) 
        self.gyro_bias = np.array([0.001, 0.001, -0.001])

    def process(self, raw_acc, raw_gyro):
        # 1. 加 Bias
        curr_acc = raw_acc + self.acc_bias
        curr_gyro = raw_gyro + self.gyro_bias
        
        # 2. 初始化
        if self.last_acc is None:
            self.last_acc = curr_acc
            self.last_gyro = curr_gyro
            return curr_acc, curr_gyro
        
        # 3. 滤波
        filt_acc = self.alpha * curr_acc + (1.0 - self.alpha) * self.last_acc
        filt_gyro = self.alpha * curr_gyro + (1.0 - self.alpha) * self.last_gyro
        
        # 更新
        self.last_acc = filt_acc
        self.last_gyro = filt_gyro
        
        return filt_acc, filt_gyro

# ================= 控制器类 (保持不变) =================
class ChassisController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.init_joystick()
        
        self.vx, self.vy, self.w = 0.0, 0.0, 0.0
        self.rail_height = 0.0
        self.current_max_speed = NORMAL_SPEED 

        self.actuators = {}
        self.wheels = {}
        
        for name in ['LF', 'RF', 'LR', 'RR']:
            s_n, d_n, r_n = f"{name}_steer", f"{name}_drive", f"{name}_rail"
            s_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, s_n)
            d_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, d_n)
            r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, r_n)
            
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_yaw_joint")
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
            print(f"✅ 手柄已连接: {self.joystick.get_name()}")
        else:
            print("❌ 未检测到手柄，请检查连接")

    def process_input(self):
        pygame.event.pump()
        self.vx, self.vy, self.w = 0.0, 0.0, 0.0
        
        if self.joystick:
            val_lx = self.joystick.get_axis(0) 
            val_ly = self.joystick.get_axis(1) 
            val_rx = self.joystick.get_axis(3) 

            if abs(val_lx) < 0.1: val_lx = 0
            if abs(val_ly) < 0.1: val_ly = 0
            if abs(val_rx) < 0.1: val_rx = 0
            
            self.vx = val_lx
            self.vy = -val_ly
            self.w  = -val_rx

            if self.joystick.get_button(5): self.current_max_speed = TURBO_SPEED
            else: self.current_max_speed = NORMAL_SPEED

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
            self.data.ctrl[wheel['drive']] = opt_speed_factor * self.current_max_speed

# ================= 主程序 =================
def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    controller = ChassisController(model, data)
    
    # 滤波器 (10Hz)
    imu_filter = ImuProcessor(sample_rate=500.0, cutoff_freq=10.0)

    # 获取传感器地址
    try:
        acc_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")
        gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        acc_adr = model.sensor_adr[acc_id]
        gyro_adr = model.sensor_adr[gyro_id]
    except:
        print("错误：XML 中找不到 imu_acc 或 imu_gyro")
        return

    # 数据记录
    history = {
        'time': [], 
        'acc_x': [], 'acc_y': [], 'acc_z': [], 
        'gyro_x': [], 'gyro_y': [], 'gyro_z': [], # 记录三轴陀螺仪
        'truth_w_z': [] # 记录真值 Z轴角速度
    }

    print("\n=== 开始仿真：请使用手柄控制 ===")
    print("👉 请尝试原地旋转、急转弯，观察陀螺仪数据")
    print("按手柄 [Start] 键或键盘 ESC 退出并查看图表")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 0.17 
        mujoco.mj_forward(model, data)
        
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            now = step_start - start_time

            # 1. 控制更新
            controller.update()
            mujoco.mj_step(model, data)
            viewer.sync()

            # 2. 数据采集
            # 读取传感器原始数据
            raw_acc = data.sensordata[acc_adr:acc_adr+3].copy()
            raw_gyro = data.sensordata[gyro_adr:gyro_adr+3].copy()
            
            # 滤波
            filt_acc, filt_gyro = imu_filter.process(raw_acc, raw_gyro)
            
            # 获取真值 (Ground Truth)
            # data.qvel 的前6位是 FreeJoint 的速度
            # 0-2: 线速度 (vx, vy, vz)
            # 3-5: 角速度 (wx, wy, wz) -> 本体坐标系下
            # 注意：如果你的 imu_site 和 body 坐标系一致，直接取 qvel[5] 即可
            truth_w_z = data.qvel[5] 

            # 存入历史
            history['time'].append(now)
            history['acc_x'].append(filt_acc[0])
            history['acc_y'].append(filt_acc[1])
            history['acc_z'].append(filt_acc[2])
            
            history['gyro_x'].append(filt_gyro[0])
            history['gyro_y'].append(filt_gyro[1])
            history['gyro_z'].append(filt_gyro[2])
            history['truth_w_z'].append(truth_w_z)

            # 退出检测
            if controller.joystick and controller.joystick.get_button(7):
                print("检测到 Start 键，结束记录...")
                break

            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

    # --- 绘图 ---
    print("正在生成分析图表...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # 图1: 加速度 (XY)
    ax1.plot(history['time'], history['acc_x'], label='Acc X', alpha=0.6)
    ax1.plot(history['time'], history['acc_y'], label='Acc Y', alpha=0.6)
    ax1.set_title('Filtered Accel (X/Y) - Motion Trend')
    ax1.set_ylabel('m/s²')
    ax1.legend()
    ax1.grid(True)

    # 图2: 核心验证 - 陀螺仪 Z轴 VS 真值
    ax2.plot(history['time'], history['gyro_z'], label='Gyro Z (Sensor)', color='purple', linewidth=2)
    ax2.plot(history['time'], history['truth_w_z'], label='Truth Z (Physics)', color='orange', linestyle='--', linewidth=2)
    ax2.set_title('Gyro Z Reliability Check (Sensor vs Ground Truth)')
    ax2.set_ylabel('rad/s')
    ax2.legend()
    ax2.grid(True)
    
    # 图3: 陀螺仪 X/Y (检查是否平稳)
    ax3.plot(history['time'], history['gyro_x'], label='Gyro X (Roll Rate)', alpha=0.7)
    ax3.plot(history['time'], history['gyro_y'], label='Gyro Y (Pitch Rate)', alpha=0.7)
    ax3.set_title('Gyro X/Y - Stability Check (Should be near 0)')
    ax3.set_ylabel('rad/s')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()