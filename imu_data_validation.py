import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt

# ================= 配置区域 =================
XML_PATH = "scene.xml"
WHEEL_RADIUS = 0.03
SIM_DURATION = 5.0

# ================= 滤波器模块 (新增) =================
class ImuProcessor:
    def __init__(self, sample_rate=500.0, cutoff_freq=10.0):
        """
        模拟真实 IMU 的信号处理链
        :param sample_rate: 仿真频率 (Hz)，对应 time.sleep(0.002) -> 500Hz
        :param cutoff_freq: 截止频率 (Hz)，越小越平滑，但滞后越大。推荐 20-40Hz
        """
        # 计算低通滤波系数 alpha
        # y[i] = alpha * x[i] + (1-alpha) * y[i-1]
        dt = 1.0 / sample_rate
        rc = 1.0 / (2 * np.pi * cutoff_freq)
        self.alpha = dt / (dt + rc)
        
        # 状态缓存
        self.last_acc = None
        self.last_gyro = None

        # 模拟零偏 (Bias) - 真实 IMU 不可能完美归零
        self.acc_bias = np.array([0.05, -0.05, 0.10]) 
        self.gyro_bias = np.array([0.002, 0.002, -0.002])

    def process(self, raw_acc, raw_gyro):
        # 1. 加上 Bias (模拟器件误差)
        curr_acc = raw_acc + self.acc_bias
        curr_gyro = raw_gyro + self.gyro_bias
        
        # 2. 初始化 (第一帧直接赋值)
        if self.last_acc is None:
            self.last_acc = curr_acc
            self.last_gyro = curr_gyro
            return curr_acc, curr_gyro
        
        # 3. 低通滤波 (DLPF) - 核心降噪步骤
        filt_acc = self.alpha * curr_acc + (1.0 - self.alpha) * self.last_acc
        filt_gyro = self.alpha * curr_gyro + (1.0 - self.alpha) * self.last_gyro
        
        # 更新缓存
        self.last_acc = filt_acc
        self.last_gyro = filt_gyro
        
        return filt_acc, filt_gyro

# ================= 你的控制器代码 (保持不变) =================
MAX_SPEED = 50.0       
ROTATION_SPEED = 3.0   
RAIL_MIN = 0.00
RAIL_MAX = 0.25        

OFFSETS = {
    'front_left':   -np.pi/4, 'front_right':  +np.pi/4,
    'rear_left':    -3*np.pi/4, 'rear_right':   +3*np.pi/4
}

WHEEL_MAP_CONFIG = {
    'front_left': 'RR', 'front_right': 'LR',  
    'rear_left': 'RF', 'rear_right': 'LF',  
}

WHEEL_GEOMETRY = {
    'front_left': (-1.0, 1.0), 'front_right': (1.0, 1.0), 
    'rear_left': (-1.0, -1.0), 'rear_right': (1.0, -1.0), 
}

class ChassisController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.actuators = {}
        self.wheels = {}
        
        for name in ['LF', 'RF', 'LR', 'RR']:
            steer_act_name, drive_act_name, rail_act_name = f"{name}_steer", f"{name}_drive", f"{name}_rail"
            steer_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, steer_act_name)
            drive_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, drive_act_name)
            rail_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, rail_act_name)
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, steer_act_name)
            if joint_id != -1: qpos_adr = model.jnt_qposadr[joint_id]
            else:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_steer_joint")
                qpos_adr = model.jnt_qposadr[joint_id] if joint_id != -1 else None
            
            self.actuators[f"{name}_data"] = {'steer_id': steer_id, 'drive_id': drive_id, 'rail_id': rail_id, 'qpos_adr': qpos_adr}

        for logic_name, xml_prefix in WHEEL_MAP_CONFIG.items():
            comp_data = self.actuators[f"{xml_prefix}_data"]
            self.wheels[logic_name] = {
                'steer_id': comp_data['steer_id'], 'drive_id': comp_data['drive_id'],
                'rail_id':  comp_data['rail_id'], 'qpos_adr': comp_data['qpos_adr'],
                'pos': WHEEL_GEOMETRY[logic_name]
            }
        self.vx, self.vy, self.w, self.rail_height = 0.0, 0.0, 0.0, 0.0

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
        for name, wheel in self.wheels.items():
            if wheel['rail_id'] != -1: self.data.ctrl[wheel['rail_id']] = self.rail_height
            rx, ry = wheel['pos'] 
            wheel_vx = self.vx - (self.w * ROTATION_SPEED) * ry
            wheel_vy = self.vy + (self.w * ROTATION_SPEED) * rx
            raw_target_speed = np.sqrt(wheel_vx**2 + wheel_vy**2)
            if raw_target_speed < 0.01:
                self.data.ctrl[wheel['drive_id']] = 0.0
                continue
            raw_target_angle = np.arctan2(wheel_vy, wheel_vx) + OFFSETS[name]
            current_angle = 0.0
            if wheel['qpos_adr'] is not None:
                raw_q = self.data.qpos[wheel['qpos_adr']]
                current_angle = np.arctan2(np.sin(raw_q), np.cos(raw_q))
            opt_angle, opt_speed = self.optimize_module(current_angle, raw_target_angle, raw_target_speed)
            self.data.ctrl[wheel['steer_id']] = opt_angle
            self.data.ctrl[wheel['drive_id']] = opt_speed * MAX_SPEED

# ================= 主程序 =================
def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    
    try:
        controller = ChassisController(model, data)
    except NameError:
        print("请把 ChassisController 类的代码粘贴到这个脚本里再运行！")
        return

    # IMU ID 获取
    try:
        acc_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")
        acc_adr = model.sensor_adr[acc_id]
    except:
        print("错误：XML 中找不到 imu_acc")
        return

    # --- 实例化滤波器 ---
    # 截止频率设为 20Hz，效果会非常明显，数据会变圆润
    imu_filter = ImuProcessor(sample_rate=500.0, cutoff_freq=20.0)

    history = {
        'time': [],
        'robot_acc_derived': [], 
        'imu_acc_y': [],      
        'imu_acc_z': [],      
    }

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        
        start_time = time.time()
        last_vel = 0.0
        last_sim_time = 0.0
        
        print("仿真开始：请观察图表中的 Z轴(绿色) 是否平稳，Y轴(橙色) 是否跟随黑色真值")

        while viewer.is_running():
            now = time.time() - start_time
            if now > SIM_DURATION: break

            # 控制逻辑
            controller.vx, controller.vy, controller.w = 0.0, 0.0, 0.0
            if 1.0 < now < 3.0:
                ramp = (now - 1.0) / 0.5 
                controller.vy = 20.0 * np.clip(ramp, 0, 1)
            
            controller.update()
            mujoco.mj_step(model, data)
            viewer.sync()

            # --- 数据处理 ---
            
            # 1. 理论加速度 (真值)
            v_current = data.qvel[1] 
            dt = data.time - last_sim_time
            acc_truth = (v_current - last_vel) / dt if dt > 1e-6 else 0.0
            last_vel = v_current
            last_sim_time = data.time

            # 2. 读取原始 IMU (Raw Data)
            # 这里读出来的数据包含 noise="0.1" 的白噪声，以及物理碰撞的剧烈震荡
            raw_acc = data.sensordata[acc_adr : acc_adr+3].copy()
            # 陀螺仪如果没定义可以传空或者全0，这里暂不处理陀螺仪
            raw_gyro = np.zeros(3) 

            # 3. 【核心】通过低通滤波器 (Filtered Data)
            filt_acc, _ = imu_filter.process(raw_acc, raw_gyro)

            # 4. 存图
            history['time'].append(now)
            history['robot_acc_derived'].append(acc_truth)
            
            # 存滤波后的数据
            history['imu_acc_y'].append(filt_acc[1])
            history['imu_acc_z'].append(filt_acc[2])

            time.sleep(0.002)

    # --- 绘图 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 图1: 水平加速度 (Y轴)
    ax1.plot(history['time'], history['robot_acc_derived'], label='True Acc (dv/dt)', color='black',  linewidth=2, alpha=0.3)
    ax1.plot(history['time'], history['imu_acc_y'], label='Filtered IMU Acc Y (DLPF=20Hz)', color='orange', linestyle='-')
    ax1.set_title('Filtered IMU vs Ground Truth (Horizontal)')
    ax1.set_ylabel('Accel (m/s²)')
    ax1.legend()
    ax1.grid(True)

    # 图2: 垂直加速度 (Z轴)
    ax2.plot(history['time'], history['imu_acc_z'], label='Filtered IMU Acc Z (Gravity)', color='green')
    ax2.set_title('Filtered IMU Z-Axis (Expect ~9.8)')
    ax2.set_ylabel('Accel (m/s²)')
    ax2.set_ylim(0, 15) 
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()