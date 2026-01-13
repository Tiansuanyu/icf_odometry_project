import mujoco
import mujoco.viewer
import numpy as np
import time
import matplotlib.pyplot as plt

# ================= 配置区域 =================
XML_PATH = "scene.xml"  # 你的XML文件路径
WHEEL_RADIUS = 0.03     # 你的椭球体半径 (必须准确)
SIM_DURATION = 5.0      # 总仿真时间

# ================= 你的控制器代码 (保持原样) =================
# ... (为了保证运行，我把你的类完整搬过来了) ...

MAX_SPEED = 50.0       
ROTATION_SPEED = 3.0   
RAIL_MIN = 0.00
RAIL_MAX = 0.25        

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
            steer_act_name = f"{name}_steer"
            drive_act_name = f"{name}_drive"
            rail_act_name  = f"{name}_rail"
            
            steer_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, steer_act_name)
            drive_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, drive_act_name)
            rail_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, rail_act_name)

            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, steer_act_name)
            if joint_id != -1:
                qpos_adr = model.jnt_qposadr[joint_id]
            else:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_steer_joint")
                qpos_adr = model.jnt_qposadr[joint_id] if joint_id != -1 else None
            
            self.actuators[f"{name}_data"] = {
                'steer_id': steer_id, 'drive_id': drive_id,
                'rail_id': rail_id, 'qpos_adr': qpos_adr
            }

        for logic_name, xml_prefix in WHEEL_MAP_CONFIG.items():
            comp_data = self.actuators[f"{xml_prefix}_data"]
            self.wheels[logic_name] = {
                'steer_id': comp_data['steer_id'], 'drive_id': comp_data['drive_id'],
                'rail_id':  comp_data['rail_id'], 'qpos_adr': comp_data['qpos_adr'],
                'pos': WHEEL_GEOMETRY[logic_name]
            }

        self.vx = 0.0 
        self.vy = 0.0 
        self.w  = 0.0
        self.rail_height = 0.0

    # 我们不需要 key_callback，因为我们会直接修改 vx, vy
    
    def optimize_module(self, current_angle, target_angle, target_speed):
        error = target_angle - current_angle
        error = np.arctan2(np.sin(error), np.cos(error))

        if abs(error) > (np.pi / 2):
            target_angle = target_angle + np.pi
            target_speed = -target_speed
            error = target_angle - current_angle
            error = np.arctan2(np.sin(error), np.cos(error))

        scale_factor = np.cos(error)
        if scale_factor < 0.1: scale_factor = 0.0
        final_speed = target_speed * scale_factor
        final_angle = np.arctan2(np.sin(target_angle), np.cos(target_angle))
        
        return final_angle, final_speed

    def update(self):
        for name, wheel in self.wheels.items():
            if wheel['rail_id'] != -1:
                self.data.ctrl[wheel['rail_id']] = self.rail_height

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

# ================= 主测试逻辑 =================

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    
    # 实例化控制器
    # 注意：请确保 ChassisController 类在这个脚本里或者被 import
    try:
        controller = ChassisController(model, data)
    except NameError:
        print("请把 ChassisController 类的代码粘贴到这个脚本里再运行！")
        return

    # 获取传感器地址
    try:
        acc_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_acc")
        gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        acc_adr = model.sensor_adr[acc_id]
        gyro_adr = model.sensor_adr[gyro_id]
        # IMU 数据的维度 (acc=3, gyro=3)
        acc_dim = model.sensor_dim[acc_id] 
    except:
        print("错误：XML 中找不到 imu_acc 或 imu_gyro，请检查 <sensor> 标签。")
        return

    history = {
        'time': [],
        'robot_vel': [],      # 真实速度
        'robot_acc_derived': [], # 从真实速度微分算出来的"理论加速度"
        'imu_acc_x': [],      # IMU 读数 X
        'imu_acc_y': [],      # IMU 读数 Y
        'imu_acc_z': [],      # IMU 读数 Z (应该是指向地面的)
        'imu_gyro_z': []      # 陀螺仪 Z (自转)
    }

    print("开始 IMU 数据验证...")
    print("  观察点 1: 静止时 Z轴加速度是否为 9.8")
    print("  观察点 2: 起步时 Y轴(或X轴)是否有推背感波峰")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        
        start_time = time.time()
        last_vel = 0.0
        last_sim_time = 0.0
        
        while viewer.is_running():
            now = time.time() - start_time
            if now > SIM_DURATION: break

            # --- 控制逻辑：猛加速再猛刹车 ---
            controller.vx = 0.0
            controller.vy = 0.0
            controller.w  = 0.0

            if 1.0 < now < 3.0:
                controller.vy = 1.0  # 模拟按下前进
            
            controller.update()
            mujoco.mj_step(model, data)
            viewer.sync()

            # --- 数据采集 ---
            
            # 1. 真实速度 (Ground Truth Velocity)
            # 假设车头朝向是 Y 轴 (根据你之前的描述是前进)
            # 如果你的车头是 X 轴，请改用 data.qvel[0]
            v_current = data.qvel[1] 
            
            # 2. 计算"理论加速度" (Ground Truth Acceleration)
            # a = dv / dt
            dt = data.time - last_sim_time
            if dt > 1e-6:
                acc_truth = (v_current - last_vel) / dt
            else:
                acc_truth = 0.0
            
            last_vel = v_current
            last_sim_time = data.time

            # 3. 读取 IMU 数据
            # sensor_data 是一个扁平大数组，需要根据地址切片
            imu_acc = data.sensordata[acc_adr : acc_adr+3]
            imu_gyro = data.sensordata[gyro_adr : gyro_adr+3]

            history['time'].append(now)
            history['robot_vel'].append(v_current)
            history['robot_acc_derived'].append(acc_truth)
            
            history['imu_acc_x'].append(imu_acc[0])
            history['imu_acc_y'].append(imu_acc[1])
            history['imu_acc_z'].append(imu_acc[2])
            history['imu_gyro_z'].append(imu_gyro[2])

            time.sleep(0.002)

    # --- 绘图分析 ---
    print("生成图表中...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 图1: 加速度对比 (核心验证)
    # IMU 数据通常包含重力，如果你车是平的，水平轴应该和 acc_truth 吻合
    # 我们画出 X 和 Y，看哪个轴是对的
    ax1.plot(history['time'], history['robot_acc_derived'], label='True Acc (dv/dt)', color='black',  linewidth=2, alpha=0.5)
    ax1.plot(history['time'], history['imu_acc_x'], label='IMU Acc X', linestyle='--')
    ax1.plot(history['time'], history['imu_acc_y'], label='IMU Acc Y', linestyle='--')
    ax1.set_title('IMU Accelerometer vs Ground Truth Acceleration')
    ax1.set_ylabel('Accel (m/s²)')
    ax1.legend()
    ax1.grid(True)

    # 图2: Z轴加速度 (重力)
    ax2.plot(history['time'], history['imu_acc_z'], label='IMU Acc Z (Gravity)', color='green')
    ax2.set_title('IMU Z-Axis (Should be ~9.8g)')
    ax2.set_ylabel('Accel (m/s²)')
    ax2.set_ylim(0, 15) # 固定量程方便观察
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# 这里请把上面的 ChassisController 类粘贴过来，或者在你的原脚本里把 main 换成这个
if __name__ == "__main__":
    main()