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
    
    # 实例化你的控制器 (用于生成控制指令)
    controller = ChassisController(model, data)

    # ==================== 【关键修复开始】 ====================
    # 1. 获取 Joint ID (只是名单序号)
    joint_names = ['LF_wheel_joint', 'RF_wheel_joint', 'LR_wheel_joint', 'RR_wheel_joint']
    wheel_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    
    # 2. 检查 ID 是否有效
    if -1 in wheel_joint_ids:
        print(f"严重错误：XML中找不到这些关节: {joint_names}")
        return

    # 3. 【核心】获取这些关节在 qvel 数组中的真实地址 (DoF Address)
    # 因为有 freejoint 占位，必须查表 jnt_dofadr 才能拿到对的地址
    wheel_qvel_adrs = [model.jnt_dofadr[i] for i in wheel_joint_ids]
    # ==================== 【关键修复结束】 ====================

    history = {'time': [], 'encoder_vel': [], 'ground_truth_vel': [], 'slip': []}

    print("开始数据验证仿真...")
    print("  阶段 1: 静止 (0-1s)")
    print("  阶段 2: 全速前进 (1-3s) -> 此时应观测到打滑")
    print("  阶段 3: 停止 (3-5s)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        
        start_time = time.time()
        
        while viewer.is_running():
            now = time.time() - start_time
            if now > 5.0: break

            # --- 模拟控制输入 ---
            controller.vx = 0.0
            controller.vy = 0.0
            controller.w  = 0.0

            if 1.0 < now < 3.0:
                controller.vy = 1.0  # 模拟按下前进
            
            # 执行控制器
            controller.update()

            # 物理步进
            mujoco.mj_step(model, data)
            viewer.sync()

            # --- 数据采集 (已修复) ---
            
            # 1. 编码器推算速度 (使用修复后的地址读取 qvel)
            # data.qvel[adr] 才是真正的轮子转速
            wheel_omegas = [data.qvel[adr] for adr in wheel_qvel_adrs]
            
            # 取绝对值平均，防止 Swerve 倒车逻辑导致负数互相抵消
            v_encoder = np.mean(np.abs(wheel_omegas)) * WHEEL_RADIUS
            
            # 2. 真实底盘速度 (Ground Truth)
            # 假设 base_link 是第一个 body，其 freejoint 速度在 qvel[0:2]
            v_truth = np.sqrt(data.qvel[0]**2 + data.qvel[1]**2)

            history['time'].append(now)
            history['encoder_vel'].append(v_encoder)
            history['ground_truth_vel'].append(v_truth)
            history['slip'].append(v_encoder - v_truth)

            time.sleep(0.002)

    # --- 绘图 ---
    print("仿真结束，生成图表中...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 图1: 速度对比
    ax1.plot(history['time'], history['encoder_vel'], label='Encoder Calc Speed (m/s)', color='blue', linewidth=2)
    ax1.plot(history['time'], history['ground_truth_vel'], label='Ground Truth Speed (m/s)', color='orange', linestyle='--', linewidth=2)
    ax1.set_title('Odometry Validation: Input vs Reality')
    ax1.set_ylabel('Speed (m/s)')
    ax1.legend()
    ax1.grid(True)

    # 图2: 打滑分析
    ax2.plot(history['time'], history['slip'], color='red')
    ax2.set_title('Slippage (Encoder - Truth)')
    ax2.set_ylabel('Slip Speed (m/s)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True)
    
    # 强制重新绘制，防止卡顿
    plt.draw()
    plt.pause(0.001)
    plt.show()

if __name__ == "__main__":
    main()