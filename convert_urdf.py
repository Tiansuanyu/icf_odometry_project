import mujoco
import os

def convert_urdf_to_mjcf(urdf_path, output_xml_path):
    # 1. 加载模型（MuJoCo 会自动解析 URDF 逻辑）
    # 如果 URDF 中有 <mujoco> 标签，它会被解析为编译选项
    model = mujoco.MjModel.from_xml_path(urdf_path)
    
    # 2. 使用 mj_saveLastXML 将其导出为标准 MJCF
    # 这是 MuJoCo 3.x 推荐的持久化方法
    mujoco.mj_saveLastXML(output_xml_path, model)
    
    print(f"成功导出: {output_xml_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_input = os.path.join(current_dir, "chassis.urdf")
    xml_output = os.path.join(current_dir, "chassis.xml")
    
    if os.path.exists(urdf_input):
        convert_urdf_to_mjcf(urdf_input, xml_output)
    else:
        print(f"错误: 找不到文件 {urdf_input}")