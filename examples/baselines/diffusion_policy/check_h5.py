import h5py
import os
import argparse

def inspect_h5_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 '{file_path}'")
        print("请检查路径是否正确，或者尝试使用绝对路径。")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            print(f"✅ 成功打开文件: {file_path}")
            
            # 1. 获取根目录下的所有 keys
            root_keys = list(f.keys())
            
            # 过滤掉常见的非数据键
            non_data_keys = ['env_info', 'metadata']
            data_keys = [k for k in root_keys if k not in non_data_keys]
            
            print("\n" + "="*40)
            print(f"📊 [数据统计]")
            print(f"总数据条数 (Episodes/Trajectories): {len(data_keys)}")
            if len(root_keys) > len(data_keys):
                print(f"发现非数据键: {[k for k in root_keys if k in non_data_keys]}")
            print("="*40)

            # 2. 打印具体的数据结构 (仅展示第一条数据以供参考)
            if len(data_keys) > 0:
                first_key = data_keys[0]
                print(f"\n📂 [数据结构示例] 仅展示 '{first_key}' 的内部层级:")
                print(f"+ {first_key}/ (Group)")

                def print_structure(name, obj):
                    indent = "    " * (name.count('/') + 1)
                    node_name = name.split('/')[-1]
                    
                    if isinstance(obj, h5py.Dataset):
                        print(f"{indent}- 📄 {node_name}: Dataset, shape={obj.shape}, dtype={obj.dtype}")
                    elif isinstance(obj, h5py.Group):
                        print(f"{indent}+ 📁 {node_name}/ (Group)")

                f[first_key].visititems(print_structure)
            else:
                print("\n⚠️ 文件中没有找到任何数据组。")

    except Exception as e:
        print(f"❌ 读取文件时发生错误: {e}")

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="查看 ManiSkill HDF5 (.h5) 数据集的结构和条数")
    parser.add_argument(
        "file_path", 
        type=str, 
        help="请提供 .h5 文件的路径 (可以是相对路径或绝对路径)"
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 将解析到的路径传入函数
    inspect_h5_dataset(args.file_path)