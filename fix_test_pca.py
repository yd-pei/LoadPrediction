import pandas as pd

# 原始文件路径（是以 tab 分隔的）
input_path = "./dataset/test_pca.csv"
output_path = "./dataset/test_pca_fixed.csv"

# 用 \t 作为分隔符读取
df = pd.read_csv(input_path, sep='\t')

# 保存为标准 CSV（逗号分隔）
df.to_csv(output_path, index=False)

print(f"✅ 已将 {input_path} 转换为标准 CSV，保存为 {output_path}")
print(f"✔️ 数据列：{list(df.columns)}")