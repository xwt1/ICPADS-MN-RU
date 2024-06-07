#!/bin/bash

# 获取脚本所在的路径
SCRIPT_DIR=$(cd $(dirname $0); pwd)

# 获取上级路径
PARENT_DIR=$(dirname "$SCRIPT_DIR")
# 定义要写入的JSON文件名，并放在脚本相同的路径下
JSON_FILE="$SCRIPT_DIR/global_config.json"
# 创建JSON内容
JSON_CONTENT=$(cat <<EOF
{
    "root_path": "$PARENT_DIR"
}
EOF
)
# 将JSON内容写入文件
echo "$JSON_CONTENT" > $JSON_FILE
# 输出确认信息
echo "The parent directory path has been written to $JSON_FILE"
