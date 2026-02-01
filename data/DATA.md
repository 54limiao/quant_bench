# 数据下载帮助

## 添加 ModelScope IP 到 Hosts

为了确保网络连接稳定，可将 ModelScope 的 IP 地址添加到本地 hosts 文件中。

首先，通过本地 ping 命令获取 IP 地址（示例 IP 为 47.92.141.220）：

```bash
echo "47.92.141.220 www.modelscope.cn" >> /etc/hosts
```

## 下载数据集和模型

使用 ModelScope CLI 下载所需的数据集和模型：

```bash
# 下载数据集
modelscope download --dataset HuggingFaceH4/ultrachat_200k --local_dir ./data/datasets/HuggingFaceH4/ultrachat_200k

# 下载模型
modelscope download --model Qwen/Qwen3-8B --local_dir ./data/models/Qwen/Qwen3-8B
```
