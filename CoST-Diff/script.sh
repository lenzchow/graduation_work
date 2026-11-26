!/usr/bin/env bash

# 执行第一个命令
# python multi_main.py --dataset operator0 --embed_mode vgae --module_type transformer --diff_T 500 --pre_trained_dim 128 --embedding_hour_dim 128 --rdcl 10

# # 等待3秒
# sleep 3

# # 执行第二个命令
python generate_data.py --dataset operator0 --embed_mode vgae --module_type transformer --diff_T 500 --pre_trained_dim 128 --embedding_hour_dim 128 --rdcl 10

# 等待3秒
sleep 3

# 执行第一个命令
# python multi_main.py --dataset operator1 --embed_mode vgae --module_type transformer --diff_T 500 --pre_trained_dim 128 --embedding_hour_dim 128 --rdcl 10

# 等待3秒
sleep 3

# 执行第二个命令
python generate_data.py --dataset operator1 --embed_mode vgae --module_type transformer --diff_T 500 --pre_trained_dim 128 --embedding_hour_dim 128 --rdcl 10

# 等待3秒
sleep 3

# 执行第一个命令
# python multi_main.py --dataset operator3 --embed_mode vgae --module_type transformer --diff_T 500 --pre_trained_dim 128 --embedding_hour_dim 128 --rdcl 10

# 等待3秒
sleep 3

# 执行第二个命令
python generate_data.py --dataset operator3 --embed_mode vgae --module_type transformer --diff_T 500 --pre_trained_dim 128 --embedding_hour_dim 128 --rdcl 10

# # 等待3秒
# sleep 3