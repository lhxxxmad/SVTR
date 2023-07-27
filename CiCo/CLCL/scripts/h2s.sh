export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
split_hosts=$(echo $ARNOLD_WORKER_HOSTS | tr ":" "\n")
split_hosts=($split_hosts)

hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/data/bsl5k.pth.tar
mkdir /mnt/bd/cxx-third/SLRT/CiCo/I3D_feature_extractor/chpt
mv bsl5k.pth.tar ../I3D_feature_extractor/chpt
hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/data/sign_feature.zip
unzip sign_feature.zip -d ./

hdfs dfs -get hdfs://haruna/home/byte_arnold_lq_mlnlc/user/chengxuxin/lhx/VL/EMCL-Net/tvr/models/ViT-B-32.pt
mv ViT-B-32.pt ./modules

export https_proxy=http://bj-rd-proxy.byted.org:3128
export http_proxy=http://bj-rd-proxy.byted.org:3128
export no_proxy=code.byted.org

pip install --upgrade nltk
pip install textaugment

git clone https://github.com/nltk/nltk_data.git
mv nltk_data/packages /home/tiger
# MSRVTT --do_train 1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.launch --nproc_per_node=8 \
--master_addr ${ARNOLD_WORKER_0_HOST} \
--master_port ${ARNOLD_WORKER_0_PORT} \
main_task_retrieval.py \
--do_train
