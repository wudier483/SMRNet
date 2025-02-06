1、Download public datasets Human3.6M and HumanEva-I in npz format. Create a folder named data in the main directory. The contents of the file are as follows:
data
├── data_3d_h36m.npz
├── data_3d_humaneva15.npz
├── data_multi_modal
│   ├── data_candi_t_his25_t_pred100_skiprate20.npz
│   └── t_his25_1_thre0.500_t_pred100_thre0.100_filtered_dlow.npz
└── humaneva_multi_modal
    ├── data_candi_t_his15_t_pred60_skiprate15.npz
    └── t_his15_1_thre0.500_t_pred60_thre0.010_index_filterd.npz

2、Environment Setup:
Run the install script file：sh install.sh

3、Training:
For Human3.6M: python main.py --cfg h36m --mode train
For HumanEva-I: python main.py --cfg humaneva --mode train
 
4、Evaluation:
Evaluate on Human3.6M: python main.py --cfg h36m --mode eval --ckpt ./checkpoints/h36m_ckpt.pt
Evaluate on HumanEva-I: python main.py --cfg humaneva --mode eval --ckpt ./checkpoints/humaneva_ckpt.pt

5、Visualization(Part of the visualization results can be found in folder demo):
For Human3.6M: python main.py --cfg h36m --mode pred --vis_row 3 --vis_col 10 --ckpt ./checkpoints/h36m_ckpt.pt
For HumanEva-I: python main.py --cfg humaneva --mode pred --vis_row 3 --vis_col 10 --ckpt ./checkpoints/humaneva_ckpt.pt

Acknowledgement
The implementation is developed based on [HumanMAC](https://github.com/LinghaoChan/HumanMAC)

