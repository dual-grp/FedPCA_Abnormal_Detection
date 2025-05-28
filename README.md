# Fed PCA: Federated PCA on Grassmann Manifold for Anomaly Detection in IoT Networks [INFOCOM2023]
This repository is for the Experiment Section of the paper: "Federated PCA on Grassmann Manifold for Anomaly Detection in IoT Networks"

Authors: Tung-Anh Nguyen, Jiayu He, Long Tan Le, Nguyen H.Tran

Paper link: https://arxiv.org/pdf/2212.12121.pdf 

# Software requirements:
- numpy, scipy, pytorch, Pillow, matplotlib.

- To download the dependencies: pip3 install -r requirements.txt

- The code can be run on any pc.
## Instruction to run the code
<pre></code>
!python3 main.py --algorithm FedPG --learning_rate 0.0001 --num_global_iters 100 --dim 30 --subusers 0.1 --local_epochs 30
!python3 main.py --algorithm FedPE --learning_rate 0.0001 --num_global_iters 100 --dim 30 --subusers 0.1 --local_epochs 30
<code></pre>

## Estimate training time for FedPG and FedPE
<pre></code>
!python3 main.py --algorithm FedPG --learning_rate 0.0001 --num_global_iters 1000 --dim 30 --subusers 0.1 --local_epochs 30
!python3 main.py --algorithm FedPE --learning_rate 0.0001 --num_global_iters 1000 --dim 30 --subusers 0.1 --local_epochs 30
!python3 main.py --algorithm FedPE --learning_rate 0.0001 --num_global_iters 1000 --dim 30 --subusers 0.1 --local_epochs 40
!python3 main.py --algorithm FedPE --learning_rate 0.0001 --num_global_iters 1000 --dim 30 --subusers 0.1 --local_epochs 60
!python3 main.py --algorithm FedPE --learning_rate 0.0001 --num_global_iters 1000 --dim 30 --subusers 0.1 --local_epochs 80
<code></pre>

## Dataset:
NSL-KDD

# Citation
If you find this repo useful, please cite our work.
<pre></code>
@inproceedings{nguyen2023federated,
  title={Federated PCA on Grassmann manifold for anomaly detection in IoT networks},
  author={Nguyen, Tung-Anh and He, Jiayu and Le, Long Tan and Bao, Wei and Tran, Nguyen H},
  booktitle={IEEE INFOCOM 2023-IEEE Conference on Computer Communications},
  pages={1--10},
  year={2023},
  organization={IEEE}
}
<code></pre>
