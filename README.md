# Fed PCA: Federated Learning Principal Component Analysis
This repository is for the Experiment Section of the paper: "Federated PCA on Grassmann Manifold for Anomaly Detection in IoT Networks"

Authors: Tung-Anh Nguyen, Jiayu He, Long Tan Le, Nguyen H.Tran
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
