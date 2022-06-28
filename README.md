**Learning Mean-Field Control for Delayed Information Load Balancing in Large Queuing Systems**

Implemented solvers:\
**RND**: random policy \
**DJSQ**: Join the shortest out of the d queues \
**MF**: Mean field solver using PPO \
**MFRND**: Mean field solver using RND \ 
**MFDJSQ**: Mean field solver using DJSQ \

For training run the **main.py** file with the following configurations: \
solver number_agents number_queues delta_t \
Example: RND 200 200 1 \
For MF training: number_agents=number_queues=1 (single-agent MDP)

**main_parallel.py** can be used to run multiple solvers (non MF) in parallel for different configurations.

For evaluation of the trained MF policies fill in the correct path and configurations in the **main_eval_policy.py** file \
and run for different configurations.

Please check **req_curr.txt** file for required packages and their versions.
