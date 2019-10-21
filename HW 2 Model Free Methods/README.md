
# Individual Project 2
# Model-free Algorithms (Monte-Carlo and Temporal Difference)

#### Due Date
* Thursday September 26, 2019 (23:59)

#### Total Points
* 100 (One Hundred)

## Goals
In this project, you will be asked to implement two model-free algorithms. The first one is Monte-Carlo(MC), including  the first visit of on-policy MC prediction and on-policy MC control for [blackjack](https://gym.openai.com/envs/Blackjack-v0/). The second one is Temporal-Difference(TD), including Sarsa(on-policy) and Q-Learning(off-policy) for [cliffwalking](https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py).

<img src="https://github.com/huiminren/DS595CS525-RL-HW/blob/master/project2/img/project2-1.png" width="80%" >
<img src="https://github.com/huiminren/DS595CS525-RL-HW/blob/master/project2/img/project2-2.png" width="80%" >

## Deliverables

Please compress your mc.py and td.py files into a zipped file (firstName_lastName_hw2.zip) and submit it to Canvas.

## Grading

* Initial policy (2 points)
* Epsilon greedy policy (8 points)
* MC prediction (first visit, on-policy) (20 points)
* MC control (first visit, on-policy) (20 points)
* Sarsa (25 points)
* Q-Learning (25 points)<br/>

<span style="color:blue">**TA will run your code twice. You will get full credits if one of the tests passes.**</span><br/>

## Hints
* On-policy first visit Monte-Carlo prediction
<img src="https://github.com/huiminren/DS595CS525-RL-HW/blob/master/project2/img/mc_predict.png" width="80%" >

* On-policy first visit Monte-Carlo control
<img src="https://github.com/huiminren/DS595CS525-RL-HW/blob/master/project2/img/mc.png" width="80%" >

* Sarsa (on-policy TD control)
<img src="https://github.com/huiminren/DS595CS525-RL-HW/blob/master/project2/img/sarsa.png" width="80%" >

* Q-learing (off-policy TD control)
<img src="https://github.com/huiminren/DS595CS525-RL-HW/blob/master/project2/img/q-learning.png" width="80%" >

## Guidelines
* Implement functions in mc.py and td.py
* Evaluate functions by typing "nosetests -v mc_test.py" and "nosetests -v td_test.py" respectively. (put your function file and test file into the same folder)


