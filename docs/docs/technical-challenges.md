# Technical Challenges & Surprises

Issues I ran into while building the AIC competition system.

---

## 1. SC Connector Won't Insert

The SC port needs way more depth than SFP (-0.045m vs -0.025m). I spent a full night collecting PPO data and got zero successful SC insertions. The reward function was also wrong for SC — it was using TCP-to-port distance which collapsed too early and gave no useful learning signal. Had to redesign the reward with connector-specific depth targets and force-based shaping.

## 2. Robot Thinks It's Done Too Early

During evaluation without ground truth, the robot declared "seated" at only 10mm depth and stopped pushing. Score was 19 instead of 75+. The rule-based phase classifier used fixed force thresholds that don't work when forces are low. I replaced it with a Gaussian HMM that tracks belief over 5 contact phases using Bayesian updates, plus a depth gate so it can't declare seated unless it's deep enough.

## 3. Zombie Middleware Processes

ROS 2 Zenoh middleware keeps running after Gazebo shuts down. The next training run fails silently because ports are taken. I had to write a cleanup script that kills everything before each run. Took me a while to figure out why training was hanging.

## 4. Installed Package vs Source Code

After editing SmartInsert.py, the robot kept running old code. Turns out the package manager caches the installed version and doesn't pick up source file edits. Had to manually copy files to the installed location every time I made changes.

## 5. CNN Training Died Silently

Kicked off 300 epochs of CNN training overnight, came back and it only ran 10. No error in the log — Python was buffering output and the process got killed quietly. Had to rerun with unbuffered output so I could actually see what happened.

## 6. Zero Episodes After Code Change

Added a retry mechanism for failed insertions, but it was too aggressive — kept retracting when it didn't need to and wasted all the time. Every trial timed out. Had to tune when retries actually trigger vs when the robot should just keep pushing.

## 7. Camera Drift During Approach

As the gripper gets closer to the port, CNN predictions drift 2-3mm because the viewpoint changes a lot. I added exponential moving average fusion where later vision updates get less weight. Still not perfect at very close range where the port starts leaving the camera's view.
