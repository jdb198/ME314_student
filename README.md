# ME314 XArm Control Package
### Overview
The ME314_XArm package provides Python-based control and teleoperation functionalities for the xArm7 robotic arm, integrating with ROS2 for Stanford University's ME 314: Robotic Dexterity taught by Dr. Monroe Kennedy III.

### Caution
Please stay away from the robot arm during operation to avoid personal injury or equipment damage.
Ensure e-stop is close and accessible before controlling arm.

### Installation (if using native Linux Ubuntu 22.04 System)

#### Install ros2 humble (for Ubuntu 22.04)
Follow instructions for ros2 humble (desktop) install: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html or copy and paste the below commands in terminal:

```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt upgrade
sudo apt install ros-humble-desktop
```

#### Install Gazebo

```bash
sudo apt install gazebo
sudo apt install ros-humble-gazebo-ros-pkgs
```

#### Install Realsense2 SDK and ROS2 Wrapper

a. Install librealsense (source: https://github.com/IntelRealSense/realsense-ros?tab=readme-ov-file#installation-on-ubuntu step #2 option #2)

```bash
# Configure Ubuntu Repositories
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl # if you haven't already installed curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
# Install librealsense2 debian package
sudo apt install ros-humble-librealsense2*
```

b. Install RealSense Wrapper (source: https://github.com/IntelRealSense/realsense-ros?tab=readme-ov-file#installation-on-ubuntu)

```bash
# Assuming Ubuntu Repositories are already configured from previous step, install realsense2 wrapper debian package
sudo apt install ros-humble-realsense2-*
```

#### Install Moveit2

```bash
sudo apt install ros-humble-moveit
```

#### Create xarm_ros2_ws and Clone me314 repo

```bash
cd ~
mkdir -p xarm_ros2_ws/src
cd ~/xarm_ros2_ws/src
git clone https://github.com/xArm-Developer/xarm_ros2.git --recursive -b $ROS_DISTRO
git clone https://github.com/RealSoloQ/ME314_XArm.git
```

#### Build Workspace (this should only need to be done once if using --symlink-install, which prevents you from having to rebuild workspace after python file changes)

```bash
cd ~/xarm_ros2_ws
colcon build --symlink-install --cmake-args "-DCMAKE_BUILD_TYPE=Release"
```

#### Control XArm using XArm API in real (not recommended)

```bash
ros2 run me314 move_A_to_B.py
```

### Setting up Docker Image (for Windows and Mac users)
1. Install Docker Desktop (or Docker Engine for CLI only)
2. In terminal run: 

```bash
docker pull aqiu218/me314_xarm_ros2
```

3. Start container using the following command (only needs to be run once): 

docker run --privileged --name me314 -p 6080:80 --shm-size=512m -v <computer-path>:<docker-path> aqiu218/me314_xarm_ros2

** In the above command, *-v <computer-path>:<docker-path>* mounts a folder on your local computer to the docker container, thus linking any files/directories/changes made on local to your docker container ubuntu system. 
** --name sets the name of the container, this can be set to anything you want.

Example:

```bash
docker run --privileged --name me314 -p 6080:80 --shm-size=512m -v /home/alex/Documents/me314_test:/home/ubuntu/Desktop/me314 aqiu218/me314_xarm_ros2
```

4. Navigate to http://localhost:6080/ and click connect. You should now see a full Ubuntu Linux desktop environment!

5. Stop container by pressing ctrl+c in host device (your laptop) terminal

6. To run container in the future, run the following command in your terminal and navigate to http://localhost:6080/:

```bash
docker start me314
```
7. To stop container (run in terminal): 

```bash
docker stop me314
```

### Testing out the me314_pkg!

1. Navigate to terminal (if using native Linux) or navigate to Terminator by clicking on the top left menu button and searching for it.

2. Run the following commands:

```bash
cd xarm_ros2_ws
source install/setup.bash
clear
```

3. To start gazebo simulation (and RViz), run the following launch command in the same terminal:

```bash
ros2 launch me314_pkg me314_xarm_gazebo.launch.py
```

It may take a while, but you should see the xarm7 spawn on a table, with a red block in front of it. (This will likely fail the first time you run the command after starting the container, Gazebo takes a really long time to load. Let it fully load with errors and then ctrl+c and re-run the command)

4. To test an example script that commands the xarm to move from point A to B, run the following command in a separate terminal while gazebo and rviz are open:

```bash
cd xarm_ros2_ws
source install/setup.bash
clear
ros2 run me314_pkg xarm_a2b_example.py
```

### Commands Summary
#### Navigate to Workspace and Source Install Before Running Any Commands

```bash
cd ~/xarm_ros2_ws
source install/setup.bash
```

#### Tele-Operation with Spacemouse

```bash
ros2 run me314_pkg xarm_spacemouse_ros2.py
```

#### Control XArm using XArm Planner (with MoveIt API) (RECOMMENDED)

1. Control in Gazebo

a. In one terminal run the following command:

```bash
ros2 launch me314_pkg me314_xarm_gazebo.launch.py
```

b. In another terminal run script (example):

```bash
ros2 run me314_pkg xarm_a2b_example.py
```

2. Control in Real

a. In one terminal run the xarm planner launch command:

```bash
ros2 launch me314_pkg me314_xarm_real.launch.py
```

b. In another terminal run script (example):

```bash
ros2 run me314_pkg xarm_a2b_example.py
```

### Tips/Notes

- If using Terminator, ctrl+shift+E is shortkey to open side by side terminal tab
- If xarm isn't spawning in gazebo, try quitting and re-running launch command
- For more info about docker check out this quickstart guide: https://github.com/armlabstanford/collaborative-robotics/wiki/Docker-Quickstart
- Docker cheat sheet commands here: https://docs.docker.com/get-started/docker_cheatsheet.pdf 

