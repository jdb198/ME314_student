FROM tiryoh/ros2-desktop-vnc:humble

RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    build-essential \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# Install Moveit 
RUN sudo apt update && sudo apt install -y ros-humble-moveit

# Install Gazebo
RUN sudo apt install gazebo
RUN sudo apt install ros-humble-gazebo-ros-pkgs

# Install Realsense
RUN sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# Install librealsense2 debian package
RUN sudo apt install -y ros-humble-librealsense2*
RUN sudo apt install -y ros-humble-realsense2-*

# Create ALSA config to fix audio errors
RUN mkdir -p /home/ubuntu/.config/
RUN echo "pcm.!default {" > /home/ubuntu/.asoundrc
RUN echo "  type null" >> /home/ubuntu/.asoundrc
RUN echo "}" >> /home/ubuntu/.asoundrc
RUN echo "pcm.null {" >> /home/ubuntu/.asoundrc
RUN echo "  type null" >> /home/ubuntu/.asoundrc
RUN echo "}" >> /home/ubuntu/.asoundrc

COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
