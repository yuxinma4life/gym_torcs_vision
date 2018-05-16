## Overview

A deep reinforcement learning project based on [Gym-TORCS](https://github.com/ugo-nama-kun/gym_torcs) and
[DDPG-Keras-Torcs](https://github.com/yanpanlau/DDPG-Keras-Torcs).

## Dependencies

- Ubuntu 16.04
- Python3
- xautomation (http://linux.die.net/man/7/xautomation)
- OpenAI-Gym (https://github.com/openai/gym)
- numpy
- OpenCV3.x
- Keras 2.0.6
- TensorFlow 1.4.0

## Build

```bash
$ mkdir -p ~/gym_torcs
$ cd ~/gym_torcs/
$ git clone --recursive https://github.com/htsai51/gym_torcs.git
$ cd ~/gym_torcs/gym_torcs/vtorcs-RL-color
$ ./configure
$ make
$ sudo make install
$ sudo make datainstall
```

Note:
* This project uses OpenCV3 to compress Torcs image.  Modify ~/gym_torcs/vtorcs-RL-color/configure to make sure paths to OpenCV3 libs are correct (segmentation fault might happen if OpenCV2 libs are linked instead).
* You can speed up make process by typing "make -j4" with 4 as the number of cores.  If build fails with multiple cores, it could be fixed by make again.

## Run

```bash
$ cd ~/gym_torcs/gym_torcs
$ sudo python3 ddpg.py
```
