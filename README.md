# Control Labyrinth Robot


## Project setup
To set up this project follow these steps:
1. Setup and activate `python3.10` virtual environment
2. Add package repository
    ```bash
    sudo add-apt-repository ppa:inivation-ppa/inivation
    sudo apt update
    ```
3. Install `libcaer-dev`
    ```bash
    sudo apt install libcaer-dev 
    ```
4. Install `opencv`
   ```bash
   sudo apt-get install python3-opencv
   ```
5. Install requirements
    ```bash
   pip install -r requirements.txt
    ```
   

### Servo Control
#### X-Axis
- low pw moves board at 0 point up
- high pw moves board at 0 point down
#### Y-Axis
- low pw moves board at 0 point up
- high pw moves board at 0 point down