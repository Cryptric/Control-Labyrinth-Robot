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
   

## TODOs

### Non-linear sampling mpc
- Make simulation callable from python
- Run control on simulation
- Generate better reward map

### Fix oscillations
- try fixing ball position with offset to path, will oscillations happen?
- try improving ball position inference
  - Find position, angle, where mapping is bad
  - Check center
  - Check angles
  - Check with ggb model

### Simulation
- check influence of friction
- what happens on wall collision (simplify to proportional reduction in speed)
- (compare efficiency with other physics engine)


### Improve control signal sampling
- Use high level path
  - ???
- To gradient based method
- Sample control signals with low frequency
  - maybe allow high frequency for collisions


### Path extraction from image pipeline
- use other features like path width to control speed of ball

### Report
- compare ball tracking methods
  - latency
  - accuracy
  - variance
- describe vision pipeline with sequence of images