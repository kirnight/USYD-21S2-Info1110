# The University of Sydney 21S2 Info1110

# Warning:
THIS IS NOT A SOLUTION OF AN ASSIGNMENT!
DO NOT COPY THE CODE FOR YOUR ASSIGNMENT! THIS MAY CAUSE ACADEMIC HONESTY PROBLEMS!

## Please use hyperparameters from this readme. Using others might cause errors.

### To change the number of asteroids, the initial location of the spaceship and the size of the map

Open .../examples/complexb.txt 
or
You can change the name of the text at line 25, line 122 in train_main.py and do the same thing at line 25 and line 83 in test_main.py

### To modify the reward

the reward is defined at line 153 in the train_main.py

## Understand and Adjust game content

You can understand and customise your own game by adjusting config.py and game_engine.py

## Requirements

Python 3

[Pytorch](http://pytorch.org/)

[Numpy](https://numpy.org/)

[Pygame](https://www.pygame.org/)

To install requirements, follow:

```bash
# PyTorch
pip install torch

# Numpy
pip install numpy

# Pygame
pip install pygame
```

## Contributions

Contributions are very welcome. If you know how to make this code better, please open an issue. If you want to submit a pull request, please open an issue first. Also, see a to-do list below.

### TODO

* Improve this README file.
* Improve the training speed.

## Visualization

To visualize the training change 

game.train(action) -> game.test(action)

in train_main.py

## Training

```bash
cd .../assignment2
python train_main.py
```

## Testing

You should train at least one time before testing.

```bash
cd .../assignment2
python test_main.py
```

## Reference

DQN code: https://ask.csdn.net/questions/6583187