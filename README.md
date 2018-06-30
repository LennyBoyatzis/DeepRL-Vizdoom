## Q-Learning
---
- Q-Learning Algorithms vs Policy-Based Algorithms
- Start with simple lookup table algo
- Then look at Neural Network equivalent

```
Unlike policy gradient methods, which attempt to learn functions which
directly map an observation to an action, Q-Learning attempts to learn
the value of being in a give state, and taking a specific action there
```

- Frozen lake
- Need an algorithm which learns long-term expected rewards
- This is exactly what q-learning is designed to provide

```
In its simplest form q-learning is a table of values for every state (row)
& and action (column) possible in the environment.
```

- Start by initializing the Q-table to be uniform (all-zeros)
- Observe rewards obtained for various actions & update table accordingly
- Make updates to the Q-table using something called the `Bellman equation`

```
Bellman equation:
The expected long-term reward for a given action is equal to the immediate
reward from the current action combined with the expected reward from the best
future action taken at the following state
```

### Tables are GREAT but they DON'T SCALE
---
- No worries for a simple grid world (16 * 4)
- For any interesting problems, tables simply don't work

```
We need a way to take a description of our state, and produce Q-values
for actions without a table, this is where neural networks come in
```

## Deep Q Network Learning
---

- Q-table
- Q-network
- Deep Q-network

To transform an ordinary QN into a DQN
---
1. Go from single-layer network to multi-layer convolutional network
2. Implement Experience Relay: allow network to train itself using stored
memories from its experience
3. Utilize a second target network, used to compute target q-values

These 3 innovations allow Google Deepmind to kill the game (2014)

Things have progressed since 2014
---
Improvements to DQN Arch since then
1. Double DQN
2. Dueling DQN

Allow for improved performance, stability and faster training time

Addition 1: Convolutional Layers
---
- Instead of considering each pixel independently, convolutional layers
allow us to consider regions of an image, and maintain spatial relationships
between objects on the screen

In Tensorflow, we can utilize the tf.contrib.layers.convolution2d function to 
easily create a convolutional layer

```
conv_layer = tf.contrib.layers.convolution2d(inputs, num_outputs, kernal_size,
stride, padding)
```
- `num_outputs`: number of filters applied to previous layer
- `kernel_size`: how large a window we would like to slide over previous layer
- `stride`: how many pixels to skips as we slide the window across the layer

Addition 2: Experience Replay
---
- Basic idea is that by storing the agents experiences, and then randomly
drawing batches of them to train the network
- By keeping the experiences random we prevent the network from only learning
  about what it is immedately doing in the environment and allow it to learn
  from a more varied array of past experiences
- Each experience is stored as a tuple of `(state, action, reward, next_state)`
- The Experience Replay buffer stores a fixed number of recent memories
- As new ones come in, old ones are removed
- When it comes to training we draw a uniform batch from the buffer and train
our network with them
- Build simple class to `store` and `retrieve` memories

Addition 3: Separate Target Network
---
- Use a 2nd network during training procedure
- Used to generate the target Q-values that will be used to compute the loss
  for every action during training
- The target networks weights are fixed and only periodically or slowly updated
  to the primary Q-networks values
- This makes training more stable

Double DQN
---


### OUR ALGO
----
- Plays the game:
    - each step saves initial state
    - the action it took
    - the reward it got
    - the next state it reached
- This data is used to train the Neural Network
- The input of the Neural Network wil be an initial state
    - i.e. a stack of 4 preprocessed frames
    - output with be its estimate of Q(s, a)

Experience Replay
---
- At each Q-learning iteration, you play one step in the game
- Instead of updating the model based on the last step
- Add all relevant information from the step you just took (current state, next
  state, action taken, reward and whether the next state is terminal) to a finite-size memory
- And then call fit batch on the sample of that memory

Why is Experience Replay useful?
---
- Successive states are highly similar
- THEREFORE there is significant risk that the network will completely forget
  about what its like to be in state that it hasn't been in for a while.
  Replaying prevents this by still showing old frames to the network

NOTE: ER requires loads of memory
---
- A single preprocessed 110x80 frame takes around 9kb
- With ER we store 1M of those ~ 9GB in frames, all in RAM
- Store frames in `np.uint8` type and convert them into floats in [0,1] range
  at the last moment
- `np.float32` will use 4x as much space
- `np.float64` will use 8x as much space
- Never copy frames (each frame will be part of 4 unique states). Make sure
  each unique frame uses the same memory in each state
- Don't call `np.array()` multiple times on a given frame
- Nor `copy.copy`
- Be careful when serializing memory

### Knowing when to explore
---
- How to choose what action to play while learning?
- When DQN is bad, we want to explore, otherwise, we keep following whatever
our initial random weight initialization told it to 
- When it gets better, we want to exploit more
- Use epsilon greedy
    - With probability epsilon, we choose a random action
    - Otherwise we choose the greedy action (maximise Q-val for that state)
    - Common to start with a high `epsilon` and reduce it as DQN goes through
      more iterations
    - This is referred to as `annealing`
    - `Annealing` linearly means the value of `epsilon` is reduced by a fixed
      amount at each iteration

