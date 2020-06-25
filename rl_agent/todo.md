TODO
====

* Implement Actor network
* Implement Critic network
* Implement training of Actor network
* Implement training of Critic network 
* Implement PID environment

Actor:
- Given state, give action tuple `(Kd', Kp', alpha)` (forward step)
- Given state, give back gradient wrt each parameter in the neural network (backprop)

Critic:
- Given state, give action tuple `(Kd', Kp', alpha)` (forward step)
- Given state, give back gradient wrt each parameter in the neural network (backprop)

PID Environment:
- Mimics Gym environment
- Supports
```python
new_state, reward, done = env.step(action)
```
Where action is `(Kd', Kp', alpha)`
- Supports `env.reset()` to restart the system
- PID controls a system implemented taken from one of the
equations in the fuzzy logic paper


Small things
============

- Reward function check
- Check update equations
- Hyperparameters involved: gamma




