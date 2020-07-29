Q Learning Reinforcement learning based Google chrome dino Game (on going)
===============================================================

![Alt text](images/dino.gif?raw=true "Chrome Dino")

## Markov Decision Process.
Please refer this blog by anrej http:// karpathy.github.io /2016/05/31/rl/ that should give you intution behind reinforcement learning.

## Q Learning

Q-learning is a model-free reinforcement learning technique were an optimal action selection policy for a finit markov decison process is done by picking the action corresponding highest Q value.
Think of every game frame as a state. Our agent (Dino) takes an action (Jump/no jump) and the next frame we get a new state. Lets say our agent is in a particular state where right infront of it there is a cactus.
If dino jumps then in next state it survives so we give the dino a positive reward, if the dino doesn`t jump in the next state it dies so we give it a negative reward. Over playing many time our dino gets really good at tackling obstucles. In Q Learning our network takes game screen shots and outputs two Q values corresponding to two actions. We pick the action corresonding to highest Q value.


## Finding game terminal state.
Screenshots will be taken cropped for our network as input. script crops the game section (game screen) and fail section (part which indicates game ended). Now this cropping section changes from person to person based on screen resolution so you have to expriment around and find it out. While playing the game same region of interests will be cropped and compared againts the fail reference section. If the game did end then the captures screenshot cropped fail reference region will match actual fail reference region and this is implmented in the function mse(image a, imageb). did_game_end function will load base fail reference image i.e the image indicating game ended or not and commapres againts the cropped section we get for the current frame. Now I have used boxcutter cli util for windows since I use windows for game based experiments. On linux you can use your own utils.
This is what my base fail reference looks like
![Alt text](images/fail.png?raw=true "Fail reference")

## Training
Input image will be shape [width=640, height=160, channel=1]. Images will be cropped and converted to Gray scale. Four game frames will be stacked and will be fed through 4 four layer convolution network. The network output two Q values. We see the output and pick the index which has highest values and perform that action and capture next frames. We check if game ended or not. If game doesnt end we give a discounted positive reward. If game ends we give a negative reward.

```python
y_batch = []
readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
for i in range(0, len(minibatch)):
	terminal = minibatch[i][4]
    # if terminal, only equals reward
    if terminal:
    	y_batch.append(r_batch[i])
    else:
        y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

    # perform gradient step
    train_step.run(feed_dict = {
                    y : y_batch,
                    a : a_batch,
                    s : s_j_batch}
                   )
```

Our y batch holds all such action value pairs and we perform gradient update on our network based on these values.

## Testing
You can let the model run over night you can see some improvemenets over the performance on how the dino takes right actions. Now I still havent found time to get back on this project complete the training as I need my GPU for other research but the current model should play decently good.

## Additional References
1. https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
2. http://karpathy.github.io/2016/05/31/rl/