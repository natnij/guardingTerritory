# Multi-Agent Reinforcement Learning in Strategy Games

This is an experimental project which implements Q-Learning, Actor-Critic learning, and Q($\lambda$)-Learning with fuzzy systems to play a multi-player strategy game of guarding-territory, with multiple invaders and defenders.

## Fuzzy Q-Learning
    
### Description of the game

- Given territory area, there are two types of players: invader and defender
- The game has invaders who follow the NE strategy
- Defenders to learn how to intercept as early in the game as possible
- All agents have the same speed. Control variable is the defenders' direction in the next move
- Q-table sharing between all defenders at the end of each round

### The fuzzy inference system

$n$ continuous inputs (angle of the invader and of the territory relative to the defender) are defuzzified into $M$ fuzzy rules. Output is a single numerical number representing the direction of the defender's next move. The rules $l (l = 1, \cdots, M)$ can be described as:

$$R^l : \text{IF } x_1 \text{ is } F_1^l, \cdots, \text{ and } x_n \text{ is } F_n^l \text{ THEN } u = c^l$$

where $x = (x_1, \cdots, x_n)$ are inputs, $F_i^l$ are the fuzzy sets corresponding to each input, $u^l$ are the infered output from each rule, $c^l$ are the center of each rule. Using _product inference engine_ and weighted average defuzzification, the final output can be written as:

$$U(x) = \cfrac{\sum\limits_{l=1}^M ( \left( \Pi_{i=1}^n \mu^{F_i^l}(x_i) \right) \cdot c^l )}{\sum\limits_{l=1}^M \left( \Pi_{i=1}^n \mu^{F_i^l}(x_i) \right)} = \sum\limits_{l=1}^M \Phi^l c^l$$

where $\mu^{F_i^l}$ is the membership function of fuzzy set $F_i^l$.

### Q-learning

After fuzzification, the two continuous inputs are discretized into 8 discrete values each. The 64 combinations of the input pairs become 64 states of the Q-learning system. 

In this case the choice of inference engine is Mamdani system, where the output action is a set of eight discrete numbers (corresponding to eight possible directions of the defender). The Mamdani system then uses a defuzzification module (in my case a linear function with $\Phi_l$ as the weight for each rule) to convert the discrete set back into continuous space.  

In the actor-critic algorithm (see next chapter), the Takagi-Sugeno inference system is used where the output is directly in continuous space.

$q(l,a)$ records the probability of choosing action $a$ under rule $l$. $Q* = max_a \sum\limits_l^M \Phi_l q(l,a)$ is used as the TD target. Value error $VE = \sum (\text{TD target}-\hat{Q})^2$ and 

$$q_{t+1}(l,a) = q_{t} + \text{const} \frac{\partial VE}{\partial q} = q_{t} + \eta (\text{TD target}-\hat{Q}) \frac{\partial Q}{\partial q}$$  

$$\quad\quad\quad\quad = q_{t} + \eta \epsilon \frac{\partial Q}{\partial q} = q_{t} + \eta \epsilon \Phi_t^l$$

is the update equation. Using the FIS parameters: action space $A=\{ a_1,a_2,\cdots,a_m \}$, final output (action): 

$$U_t(x_t)=\sum\limits_{l-1}^M \Phi_t^l a_t^l$$

where actions are selected according to the $\epsilon$-greedy algorithm to ensure exploration:

$$a^l = \begin{cases}
	\text{random action from A} & \quad Prob(\epsilon) \\
	argmax_{a \in A}(q(l,a)) & \quad Prob(1-\epsilon)
\end{cases}$$

And value of the action is 

$$Q_t(x_t) = \sum\limits_{l-1}^M \Phi_t^l q_t(l,a_t^l)$$

$$Q_t^{*}(x_t) = \sum\limits_{l-1}^M \Phi_t^l max_{a \in A} q_t(l,a)$$

TD-error: $\epsilon_{t+1} = r_{t+1} + \gamma Q_t^{*}(x_{t+1}) - Q_t(x_t)$

Parameter update: $q_{t+1}(l,a_t^l) = q_t(l,a_t^l) + \eta \epsilon_{t+1} \Phi_t^l$

Pseudo code:

>Initialize $q(\cdot) = 0$ and $Q(\cdot) = 0$ 
>for Each time step do: 
&nbsp;&nbsp;Choose action for each rule based on $\epsilon$ at time t;  
&nbsp;&nbsp;Compute global continuous action $U_t(x_t)$;  
&nbsp;&nbsp;Compute $Q_t(x_t)$;  
&nbsp;&nbsp;Take $U_t(x_t)$ and run the game;  
&nbsp;&nbsp;Obtain reward $$r_{t+1}$ and new inputs $x_{t+1}$;  
&nbsp;&nbsp;Compute $Q_t^{*}(x_{t+1})$;  
&nbsp;&nbsp;Compute TD error $\epsilon_{t+1}$;  
&nbsp;&nbsp;Update $q_{t+1}(l,a_t^l), l = 1,\cdots, M$;  
end for

Since the actual reward can only be obtained at the end of each episode, a reward-shaping is used to estimate the reward in each time step. In my case it is simply the difference between invader angle and territory angle - it is assumed that a better direction for the defender is to be in between invader and territory, therefore a better angle difference would be 180 degrees.

### Results

Positions for two invaders and three defenders are randomly set within a range. Territory is fixed. The invaders will calculate their NE strategy in the beginning of the game and adhere to the strategy. Defenders have to learn their strategies to intercept. All agents have speed of 1 and can turn in any direction at any time step.

The game runs for n=100 episodes, and at the end of each episode, the q-tables are shared between all defenders with a simple weighting function. 

>round 9 won: 3 out of the last 10 games  
round 19 won: 14 out of the last 10 games  
round 29 won: 15 out of the last 10 games  
round 39 won: 10 out of the last 10 games  
round 49 won: 9 out of the last 10 games  
round 59 won: 17 out of the last 10 games  
round 69 won: 13 out of the last 10 games  
round 79 won: 12 out of the last 10 games  
round 89 won: 14 out of the last 10 games  
round 99 won: 19 out of the last 10 games

red dots are defenders, green dots are invaders. the lines are the trajectories. 

first round: random initialization:

![alt_text](doc/images/round1.jpg)

third round: defenders which are initialized to positions which are similar to the rounds before can already successfully intercept. However more exploration is needed to learn a better strategy with different positions:

![alt_text](doc/images/round3.jpg)

60th round: all defender agents have a higher rate of interception:

![alt_text](doc/images/round60.jpg)

## Fuzzy Actor-Critic Learning

Second algorithm tested is the actor-critic learning. 

### Policy gradient

$s_t$: state in time step t

$a_t$: action in time step t

$\theta$: policy parameters, to be learned

$H$: horizon, maximum time steps

$N$: number of episodes

$\tau=(s_0,a_0,s_1,a_1,\cdots,s_H,a_H)$ trajectory

$r(s_{i,t},a_{i,t})$: the ith trajectory，reward at ith step

$p(s_{t+1}|s_t,a_t)$：transition probability

$p_{\theta}(\tau)$: probability of selecting trajectory tau under policy theta

$J(\theta)$: reward of policy theta

objective：maximize $J(\theta)=\sum\limits_t^H r(s_{t},a_{t})=\int p_{\theta}(\tau) r(\tau) d\tau$

update equation: $\theta = \theta + \alpha \cfrac{\partial J(\theta)}{\partial \theta}$

we need to get: $\triangledown_{\theta} J(\theta) = \cfrac{\partial J(\theta)}{\partial \theta}$

to avoid vanishing/exploding gradient, use log: $\triangledown_{x} log(f(x))=\cfrac{1}{f(x)}\triangledown_x f(x)$:

$$\triangledown_{\theta} J(\theta) = \int \triangledown_\theta p_\theta(\tau) r(\tau) d\tau$$  

$$\quad\quad\quad\quad = \int p_\theta(\tau) \triangledown_\theta log(p_\theta(\tau)) r(\tau) d\tau$$  

$$\quad\quad\quad\quad = \mathbb{E}_{\tau \sim p_\theta(\tau)}[ \triangledown_\theta log(p_\theta(\tau)) r(\tau) ]$$

where

$$p_\theta(\tau)=p(s_0)\prod_{t=1}^H p_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)$$  

$$\implies log(p_\theta(\tau)) = log(p(s_0)) + \sum\limits_t^H log(p_\theta(a_t|s_t)) + \sum\limits_t^H log(p(s_{t+1}|s_t,a_t))$$  

$$\implies \triangledown_\theta log(p_\theta(\tau)) =\sum\limits_t^H \triangledown_\theta log(p_\theta(a_t|s_t))$$  

$$\implies \triangledown_{\theta} J(\theta) = \cfrac{1}{N} \sum\limits_i^N ( \sum\limits_{t=1}^H \triangledown_\theta log(p_\theta(a_t|s_t)) \sum\limits_{t'=t}^H r(s_{i,t},a_{i,t}) )$$  

Since we have added a white noise around the output action $u_t$, $p_\theta(a_t|s_t)=\mathcal{N}(f(s_t),\sigma)$ 

based on the pdf of gaussian distribution we get:

$$p_\theta(a_t|s_t)=\cfrac{1}{\sqrt{2\pi \sigma}}\exp( -\cfrac{1}{2} \cfrac{(x-f_\theta(s_t))^2}{\sigma} )$$ 

$$\implies log(p_\theta(a_t|s_t)) = -\cfrac{1}{2}log(2\pi\sigma) -\cfrac{1}{2} \cfrac{(x-f_\theta(s_t))^2}{\sigma}$$

$$\implies \triangledown_\theta log(p_\theta(a_t|s_t)) = \cfrac{x-f_\theta(s_t)}{\sigma}\triangledown_\theta f_\theta(s_t)$$

In the case of actor-critic learning, actor is using policy gradient with reward from the critic (which uses Q-learning):

$$Q(s_t,a_t)=\sum\limits_{t'=t}^H \mathbb{E}_{p_\theta} [ r(s_{t'},a_{t'})|s_t,a_t ]$$ 

is the TRUE EXPECTED reward-TO-GO, or total reward from taking a_t in state s_t. To reduce variance, baseline V is subtracted from TD target:

$$V(s_t)=\mathbb{E}_{a_t \sim p_\theta(a_t|s_t)} [ Q(s_t,a_t) ] = \sum\limits_{t'=t}^H \mathbb{E}_{p_\theta}[ r(s_{t'},a_{t'}|s_t) ]$$

is the TRUE EXPECTED state-value. or total reward from $s_t$.

$$\implies \triangledown_\theta J(\theta) = \cfrac{1}{N} \sum\limits_i^N \sum\limits_t^H \triangledown_\theta log(p_\theta(a_{i,t}|s_{i,t})) ( Q(s_{i,t},a_{i,t}) - V(s_{i,t}) )$$

Advantage: $A(s_t,a_t) = Q(s_{t},a_{t}) - V(s_{t})$, and

$$Q(s_t,a_t)=r(s_{t+1},a_{t+1}) + \mathbb{E}_{s_{t+1} \sim p(s_{t+1}|s_t,a_t)} [ V(s_{t+1}) ] \approx r(s_{t+1},a_{t+1}) + V(s_{t+1})$$  

$$\implies A(s_t,a_t) \approx r(s_{t+1},a_{t+1}) + V(s_{t+1}) - V(s_{t})$$  

$$\implies \triangledown_\theta J(\theta) = \cfrac{1}{N} \sum\limits_i^N \sum\limits_t^H \triangledown_\theta log(p_\theta(a_{i,t}|s_{i,t})) A(s_{i,t},a_{i,t})$$

if to add discount factor $\gamma$ to the future terms, we get the same formulation as in my experimental case.

### Actor-Critic Learning

![alt_text](doc/images/actorCritic.jpg)

In my case, using FIS parameters in the actor-critic:

**Actor**

$$\cfrac{\partial V_t}{\partial w_t^l} = \triangledown_w \log(p_w(s_t,u'_t)) A(s_t,u'_t)$$

where $p_w(s_t,u'_t) = \mathcal{N}(u_t, \sigma)$, 

$$A(s_t,u'_t))=r_{t+1} + \gamma \hat V_{t+1} - \hat V_t = \Delta$$

and $\cfrac{\partial u_t}{\partial w_t^l}=\phi_t^l$

$$\implies \cfrac{\partial V_t}{\partial w_t^l} = \Delta \cfrac{\hat{u}_t-u_t}{\sigma} \phi_t^l$$

where $\hat{u}_t = normal(u_t, \sigma)$ is the final output action.

The FIS engine in the actor module is a Sugeno type inference system $u_t = \sum\limits_{l=1}^M \Phi^l w_t^l$ whose linear parameters $w_l$ are to be learned:

$$w_{t+1}^l = w_{t}^l + \beta \text{sign } \{ \Delta (\cfrac{u'_t-u_t}{\sigma} ) \} \cfrac{\partial u}{\partial w^l}$$

TD error $\Delta = r_{t+1} + \gamma \hat{V}_{t+1} - \hat{V}_t$, 

$\cfrac{\partial u}{\partial w^l} = \Phi_t^l$, and $\beta \in (0,1)$$ is the actor's learning rate，$u'_t$ is the actual output action, $u_t$ is the calculated output from actor, $\sigma$ is the variance of the white noise added, $\gamma$ is the discount factor.

**Critic**

State value $V_t = E{ \sum\limits_{k=0}^{\infty} \gamma^k r_{t+k+1} \}$ or $V_t = r_{t+1} + \gamma V_{t+1},\hat{V}_t = \sum\limits_{l=1}^M \Phi^l c_t^l$

update: $c_{t+1}^l = c_t^l + \alpha \Delta \cfrac{\partial \hat{V}}{\partial c^l}$ where $\cfrac{\partial \hat{V}}{\partial c^l} = \Phi^l$

$\alpha \in (0,1)$ is the critic's learning rate.

>Initialize $\hat{V} = 0, c^l = 0, w^l = 0 \text{ for } l = 1,\cdots, M$,  
$\quad\alpha_a, \alpha_c, \sigma, \gamma$.  
>for Each time step do:  
>&nbsp;&nbsp;&nbsp;&nbsp;Obtain inputs $x_t$;  
>&nbsp;&nbsp;&nbsp;&nbsp;Calculate output of actor:$u_t = \sum\limits_{l=1}^M \Phi^l w_t^l$;  
>&nbsp;&nbsp;&nbsp;&nbsp;Calculate output of critic: $\hat V_t = \sum\limits_{l=1}^M \Phi^l c_t^l$;  
>&nbsp;&nbsp;&nbsp;&nbsp;Run the game for current time step;  
>&nbsp;&nbsp;&nbsp;&nbsp;Obtain reward $r_{t+1}$ and new inputs $x_{t+1}$;  
>&nbsp;&nbsp;&nbsp;&nbsp;Calculate $\hat V_{t+1}$ from $\hat V_t = \sum\limits_{l=1}^M \Phi^l c_t^l$;  
>&nbsp;&nbsp;&nbsp;&nbsp;Calculate $\Delta = r_{t+1} + \gamma \hat V_{t+1} - \hat V_t$;  
>&nbsp;&nbsp;&nbsp;&nbsp;Update $c_{t+1}^l=c_t^l+\alpha \Delta \Phi_t^l$,  
>$\quad w_{t+1}^l=w_t^l+\beta \text{sign } \{ \Delta ( \cfrac{u'_t-u_t}{\sigma} ) \} \Phi_t^l$.  
>end for.

## QLFIS (Q($\lambda$)-learning fuzzy inference system

Another algorithm tested is the QLFIS algorithm, which consists of a fuzzy logic controller and an estimator. The FLC is a TD($\lambda$)-learning algorithm, whereas the estimator simply uses Q-learning. 

### QLFIS

![alt_text](doc/images/maml5_FIS.jpg)

The FLC and the estimator (FIS) have independent membership functions and parameters to learn (e.g. in the case of gaussian membership function, the mean and variance of the MF is also learned). 

$\alpha$ as FLC's learning rate, $\beta$ as estimator FIS's learning rate, membership degrees are $\Phi_{FLC}^l$ and $\Phi_{FIS}^l$, $w^l$ is the FLC's output parameters, $c^l$ the FIS's output parameters, $\Delta$ is TD error,  $u_t$ the calculated output action, $u'_t$ is the actual output action with white noise, $Q(s,a)$ is the state action value, r is reward，$e_t$ is eligibility trace.

## Comparison of the three algorithms

Simulation:

- 1 invader playing NE strategy, 2 defenders learning best interception strategy
- 20 rounds played, each with 100 episodes
- Average interception distance and average games won are recorded
- Defenders share parameters at the end of each episode. Importance of each update is weighed by the interception distance.
- Reward-shaping is done in every time step. The reward is based on difference between invader angle and territory angle.
- At the end of each round the parameter tables are reset.
- Fuzzy Q-learning (FQL), fuzzy actor-critic (FACL), Q(lambda)-learning with FIS parameter learning (QLFIS) are tested

Result:

- Except for FQL, the other two algorithms did not learn. 
- The FACL has very high variance in both number of winning games and interception distance.
- The QLFIS algorithm updates very slowly due to increased number of parameters to be learned in the FIS systems. 

Potential improvements: 
- Other (better) ways of reward shaping
- Other parameter sharing algorithms at the end of the episode
- Other RL algorithms e.g. PPO

![alt_text](doc/images/FIS_winningGame.jpg)

![alt_text](doc/images/FIS_distance.jpg)
