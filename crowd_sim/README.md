# Simulation Framework
## Environment
The environment contains n+1 agents. N of them are humans controlled by certain unknown
policy. The other is robot and it's controlled by one known policy.
The environment is built on top of OpenAI gym library, and has implemented two abstract methods.
* reset(): the environment will reset positions for all the agents and return observation 
for robot. Observation for one agent is the observable states of all other agents.
* step(action): taking action of the robot as input, the environment computes observation
for each agent and call agent.act(observation) to get actions of agents. Then environment detects
whether there is a collision between agents. If not, the states of agents will be updated. Then 
observation, reward, done will be returned.


## Agent
Agent is a base class, and has two derived class of human and robot. Agent class holds
all the physical properties of an agent, including position, velocity, orientation, policy and etc.
* kinematics: holonomic (move in any direction)
* act(observation): transform observation to state and pass it to policy


## Policy
Policy takes state as input and output an action. Current available policies:
* ORCA: compute collision-free velocity under the reciprocal assumption
* CARL: learn a value network to predict the value of a state and during inference it predicts action


## State
There are multiple definition of states in different cases. The state of an agent representing all
the knowledge of environments is defined as JointState, and it's different from the state of the whole environment.
* ObservableState: position, velocity, radius of one agent
* FullState: position, velocity, radius, goal position, preferred velocity, rotation
* DualState: concatenation of one agent's full state and one another agent's observable state
* JoinState: concatenation of one agent's full state and all other agents' observable states 


## Action
There are two types of actions depending on what kinematics constraint the agent has.
* ActionRot: (velocity, rotation angle) if kinematics == 'unicycle'