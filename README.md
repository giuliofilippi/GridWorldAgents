# Grid World Agents

## Original Approach
Code for replicating Khuong paper. Stigmergic construction and topochemical information shape ant nest architecture. We run every step and every move.

## Markov Approach
We extend the previous mean field model to allow for an arbitrary transition matrix T. This case includes the previous one but can be extended to a wider range of cases. We also no longer assume m is large enough to mix the distribution, we now explicitely take T^m to yield the probability distributions for pellet agents and no pellet agents. We then sample the correct number of times (without replacement) from those two distributions to get the new locations at each step. In essence this is the same as actually running the movements of the agents synchronously. We avoid num_agents*num_moves steps in this way, substituted by some overhead.

## Mean Field Approach

Mean field approximations to agent based models in grid world.

## ML Approach

THe goal is to learn the rules of the world and later learn to build in the world.
