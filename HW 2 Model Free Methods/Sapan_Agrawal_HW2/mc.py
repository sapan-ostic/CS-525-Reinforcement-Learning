#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
	Monte-Carlo
	In this problem, you will implememnt an AI player for Blackjack.
	The main goal of this problem is to get familar with Monte-Carlo algorithm.
	You could test the correctness of your code 
	by typing 'nosetests -v mc_test.python3' in the terminal.
	
	You don't have to follow the comments to write your code. They are provided
	as hints in case you need. 
'''
#-------------------------------------------------------------------------

def initial_policy(observation):
	"""A policy that sticks if the player score is >= 20 and his otherwise
	
	Parameters:
	-----------
	observation:
	Returns:
	--------
	action: 0 or 1
		0: STICK
		1: HIT
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	# get parameters from observ:ation
	print(observation)
	if(observation[0]>=20):
		action = 0
	else:
		action = 1	
	# action

	############################
	return action 

def mc_prediction(policy, env, n_episodes, gamma = 1.0):
	"""Given policy using sampling to calculate the value function 
		by using Monte Carlo first visit algorithm.
	
	Parameters:
	-----------
	policy: function
		A function that maps an obversation to action probabilities
	env: function
		OpenAI gym environment
	n_episodes: int
		Number of episodes to sample
	gamma: float
		Gamma discount factor
	Returns:
	--------
	V: defaultdict(float)
		A dictionary that maps from state to value
	"""
	# initialize empty dictionaries

	returns_sum = defaultdict(float)
	returns_count = defaultdict(float)

	# a nested dictionary that maps state -> value
	V = defaultdict(float)

	############################
	# YOUR IMPLEMENTATION HERE #
	# loop each episode

	for i_episodes in range(n_episodes):

		# initialize the episode
		old_obs = env.reset()

		# generate empty episode list
		episode = []
		terminate = False

		# loop until episode generation is done
		while(terminate == False):
			
			# select an action
			action = policy(old_obs) # random action

			# return a reward and new state
			new_obs, reward, done, info = env.step(action)

			# append state, action, reward to episode
			episode.append([old_obs,action,reward])

			# update state to new state
			old_obs = new_obs
			terminate = done

		
		Gvalues = defaultdict(float)
		i = len(episode)
		G = 0
		
		# Get total return for all states in episode
		# loop for each step of episode, t = T-1, T-2,...,0
		for [observation, action, reward] in reversed(episode):
			# Compute Return
			G = gamma*G + reward

			# Store return
			Gvalues[i] = G
			i -= 1

		i = 1
		states_visited = []

		for [observation, action, reward] in episode:

			# unless state_t appears in states
			if observation not in states_visited:
				# update return_count
				returns_count[observation] += 1 

				# update return_sum
				returns_sum[observation] += Gvalues[i]

				# calculate average return for this state over all sampled episodes
				V[observation] = returns_sum[observation]/returns_count[observation]

				states_visited.append(observation)
			
			i += 1 
	############################
	return V

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
	"""Selects epsilon-greedy action for supplied state.
	
	Parameters:
	-----------
	Q: dict()
		A dictionary  that maps from state -> action-values,
		where Q[s][a] is the estimated action value corresponding to state s and action a. 
	state: int
		current state
	nA: int
		Number of actions in the environment
	epsilon: float
		The probability to select a random action, range between 0 and 1
	
	Returns:
	--------
	action: int
		action based current state
	Hints:
	------
	With probability (1 − epsilon) choose the greedy action.
	With probability epsilon choose an action at random.
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	stateActionProbability = Q[state]
	best_policy = np.argmax(stateActionProbability)
	print(epsilon)
	print(nA)
	policy = np.ones(nA, float)*(epsilon/nA)
	policy[best_policy] = (epsilon/nA) + 1 - epsilon

	action = np.random.choice(np.arange(len(stateActionProbability)), p = policy)

	############################
	return action

def mc_control_epsilon_greedy(env, n_episodes, gamma = 1.0, epsilon = 0.1):
	"""Monte Carlo control with exploring starts. 
		Find an optimal epsilon-greedy policy.
	
	Parameters:
	-----------
	env: function
		OpenAI gym environment
	n_episodes: int
		Number of episodes to sample
	gamma: float
		Gamma discount factor
	epsilon: float
		The probability to select a random action, range between 0 and 1
	Returns:
	--------
	Q: dict()
		A dictionary  that maps from state -> action-values,
		where Q[s][a] is the estimated action value corresponding to state s and action a.
	Hint:
	-----
	You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
	and episode must > 0.    
	"""
	#{

	returns_sum = defaultdict(float)
	returns_count = defaultdict(float)

	# a nested dictionary that maps state -> (action -> action-value)
	# e.g. Q[state] = np.darrary(nA)
	Q = defaultdict(lambda: np.zeros(env.action_space.n))
	
	############################
	# YOUR IMPLEMENTATION HERE #
	for i_episodes in range(n_episodes):

		# define decaying epsilon
		if epsilon <= 0.05:
			epsilon = 0.05 # Set epsilon to prevent it becoming -ve
		else:
			epsilon = epsilon - (0.1/n_episodes)

		# initialize the episode
		old_obs = env.reset()
		terminate = False

		# generate empty episode list
		episode = []

		# loop until one episode generation is done
		while(terminate == False):

			# get an action from epsilon greedy policy
			action = epsilon_greedy(Q, old_obs, env.action_space.n, epsilon)
			
			# return a reward and new state
			new_obs, reward, done, info = env.step(action)

			# append state, action, reward to episode
			episode.append([old_obs, action, reward])

			# update state to new state
			old_obs = new_obs
			terminate = done
			
		G = 0
		Gvalues = defaultdict(float) #dictionary for expected return for all states action pair
		i = len(episode)

		# loop for each step of episode, t = T-1, T-2, ...,0
		for [observation, action, reward] in reversed(episode):
			actionState = (observation,action) 
			
			# compute G
			G = gamma*G + reward

			# Store total return for all states
			Gvalues[i] = G
			i -= 1 

		visited_actionState = []
		i = 1

		for [observation, action, reward] in episode:
			actionState = (observation,action) 
			# unless the pair state_t, action_t appears in <state action> pair list
			if actionState not in visited_actionState:

				# update return_count
				returns_count[actionState] += 1

				# update return_sum
				returns_sum[actionState] += Gvalues[i]

				# calculate average return for this state over all sampled episodes
				Q[observation][action] = returns_sum[actionState]/returns_count[actionState]

				visited_actionState.append(actionState)
			i += 1 	
		
	return Q