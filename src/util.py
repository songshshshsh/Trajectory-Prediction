class Person(Object):
	def __init__(agent, past_traj, intention_getter, pos,):
		self.agent = agent
		self.neighbor = initial_neighbor_from_set(pos)
		self.pos = pos
		self.intention = intention_getter(past_traj)

	def learn():
		# a network predicting traj from current neighbor and pos and intention