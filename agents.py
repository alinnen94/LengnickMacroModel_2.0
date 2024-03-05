from mesa import Agent
import numpy as np
import random
import math


# Firm class
class Firm(Agent):
    def __init__(self, id, model, w, inv, inv_min, inv_max, p,
                 p_min, p_max, delta, Phi_min, Phi_max,phi_min,
                 phi_max, theta, lambda_, gamma, Theta, open_position):

        # initialise the parent class with required parameters
        super().__init__(id, model)
        self.w = w
        self.inv = inv
        self.inv_min = inv_min
        self.inv_max = inv_max
        self.p = p
        self.p_min = p_min
        self.p_max = p_max
        self.delta = delta
        self.Phi_min = Phi_min
        self.Phi_max = Phi_max
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.theta = theta
        self.lambda_ = lambda_
        self.gamma = gamma
        self.Theta = Theta
        self.open_position = open_position
        self.connections_typeA = []
        self.connections_typeB = []

    # following eq(13-14) in main paper
    def produce(self):
        self.dailyProduces = self.lambda_ * self.getTypeB().size()
        self.inv += self.lambda_ * self.getTypeB().size()

    # following eq(5) in main paper
    def new_wage(self):
        random_factor = np.random.uniform(-self.delta, self.delta)
        self.w = self.w * (1) # + getRnd().getDblFromTo(-self.delta, self.delta)
        print(f"Firm {self.unique_id} has a new wage of {random_factor}")

    # following eq(6) and (7) in main paper
    def updateInvRange(self):
        self.inv_max =  self.Phi_max * self.d
        self.inv_min = self.Phi_min * self.d

    # def updateDemandForLabour(self):
    #     if(self.openPosition > 0):
    #       self.num_months_with_openpositions = b + 1
    #     else:
    #           self.w * = 0.9 # the firm filled all open positions, therefore it reduces the wage
    #     if(self.num_months_with_openpositions == self.gamma):
    #        self.w * = 1.1 # the firm had openpositions for gamma months, therefore it reduces the wage by 10%

    # following eq(8) and (9) in main paper
    def updatePriceRange(self):
        self.p_max = self.phi_max * self.mc
        self.p_min = self.phi_min * self.mc

    def increasePrice(self):
        self.p = self.p # * (1 + Sim.getRnd().getDblFromTo(0, self.theta)
        print("increase price: " + self.p)

    def decreasePrice(self):
        self.p = self.p # (1 - Sim.getRnd().getDblFromTo(0, self.theta)
        print("decrease price: " + self.p)


# Household class
class Household(Agent):
    def __init__(self, id, model, w, m, c, num_typeA, dividend, employed, P, alpha):

        # initialise the parent class with required parameters
        super().__init__(id, model)
        self.w = w
        self.m = m
        self.c = c
        self.num_typeA = num_typeA
        self.dividend = dividend
        self.employed = employed
        self.P = P
        self.alpha = alpha

        self.connections_typeA = []
        self.connections_typeB = []

    def updateConsumption(self):
        self.c = 1 # math.min(self.m / self.P) * Math.exp(self.alpha), self.m / self.P
