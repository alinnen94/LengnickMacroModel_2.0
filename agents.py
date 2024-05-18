from mesa import Agent
import numpy as np
import random
import math


# Firm class
class Firm(Agent):
    def __init__(self, id, model, w, m, m_buffer, inv, inv_min, inv_max, d, mc, p,
                 p_min, p_max, delta, Phi_min, Phi_max,phi_min,phi_max,
                 theta, lambda_, gamma, Theta, open_position,
                 to_fire, num_months_with_open_positions, daily_produces):

        # initialise the parent class with required parameters
        super().__init__(id, model)
        self.w = w
        self.m = m
        self.m_buffer = m_buffer
        self.inv = inv
        self.inv_min = inv_min
        self.inv_max = inv_max
        self.d = d
        self.mc = mc
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
        self.to_fire = to_fire
        self.num_months_with_open_positions = num_months_with_open_positions
        self.daily_produces = daily_produces
        self.connections_typeA = []
        self.connections_typeB = []

    # following eq(13-14) in main paper
    def produce(self):
        self.daily_produces = self.lambda_ * len(self.connections_typeB)
        self.inv += self.lambda_ * len(self.connections_typeB)

    # following eq(5) in main paper
    def new_wage(self):
        random_factor = np.random.uniform(-self.delta, self.delta)
        self.w = self.w * (1 + random_factor)

    # following eq(6) and (7) in main paper
    def update_inv_range(self):
        self.inv_max = int(self.Phi_max * self.d)
        self.inv_min = int(self.Phi_min * self.d)

    def update_demand_for_labour(self):
        if self.open_position > 0:
            self.num_months_with_open_positions += 1
        else:
            self.w = self.w * 0.9  # Reduce the wage if all open positions are filled

        if self.num_months_with_open_positions == self.gamma:
            self.w = self.w * 1.1  # Reduce the wage by 10% if there were open positions for gamma months

        if self.inv < self.inv_min:
            self.open_position += 1  # Hire one worker

            # Consider increasing price
            if self.p < self.p_max and random.random() < self.theta:
                self.increase_price()

        elif self.inv > self.inv_max:
            self.to_fire += 1  # Fire one worker

            # Consider decreasing price
            if self.p > self.p_min and random.random() < self.theta:
                self.decrease_price()

    # following eq(8) and (9) in main paper
    def update_price_range(self):
        self.p_max = int(self.phi_max * self.mc)
        self.p_min = int(self.phi_min * self.mc)

    # following eq(10)
    def increase_price(self):
        self.p = self.p * (1 + random.uniform(0, self.theta))
        print("increase price: " + self.p)

    def decrease_price(self):
        self.p = self.p * (1 - random.uniform(0, self.theta))
        print("decrease price: " + self.p)

    def get_to_fire(self):
        if len(self.connections_typeB) > 0:
            return self.to_fire


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
