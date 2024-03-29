"""
This model is built following Lengnick(2013) paper on
Journal of Economic Behavior & Organization 86 (2013) 102– 120.
"""
import agents
from agents import Firm, Household
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation, BaseScheduler, DiscreteEventScheduler
import numpy as np
import random
import pandas as pd

# Data collector functions


# Return number of agents
def get_num_agents(model):
    agents = [a for a in model.schedule.agents]
    return len(agents)


# Sum of all agent savings
def get_total_savings(model):
    agent_savings = sum(agent.m for agent in model.schedule.agents if isinstance(agent, Household))


# Model class
class MacroModel(Model):

    # Default parameters as of pg. 110
    def __init__(self,
                 seed=1,
                 init_households=1000,
                 F=100,
                 num_typeA=7,
                 delta=0.019,
                 phi_max=1,
                 phi_min=0.25,
                 theta=0.02,
                 Phi_max=1.15,
                 Phi_min=1.025,
                 alpha=0.9,
                 Psi_price=0.25,
                 Psi_quant=0.25,
                 xi=0.01,
                 beta=50,
                 pi=0.1,
                 n=7,
                 gamma=24,
                 lambda_=3,
                 Theta=0.75):

        super().__init__()
        self.schedule = DiscreteEventScheduler(self, time_step=21)
        self.num_households = init_households
        self.num_firms = F
        self.Household_list = []
        self.Firm_list = []
        self.current_day = 1
        # self.schedule_events()
        # self.schedule = RandomActivation(self)

        # Initialise random number generator. The same seed produces the same random number sequence
        np.random.seed(seed)

        # Create household agents for the model according to number set by user
        for i in range(self.num_households):
            # Set reservation wage, liquidity and consumption
            w = np.random.normal(loc=1, scale=0.2)
            m = np.random.normal(loc=1, scale=0.2)
            c = np.random.randint(low=21, high=105)
            dividend = 0
            employed = False
            P = 0

            h = Household(i, self, w, m, c, num_typeA, dividend, employed, P, alpha)
            self.schedule.add(h)
            self.Household_list.append(h)

        # Create firm agents for the model according to number set by user
        for i in range(self.num_firms):
            # Set offered wage, inventory value and price level
            w = np.random.normal(loc=1, scale=0.2)
            inv = np.random.randint(low=0, high=10)
            p = np.random.normal(loc=0.1, scale= 0.2)
            open_position = 0

            f = Firm(i + init_households, self, w, 0, inv * 0.9, inv * 1.1, p, p * 0.9, p * 1.1,
                     delta, Phi_min, Phi_max, phi_min, phi_max, theta,
                     lambda_, gamma, Theta, open_position)
            self.schedule.add(f)
            self.Firm_list.append(f)

        # Create connections (type A: household to firm, type B: firm to household)

        # Type B connections
        for household in self.Household_list:
            newF = random.randint(0, len(self.Firm_list) - 1)
            household.connections_typeB.append(newF)
            self.Firm_list[newF].connections_typeB.append(household)

        # Type A connections
        for household in self.Household_list:
            counter = 0

            while counter < num_typeA:
                f = np.random.randint(0, len(self.Firm_list) - 1)
                firm = self.Firm_list[f]

                if firm not in household.connections_typeA:
                    household.connections_typeA.append(firm)
                    counter += 1
                else:
                    continue

        # Variables to collect data for
        self.datacollector = DataCollector(
            model_reporters={
                "Number of agents": get_num_agents,
                "Total savings": get_total_savings,
                "Total Household Savings": lambda model: sum(
                    agent.m for agent in model.schedule.agents if isinstance(agent, Household))
            },
            agent_reporters={}
        )

    # Schedule events
        # Beginning of the month events
    def beginning_month_events(self):
        # Firms assign new wage (equation 5 in main paper)
        for firm in self.Firm_list:
            firm.new_wage()
        # {"agent": "Firm", "method": "updateInvRange"},
        # {"agent": "Firm", "method": "updatePriceRange"},
        # {"agent": "Firm", "method": "updateDemandForLabour"},
        # {"agent": "Model", "method": "updateTypeA_Price"},
        # {"agent": "Model", "method": "updateTypeA_Quantity"},
        # {"agent": "Model", "method": "updateTypeB"},
        # {"agent": "Model", "method": "updateHouseholdsAveragePrices"}

        # Daily events
        self.daily_events = [
            {"agent": "Model", "method": "GoodMarketDailyEvents"},
            {"agent": "Firm", "method": "produce"},
            {"agent": "Model", "method": "QuantityProduced"}
        ]

        # End of the month events
        self.end_month_events = [
            {"agent": "Model", "method": "firmsPayWages"},
            {"agent": "Model", "method": "firmsPayProfits"}
        ]

    def monthly_events(self):

        self.current_day = 1

        while self.current_day <= 21:
            # Schedule beginning of the month events
            if self.current_day == 1:
                self.beginning_month_events()

            # Schedule daily events
            if self.current_day >= 1 or self.current_day <= 21:
                # print(self.current_day)
                a = 1

            if self.current_day == 20:
                a = 2

            self.current_day += 1

    def export_connections(self, filename):
        type_a_connections = []
        type_b_connections = []

        # Iterate through agents and collect connections
        for agent in self.schedule.agents:
            if isinstance(agent, Household):
                for firm in agent.connections_typeA:
                    type_a_connections.append((agent.unique_id, firm.unique_id))
            if isinstance(agent, Firm):
                for household in agent.connections_typeB:
                    type_b_connections.append((agent.unique_id, household.unique_id))

        # Convert lists to pandas DataFrames
        df_type_a = pd.DataFrame(type_a_connections, columns=['Household', 'Firm'])
        df_type_b = pd.DataFrame(type_b_connections, columns=['Firm', 'Household'])

        # Export DataFrames to Excel file
        with pd.ExcelWriter(filename) as writer:
            df_type_a.to_excel(writer, sheet_name='Type_A_Connections', index=False)
            df_type_b.to_excel(writer, sheet_name='Type_B_Connections', index=False)

    def get_random(self):
        return self.random

    def get_household_savings(self):
        for household in self.Household_list:
            print(f"Household {household.unique_id}: Savings = {household.m}")

    def step(self):
        self.schedule.step()
        self.monthly_events()
        # Collect data
        self.datacollector.collect(self)
        self.export_connections('connect.xlsx')
