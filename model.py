"""
This model is built following Lengnick(2013) paper on
Journal of Economic Behavior & Organization 86 (2013) 102– 120.
"""
import agents
from agents import Firm, Household
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.time import DiscreteEventScheduler
import numpy as np
import random
import pandas as pd
import statistics

# Data collector functions


# Return number of agents
def get_num_agents(model):
    agents = [a for a in model.schedule.agents]
    return len(agents)


# Sum of all agent savings
def get_total_savings(model):
    agent_savings = sum(agent.m for agent in model.schedule.agents if isinstance(agent, Household))


# Returns firm prices
def get_prices(model):
    return [firm.p for firm in model.FI_list]


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
        self.num_typeA = num_typeA
        self.Psi_price = Psi_price
        self.xi = xi
        self.pi = pi
        self.beta = beta
        self.n = n
        self.Household_list = []
        self.Firm_list = []
        self.current_day = 1
        self.total_production = 0
        self.matrix_A_constraints = [[0] * self.num_firms for _ in range(self.num_households)]
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

            f = Firm(i + init_households, self, w, 0, 0, 0, inv * 0.9, inv * 1.1, 0, 0, p,
                     p * 0.9, p * 1.1,delta, Phi_min, Phi_max, phi_min, phi_max, theta, lambda_,
                     gamma, Theta, open_position, 0, 0, 0)
            self.schedule.add(f)
            self.Firm_list.append(f)

        # Create connections (type A: household to firm, type B: firm to household)

        # Type B connections
        for household in self.Household_list:
            newF = random.randint(0, len(self.Firm_list) - 1)
            household.connections_typeB.append(newF)
            household.employed = True
            self.Firm_list[newF].connections_typeB.append(household)

        # Type A connections
        for household in self.Household_list:
            counter = 0

            while counter < num_typeA:
                f = np.random.randint(0, len(self.Firm_list) - 1)
                firm = self.Firm_list[f]

                if firm not in household.connections_typeA:
                    household.connections_typeA.append(firm)
                    firm.connections_typeA.append(household)
                    counter += 1
                else:
                    continue

        # Variables to collect data for
        self.datacollector = DataCollector(
            model_reporters={
                "Household Liquidity": lambda model: sum(
                    agent.m for agent in model.schedule.agents if isinstance(agent, Household)),
                "Price": lambda model: statistics.mean(
                    agent.p for agent in model.schedule.agents if isinstance(agent, Firm)),
                "Wage": lambda model: statistics.mean(
                    agent.w for agent in model.schedule.agents if isinstance(agent, Firm)),
                "Open Positions": lambda model: sum(
                    agent.open_position for agent in model.schedule.agents if isinstance(agent, Firm)),
                "Firm Liquidity": lambda model: sum(
                    agent.m for agent in model.schedule.agents if isinstance(agent, Firm)),
                "Employment": lambda model: sum(
                    agent.employed for agent in model.schedule.agents if isinstance(agent, Household))
            },
            agent_reporters={}
        )

    # Schedule events
        # Beginning of the month events
    def beginning_month_events(self):
        # Firms assign new wage (equation 5 in main paper)
        for firm in self.Firm_list:
            firm.new_wage()
        for firm in self.Firm_list:
            firm.update_inv_range()
        for firm in self.Firm_list:
            firm.update_price_range()
        for firm in self.Firm_list:
            firm.update_demand_for_labour()
        self.update_type_a_price()
        self.update_type_a_quantity()
        self.update_type_b()
        self.update_households_average_prices()

        # {"agent": "Model", "method": "updateTypeB"},
        # {"agent": "Model", "method": "updateHouseholdsAveragePrices"}

    # Daily events
    def daily_events(self):
        # self.good_market_daily_events()
        for firm in self.Firm_list:
            firm.produce()
        self.quantity_produced()

    # End of the month events
    def end_month_events(self):
        self.firms_pay_wages()
        self.firms_pay_profits()

    def monthly_events(self):

        self.current_day = 1

        while self.current_day <= 21:
            # Schedule beginning of the month events
            if self.current_day == 1:
                self.beginning_month_events()
            # Schedule daily events
            if self.current_day >= 1 or self.current_day <= 21:
                self.daily_events()
            # Schedule end of month events
            if self.current_day == 21:
                self.end_month_events()

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

    def update_type_a_price(self):
        for h in self.Household_list:
            f_index = random.randint(0, self.num_typeA - 1)
            f = h.connections_typeA[f_index]
            random_firm_index = random.randint(0, len(self.Firm_list) - 1)
            f_new = self.Firm_list[random_firm_index]

            # Find a current firm to which h is not connected
            while f_new in h.connections_typeA:
                new_index = random.randint(0, len(self.Firm_list) - 1)
                f_new = self.Firm_list[new_index]

            rnd = random.random()
            price_f = f.p
            price_f_new = f_new.p

            if rnd < self.Psi_price and (price_f - price_f_new) / price_f >= self.xi:
                h.connections_typeA.remove(f)
                h.connections_typeA.append(f_new)
                f.connections_typeA.remove(h)
                f_new.connections_typeA.append(h)

    def update_type_a_quantity(self):
        print(len(self.matrix_A_constraints))
        for h in self.Household_list:
            tot_constraint = 0
            constraints = [0] * self.num_typeA

            for f in self.Firm_list:
                self.matrix_A_constraints[h.unique_id][f.unique_id] = 10

            # # Calculate total constraint and create constraints list
            # for f_index in range(self.num_typeA):
            #     index = h.connections_typeA[f_index]

            #     constraint = self.matrix_A_constraints[index]
        print(self.matrix_A_constraints)

    def update_type_b(self):
        prob = 0
        firm = "a"
        for f in self.Firm_list:
            if f.to_fire > 0:
                for i in range(f.to_fire):
                    # Ensure there are connections to fire from
                    if len(f.connections_typeB) > 0:
                        to_fire = random.randint(0, len(f.connections_typeB) - 1)
                        household_id = f.connections_typeB.pop(to_fire)
                        print(household_id)
                        household_id.employed = False

        for h in self.Household_list:
            firm = self.Firm_list[h.connections_typeB[0]]
            if h.employed:
                beta = 1
                if len(h.connections_typeB) < 2:
                    prob = self.pi if h.w <= firm.w else 1
                else:
                    print("Household employed by more than 1 firm")
            else:
                beta = self.beta
                prob = 1

            # Probability check
            if random.random() < prob:
                temp_list = []

                for f in self.Firm_list:
                    temp_list.append(f)

                if h.employed:
                    temp_list.remove(firm)

                for r in range(beta):
                    rnd = random.randint(0, len(temp_list) - 1)
                    selected_firm = temp_list[rnd]

                    if h.employed and selected_firm.open_position > 0:
                        if selected_firm.w > firm.w:
                            # Update type B connections
                            h.connections_typeB.remove(firm)
                            h.connections_typeB.append(selected_firm)
                            firm.connections_typeB.remove(h)
                            selected_firm.connections_typeB.append(h)

                            h.employed = True

                    else:
                        if selected_firm.w >= h.w and selected_firm.open_position > 0:
                            # Update type B connections
                            h.connections_typeB.remove(firm)
                            h.connections_typeB.append(selected_firm)
                            selected_firm.connections_typeB.append(h)

                            h.employed = True
                            if len(firm.connections_typeB) > 0:
                                firm.connections_typeB.remove(h)

                    temp_list.pop(rnd)

    def update_households_average_prices(self):
        for household in self.Household_list:
            p_total = 0
            # Find connected firms and get sum of prices
            for f in household.connections_typeA:
                p_total += f.p
            # Set household price as average of connected firm prices
            household.p = p_total/self.num_typeA

    def good_market_daily_events(self):
        # Create randomised household list
        random_household_list = random.sample(self.Household_list, len(self.Household_list))
        # Iterate randomised household list
        for household in random_household_list:
            temp_list = []
            for f in household.connections_typeA:
                temp_list.append(f)

            purchased_quantity = 0
            household_demands = int(household.c / 21)
            visited_firms = 0

            while household.m > 0 and (purchased_quantity / household_demands) < 0.95 and visited_firms < self.n:
                f = random.randint(0, len(temp_list) - 1)
                firm = temp_list[f]

                firm.d += (household_demands - purchased_quantity)

                # Check if firm has enough goods to satisfy the demand of the household
                transaction_quantity = min(household_demands - purchased_quantity, firm.inv)

                # Goods constraints are updated - UNSURE
                # self.matrix_A_constraints[household.unique_id][
                #     firm.unique_id] = household_demands - purchased_quantity - transaction_quantity

                # Check if the household has enough liquidity
                transaction_quantity = min(transaction_quantity, int(household.m / firm.p))

                # Update firm inventory
                firm.inv -= transaction_quantity

                # Update firm liquidity
                firm.m += (transaction_quantity * firm.p)

                # Update household liquidity
                household.m -= (transaction_quantity * firm.p)

                # Update the all purchased quantity
                purchased_quantity += transaction_quantity

                # Remove firm from the temp list of firms
                temp_list.remove(firm)
                visited_firms += 1

    def quantity_produced(self):
        production = 0
        for firm in self.Firm_list:
            production += firm.daily_produces

        self.total_production = production

    def firms_pay_wages(self):
        for household in self.Household_list:
            firm = self.Firm_list[household.connections_typeB[0]]

            amount_paid = min(firm.w, (firm.m + firm.m_buffer))

            household.m = household.m + amount_paid

            amount_from_m = min(firm.m, amount_paid)
            amount_from_buffer = min((amount_paid - amount_from_m), firm.m_buffer)

            firm.m = firm.m - amount_from_m
            firm.m_buffer = firm.m_buffer - amount_from_buffer

    def firms_pay_profits(self):
        aggregated_profit = 0
        aggregated_household_wealth = 0

        # Calculate aggregated profits and rest firm liquidity
        for firm in self.Firm_list:
            aggregated_profit += firm.m
            firm.m = 0

        # Calculate aggregated household wealth
        for household in self.Household_list:
            aggregated_household_wealth += household.m

        # Distribute profits among households
        for household in self.Household_list:
            if aggregated_household_wealth != 0:
                dividend = aggregated_profit * (household.m / aggregated_household_wealth)
                household.dividend = dividend
                household.m += dividend

    def step(self):
        self.schedule.step()
        self.monthly_events()
        # Collect data
        self.datacollector.collect(self)
        self.export_connections('connect.xlsx')
