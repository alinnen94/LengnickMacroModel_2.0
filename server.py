import mesa.visualization
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer

from agents import Firm, Household
from model import MacroModel
from matplotlib.figure import Figure


class TitleTextElement(TextElement):
    def __init__(self, title):
        self.title = title

    def render(self, model):
        return self.title


class SpacerTextElement(TextElement):
    def render(self, model):
        return "<br>"


# dictionary of user settable parameters - these map to the model __init__ parameters
model_params = {
    "init_households": mesa.visualization.Slider(
        "Households", value=1000, min_value=1, max_value=3000, description="Number of Households"
    )
}

model = MacroModel()

chart_hh_liquidity_title = TitleTextElement("Household Liquidity")
chart_hh_liquidity = ChartModule([{"Label": "Household Liquidity", "Color": "red"}],
                                 data_collector_name='datacollector')

chart_firm_liquidity_title = TitleTextElement("Firm Liquidity")
chart_firm_liquidity = ChartModule([{"Label": "Firm Liquidity", "Color": "Purple"}],
                                   data_collector_name='datacollector')

chart_price_title = TitleTextElement("Price Trends")
chart_price = ChartModule([{"Label": "Min Price", "Color": "Green"},
                           {"Label": "Mean Price", "Color": "Red"},
                           {"Label": "Max Price", "Color": "Blue"}],
                          data_collector_name='datacollector')

chart_wage_title = TitleTextElement("Wage Trends")
chart_wage = ChartModule([{"Label": "Min Wage", "Color": "Green"},
                          {"Label": "Mean Wage", "Color": "Red"},
                          {"Label": "Max Wage", "Color": "Blue"}],
                         data_collector_name='datacollector')

chart_positions_title = TitleTextElement("Open Positions")
chart_positions = ChartModule([{"Label": "Min Open Positions", "Color": "Green"},
                               {"Label": "Mean Open Positions", "Color": "Red"},
                               {"Label": "Max Open Positions", "Color": "Blue"}],
                              data_collector_name='datacollector')

chart_employment_title = TitleTextElement("Employment Rate")
chart_employment = ChartModule([{"Label": "Employment Rate", "Color": "Yellow"}],
                               data_collector_name='datacollector')

chart_dividend_title = TitleTextElement("Dividend Trends")
chart_dividend = ChartModule([{"Label": "Min Dividend", "Color": "Green"},
                              {"Label": "Mean Dividend", "Color": "Red"},
                              {"Label": "Max Dividend", "Color": "Blue"}],
                             data_collector_name='datacollector')

chart_production_title = TitleTextElement("Total Production")
chart_production = ChartModule([{"Label": "Total Production", "Color": "red"}],
                               data_collector_name='datacollector')
spacer = SpacerTextElement()

server = ModularServer(
    MacroModel,
    [chart_price_title, chart_price, spacer,
     chart_wage_title, chart_wage, spacer,
     chart_positions_title, chart_positions, spacer,
     chart_employment_title, chart_employment, spacer,
     chart_dividend_title, chart_dividend, spacer,
     chart_production_title, chart_production, spacer,
     chart_hh_liquidity_title, chart_hh_liquidity, spacer,
     chart_firm_liquidity_title, chart_firm_liquidity, spacer],
    name="Household Model",
    model_params=model_params
)
