import mesa.visualization
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer

from agents import Firm, Household
from model import MacroModel
from matplotlib.figure import Figure


class StepsTextElement(TextElement):
    def render(self, model):
        return f"Steps: {model.schedule.steps}"


# dictionary of user settable parameters - these map to the model __init__ parameters
model_params = {
    "init_households": mesa.visualization.Slider(
        "Households", value=1200, min_value=1, max_value=3000, description="Number of households"
    )
}

model = MacroModel()

chart_hh_liquidity = ChartModule([{"Label": "Household Liquidity", "Color": "red"}],
                                 data_collector_name='datacollector')
chart_firm_liquidity = ChartModule([{"Label": "Firm Liquidity", "Color": "Purple"}],
                                   data_collector_name='datacollector')
chart_price = ChartModule([{"Label": "Price", "Color": "Blue"}], data_collector_name='datacollector')
chart_wage = ChartModule([{"Label": "Wage", "Color": "Green"}], data_collector_name='datacollector')
chart_positions = ChartModule([{"Label": "Open Positions", "Color": "Orange"}], data_collector_name='datacollector')
chart_employment = ChartModule([{"Label": "Employment", "Color": "Yellow"}], data_collector_name='datacollector')
text_element = StepsTextElement()

server = ModularServer(
    MacroModel,
    [chart_hh_liquidity, chart_firm_liquidity, chart_price,
     chart_wage, chart_positions, chart_employment, text_element],
    name="Household Model",
    model_params=model_params
)
