import mesa.visualization
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer

from agents import Firm, Household
from model import MacroModel
from matplotlib.figure import Figure


# Green
RICH_COLOR = "#46FF33"
# Red
POOR_COLOR = "#FF3C33"
# Blue
MID_COLOR = "#3349FF"


# def savings_histogram(model):
#     fig = Figure()
#     ax = fig.subplots()
#     savings = sum(agent.m for agent in model.schedule.agents if isinstance(agent, Household))
#     ax.hist(savings, bins=10)


def household_portrayal(agent):
    if agent is None:
        return

    portrayal = {}

    # update portrayal characteristics for each Person object
    if isinstance(agent, Household):
        portrayal["Shape"] = "circle"
        portrayal["r"] = 0.5
        portrayal["Layer"] = 0
        portrayal["Filled"] = "true"

        color = MID_COLOR

        # set agent color based on savings and loans
        if agent.savings > agent.model.rich_threshold:
            color = RICH_COLOR
        if agent.savings < 10 and agent.loans < 10:
            color = MID_COLOR
        if agent.loans > 10:
            color = POOR_COLOR

        portrayal["Color"] = color

    savings = sum(agent.m for agent in model.schedule.agents if isinstance(agent, Household))

    return portrayal


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

chart_element = ChartModule([{"Label": "Total Household Savings", "Color": "red"}], data_collector_name='datacollector')

text_element = StepsTextElement()

server = ModularServer(
    MacroModel,
    [chart_element, text_element],
    name="Household Model",
    model_params=model_params
)
