# config/graph_colors.py

class _GraphColors:
    def __init__(self):
        self.green = "#55b8b1"
        self.purple = "#466cd4"
        self.blue = "#3ca4ff"
        self.yellow = "#eda205"
        self.orange = "#e35809"
        self.pink = "#b138a0"
        self.bright_blue = "#82ebfd"

        # Aliases
        self.final_results = self.green
        self.registration = self.purple
        self.gps = self.orange
        self.slam = self.pink
        self.fitness = self.pink
        self.inlier_rmse = self.yellow
        self.dots = self.bright_blue

    def __getitem__(self, item):
        return getattr(self, item)

graph_colors = _GraphColors()