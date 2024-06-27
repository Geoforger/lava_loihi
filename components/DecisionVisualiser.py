import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.py.ports import PyInPort, PyRefPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.process.ports.ports import InPort, RefPort
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.resources import CPU
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol


class DecisionVisualiser(AbstractProcess):
    """
    Process that visualises the output of our decision making algorithm

    Parameters:
    -----------
        window_name (string, optional):
            Name of the CV window. Defaults 'Decision Visualiser'
    Returns:
        None
    """

    def __init__(
        self, 
        net_out_shape,
        window
    ) -> None:

        self.in_shape = (1,)
        self.net_out_shape = net_out_shape
        self.window = window
        self.bar_container = None
        self.background = None

        self.a_in = InPort(shape=self.in_shape)
        self.conf_in = RefPort((1,))
        self.acc_in = RefPort(self.net_out_shape)

        # Values to take from Var ports
        self.acc = None
        self.conf = None
        self.decision = None

        super().__init__(
            in_shape=self.in_shape,
            net_out_shape=self.net_out_shape,
            window=self.window,
            acc=self.acc,
            conf=self.conf,
            decision=self.decision,
            bar_container=self.bar_container,
            background=self.background
        )


@implements(proc=DecisionVisualiser, protocol=LoihiProtocol)
@requires(CPU)
class PySparseDecisionVisualiserModel(PyLoihiProcessModel):
    a_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
    acc_in: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, np.int64)
    conf_in: PyRefPort = LavaPyType(PyRefPort.VEC_DENSE, float)

    # acc: np.ndarray = LavaPyType(np.ndarray, np.int64)
    # conf: np.ndarray = LavaPyType(np.ndarray, float)

    def __init__(self, proc_params) -> None:
        super().__init__(proc_params)
        self.in_shape = proc_params["in_shape"]
        self.net_out_shape = proc_params["net_out_shape"]
        self.acc = proc_params["acc"]
        self.conf = proc_params["conf"]
        self.decision = proc_params["decision"]
        self.window = proc_params["window"]
        self.bar_container = proc_params["bar_container"]
        self.background = proc_params["background"]

        # Reference to the tkinter window elements
        self.fig = self.window.fig
        self.ax = self.window.ax
        self.canvas = self.window.canvas
        self.classification_label = self.window.classification_label
        self.confidence_label = self.window.confidence_label

    def init_plot(self):
        bar_x = np.arange(self.net_out_shape[0])
        bar_height = 0

        self.ax.clear()
        self.bar_container = self.ax.bar(bar_x, bar_height)
        self.ax.set_xlabel("Neuron Idx")
        self.ax.set_ylabel("Spike Count")
        self.ax.set_title("Decision Plotter")
        self.ax.set_ylim(0, 500)
        self.ax.set_yticks(np.arange(0,1050,50))
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        self.classification_label.config(text=f"Classification: N/A")
        self.confidence_label.config(text=f"Confidence: {0.0}")

    # def update_plot(self):
    #     bar_x = np.arange(len(self.acc))
    #     bar_height = self.acc

    #     self.ax.clear()
    #     self.ax.bar(bar_x, bar_height)
    #     self.ax.set_xlabel("Neuron Idx")
    #     self.ax.set_ylabel("Spike Count")
    #     self.ax.set_title("Decision Plotter")
    #     self.canvas.draw()

    #     self.classification_label.config(text=f"Classification: {self.decision}")
    #     self.confidence_label.config(text=f"Confidence: {self.conf[0]:.2f}")

    def update_plot(self):
        bar_height = self.acc

        # Update bar heights
        for bar, height in zip(self.bar_container, bar_height):
            bar.set_height(height)

        self.canvas.restore_region(self.background)  # Restore the background

        # Redraw only the bars and the y-axis
        for bar in self.bar_container:
            self.ax.draw_artist(bar)

        self.canvas.blit(self.ax.bbox)
        self.canvas.flush_events()  # Ensure the canvas updates

        self.background = self.canvas.copy_from_bbox(self.ax.bbox)  # Update the background

        self.classification_label.config(text=f"Classification: {self.decision}")
        self.confidence_label.config(text=f"Confidence: {self.conf[0]:.2f}")

    def post_guard(self):
        return True

    def run_post_mgmt(self):
        # Recieve values for accumulator and conf in post mgmt
        self.acc = self.acc_in.read()
        self.conf = self.conf_in.read()

    def run_spk(self) -> None:
        if self.time_step == 1:
            self.window.root.after(0, self.init_plot)
            self.window.root.update()

        # NOTE: Due to run_post_mgmt, we are 1ts out of sync with network
        if (self.acc is not None) & (self.conf != None):
            # print(f"Vis Accumulator: {self.acc} : {self.time_step}")
            # print(f"Vis Conf: {self.conf} : {self.time_step}")
            if self.time_step % 10 == 0:
                self.window.root.after(0, self.update_plot)
                self.window.root.update()

        # Recieve decision for next timestep
        self.decision = self.a_in.recv()


class VisualiserWindow:
    def __init__(self, root):
        self.root = root
        # self.root.title("Dynamic Bar Plot with Tkinter")

        self.fig = Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.classification_label = tk.Label(self.root, text="Classification: N/A")
        self.classification_label.pack(side=tk.LEFT)
        self.confidence_label = tk.Label(self.root, text="Confidence: 0.0")
        self.confidence_label.pack(side=tk.RIGHT)
