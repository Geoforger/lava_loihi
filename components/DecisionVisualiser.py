import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

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
        net_out_shape (tuple):
            Output shape of network
        window (VisualiserWindow class):
            Main thread tkinter window to display to
        frequency (int, optional):
            Frequency of window update. Default=10
        save_output (string, optional)
            Save path for output plot on sim end. Default=None
    Returns:
        None
    """

    def __init__(
        self, 
        net_out_shape,
        window,
        frequency=10,
        save_output=None
    ) -> None:

        self.in_shape = (1,)
        self.net_out_shape = net_out_shape
        self.window = window
        self.frequency = frequency
        self.bar_container = None
        self.background = None
        self.save_output = save_output

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
            frequency=self.frequency,
            save_output=self.save_output,
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

    def __init__(self, proc_params) -> None:
        super().__init__(proc_params)
        self.in_shape = proc_params["in_shape"]
        self.net_out_shape = proc_params["net_out_shape"]
        self.frequency = proc_params["frequency"]
        self.save_output = proc_params["save_output"] 
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
        self.ax.set_ylim(0, 100)
        self.ax.set_yticks(np.arange(0,150,50))
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

        self.classification_label.config(text=f"Classification: N/A")
        self.confidence_label.config(text=f"Confidence: {0.0}")

    def update_plot(self):
        bar_height = self.acc

        # Update bar heights
        for bar, height in zip(self.bar_container, bar_height):
            bar.set_height(height)

        self.canvas.restore_region(self.background)

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
            if self.time_step % self.frequency == 0:
                self.window.root.after(0, self.update_plot)
                self.window.root.update()

        # Recieve decision for next timestep
        self.decision = self.a_in.recv()

    def _pause(self) -> None:
        """Pause was called by the runtime"""
        super()._pause()

    def _stop(self) -> None:
        """
        Stop was called by the runtime.
        """
        if self.save_output is not None:
            self.fig.savefig(self.save_output, dpi=300, bbox_inches="tight")

        self.window.root.quit()
        super()._stop()


class VisualiserWindow:
    def __init__(self, root):
        self.root = root

        self.fig = Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.classification_label = tk.Label(self.root, text="Classification: N/A")
        self.classification_label.pack(side=tk.LEFT)
        self.confidence_label = tk.Label(self.root, text="Confidence: 0.0")
        self.confidence_label.pack(side=tk.RIGHT)
