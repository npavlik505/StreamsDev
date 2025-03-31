import numpy as np
from config import Config, JetMethod, Jet
import libstreams as streams
from abc import ABC, abstractmethod
from typing import Optional, Dict
import utils
import math

# the equation of the polynomial for the jet actuation in coordinates local to the jet
# actuator. This means that the jet actuator starts at x = 0 and ends at x = slot_end
# 
# this must be recomputed at every amplitude change
class Polynomial():
    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x_idx: int) -> float:
        return self.a * x_idx**2  + \
                self.b * x_idx + \
                self.c

# helper class to recompute the polynomial of the jet actuator
class PolynomialFactory():
    def __init__(self, vertex_x: float, slot_start: float, slot_end: float):
        self.slot_start = slot_start
        self.slot_end = slot_end
        self.vertex_x  = vertex_x

    def poly(self, amplitude: float) -> Polynomial:
        # see streams section of lab documentation for the derivation of this
        #
        # in essence, it is solving for the coefficients a,b,c of the polynomial
        # y = ax^2 + bx +c 
        # using the fact that
        # y(jet start) = 0
        # y(jet end) = 0
        # y((jet start + jet_end) / 2) = amplitude
        a = amplitude/(self.vertex_x**2 - self.vertex_x*self.slot_end- (self.vertex_x - self.slot_end)*self.slot_start)
        b = -(amplitude*self.slot_end+ amplitude*self.slot_start)/(self.vertex_x**2 - self.vertex_x*self.slot_end- (self.vertex_x - self.slot_end)*self.slot_start)
        c = amplitude*self.slot_end*self.slot_start/(self.vertex_x**2 - self.vertex_x*self.slot_end- (self.vertex_x - self.slot_end)*self.slot_start)

        return Polynomial(a, b, c)

class JetActuator():
    def __init__(self, rank: int, config: Config, slot_start: int, slot_end: int):
        self.rank = rank
        self.config = config

        vertex_x = (slot_start + slot_end) / 2
        self.factory = PolynomialFactory(vertex_x, slot_start, slot_end)

        def get_int_from_wrapper(wrapper):
            val = np.empty(1, dtype=np.int32)
            wrapper(val)
            return val[0]

        self.local_slot_start_x = get_int_from_wrapper(streams.wrap_get_x_start_slot)
        self.local_slot_nx = get_int_from_wrapper(streams.wrap_get_nx_slot)
        self.local_slot_nz = get_int_from_wrapper(streams.wrap_get_nz_slot)

        self.has_slot = self.local_slot_start_x != -1

        if self.has_slot:
            arr = np.empty((self.local_slot_nx, self.local_slot_nz), dtype=np.float64)
            streams.wrap_get_blowing_bc_slot_velocity(arr)
            self.bc_velocity = arr

    def set_amplitude(self, amplitude: float):
        # WARNING: copying to GPU and copying to CPU must happen on ALL mpi procs
        streams.wrap_copy_blowing_bc_to_cpu()

        if not self.has_slot:
            streams.wrap_copy_blowing_bc_to_gpu()
            return None

        # calculate the equation of the polynomial that the velocity of the jet
        # actuator will have with this amplitude
        poly = self.factory.poly(amplitude)

        for idx in range(self.local_slot_nx):
            local_x = self.local_slot_start_x + idx
            global_x = self.config.local_to_global_x(local_x, self.rank)

            velo = poly.evaluate(global_x)
            self.bc_velocity[idx, 0:self.local_slot_nz] = velo

        # copy everything back to the GPU
        streams.wrap_copy_blowing_bc_to_gpu()
        return None


class AbstractActuator(ABC):
    @abstractmethod
    # returns the amplitude of the jet that was used
    def step_actuator(self, time: float) -> float:
        pass

class NoActuation(AbstractActuator):
    def __init__(self):
        utils.hprint("skipping initialization of jet actuator")
        pass

    # returns the amplitude of the jet that was used
    def step_actuator(self, _:float) -> float:
        return 0.

class ConstantActuator(AbstractActuator):
    def __init__(self, amplitude: float, slot_start: int, slot_end: int, rank: int, config: Config):
        utils.hprint("initializing a constant velocity actuator")

        self.slot_start = slot_start
        self.slot_end = slot_end
        self.amplitude = amplitude

        self.actuator = JetActuator(rank, config, slot_start, slot_end)

    # returns the amplitude of the jet that was used
    def step_actuator(self, _: float) -> float:
        self.actuator.set_amplitude(self.amplitude)
        return self.amplitude

class SinusoidalActuator(AbstractActuator):
    def __init__(self, amplitude: float, slot_start: int, slot_end: int, rank: int, config: Config, angular_frequency:float ):
        utils.hprint("initializing a constant velocity actuator")

        self.slot_start = slot_start
        self.slot_end = slot_end
        self.amplitude = amplitude

        self.actuator = JetActuator(rank, config, slot_start, slot_end)
        self.angular_frequency = angular_frequency

    # returns the amplitude of the jet that was used
    def step_actuator(self, time: float) -> float:
        adjusted_amplitude = math.sin(self.angular_frequency * time)

        self.actuator.set_amplitude(adjusted_amplitude)

        return adjusted_amplitude

def init_actuator(rank: int, config: Config) -> AbstractActuator:
    jet_config = config.jet

    if jet_config.jet_method == JetMethod.none:
        return NoActuation()
    elif jet_config.jet_method == JetMethod.constant:
        print(jet_config.extra_json)
        # these should be guaranteed to exist in the additional json information
        # so we can essentially ignore the errors that we have here
        slot_start = jet_config.extra_json["slot_start"]
        slot_end = jet_config.extra_json["slot_end"]
        amplitude = jet_config.extra_json["amplitude"]

        return ConstantActuator(amplitude, slot_start, slot_end, rank, config);
    elif jet_config.jet_method == JetMethod.sinusoidal:
        print(jet_config.extra_json)
        # these should be guaranteed to exist in the additional json information
        # so we can essentially ignore the errors that we have here
        slot_start = jet_config.extra_json["slot_start"]
        slot_end = jet_config.extra_json["slot_end"]
        amplitude = jet_config.extra_json["amplitude"]
        angular_frequency = jet_config.extra_json["angular_frequency"]

        return SinusoidalActuator(amplitude, slot_start, slot_end, rank, config, angular_frequency);
    elif jet_config.jet_method == JetMethod.adaptive:
        exit()
    else:
        exit()
