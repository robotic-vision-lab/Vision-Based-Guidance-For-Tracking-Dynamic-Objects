import ctypes

class HighPrecisionClock:
    """High precision clock for time resolution in microseconds
    """

    def __init__(self):
        self.micro_timestamp = self.micros()

    def tick(self, framerate):
        """Implements appropriate delay given a framerate.

        Args:
            framerate (float): Desired framerate

        Returns:
            float: time elapsed
        """
        self.delay_microseconds(1000000 // framerate )

        _new_micro_ts = self.micros()
        self.time_diff = _new_micro_ts - self.micro_timestamp
        self.micro_timestamp = _new_micro_ts

        return self.time_diff

    @staticmethod
    def micros():
        """return timestamp in microseconds"""
        tics = ctypes.c_int64()
        freq = ctypes.c_int64()

        # get ticks on the internal ~3.2GHz QPC clock
        ctypes.windll.Kernel32.QueryPerformanceCounter(ctypes.byref(tics))
        # get the actual freq. of the internal ~3.2GHz QPC clock
        ctypes.windll.Kernel32.QueryPerformanceFrequency(ctypes.byref(freq))

        t_us = tics.value * 1e6 / freq.value
        return t_us

    def delay_microseconds(self, delay_us):
        """delay for delay_us microseconds (us)"""
        t_start = self.micros()
        while (self.micros() - t_start < delay_us):
            pass    # do nothing
        return
