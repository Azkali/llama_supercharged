from supercharger.system.MEMCON.slotter import Slotter
from supercharger.system.MEMCON.runnables import Runnables
import json

class MEMCON(Slotter):
    def __call__(self, blueJSON):
        self._parse(blueJSON)

    def _parse(self, blueJSON):
        data = json.load(blueJSON)

        if isinstance(data.slot, int) and not isinstance(data.slot, bool):
            print("slot number detected.")
        else:
            print("slot number not detected. assuming SLACT.")

        if instruction not in Runnables():
            print("stop.")
            exit()

    def _check_slot(self, slot):
        idx = self._find_slot(slot)
        if idx is None: return "this slot does not exist."
