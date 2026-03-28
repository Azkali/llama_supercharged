# a slot contains basic information.
import json

# these classes are in the wrong order because python reads files top-to-bottom.
# (though you can put Slot after Slotter since the function contents are executed later.)
class Slot:
    def __init__(self, id):
        self.id = id
        self.model = None
        self.json = None
        self.jtype = None

class Slotter:
    def __init__(self):
        self.ncounter = 0 # internal id. will break program after two billion slots!
        self.slots = []

    # should new_slot and del_slot be "private" functions?
    def new_slot(self, number):
        idx = self._find_slot(number)
        if idx is None:
            #self.slots.append({"id": self.ncounter, "model": None, "json": None, "jtype": None})
            self.slots.append(Slot(self.ncounter))
            self.ncounter += 1
            return self.ncounter - 1
        else:
            return -1

    def del_slot(self, number):
        idx = self._find_slot(number)
        if idx is None:
            return -1
        else:
            self.slots.pop(idx)
            return 0

    def list_slots(self):
        return "unimplemented."

    def _find_slot(self, number):
        idxs = [i for i, s in enumerate(self.slots) if s.id == number]
        if len(idxs) == 0:
            return None
        elif len(idxs) > 1: # literal impossible case.
            print("you smashed the stack.")
            exit(1234)
        else:
            return idxs[0]

# SlotMan directly executes data; it does not store instructions.
class SlotMan(Slotter):
    def parse_SLACT(self, data):
        SLACT = json.load(data)

        slot = SLACT.get("slot")
        action = SLACT.get("action")

        if action == "create": return self.new_slot(slot)
        if action == "delete": return self.del_slot(slot)
        else: return -1
