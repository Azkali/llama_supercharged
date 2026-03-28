from supercharger.system.MEMCON.slotter import Slotter

class Runnables(Slotter):

    # allow-safety section.

    def __call__(self):
        return [
            "load",
            "unload",
            "wipe",
            "mcheck",
            "reserve",
            "deserve",
            "init_full",
            "init_half",
            "execute",
            "crash",
        ]

    # memory section.

    def load(self, idx):
        return self.slots[idx].model.load()

    def unload(self, idx):
        return self.slots[idx].model.unload()

    def wipe(self, idx):
        return "i don't wipe. slots! i don't wipe slots!"
        #kill this guy.
        #error0 = self.slots[idx].model.unload() # could error if model wasn't even loaded.
        #self.slots[idx] = Slot(self.slots[idx].id)
        #return error0 # should error before slot reset? dangerous.

    def mcheck(self, idx):
        return "like four bytes at least." # unimplemented.

    def reserve(self, idx):
        return self.mcheck(0)
    
    def deserve(self, idx):
        return self.mcheck(0)

    # trigger section.

    def init_full(self, idx):
        return "unimplemented."

    def init_half(self, idx):
        return "unimplemented."

    def execute(self, idx):
        return "unimplemented."

    def crash(self, idx):
        return "unimplemented."
