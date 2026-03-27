import logging
from types import SimpleNamespace

PREFIX = "SPCHARGER"
FORMATTING = SimpleNamespace(
    BO_FMT = "\x1b[1m",     # bold.
    BL_FMT = "\x1b[5m",     # blink.
    NS_FMT = "\x1b[95m",    # not set.      magenta.
    DE_FMT = "\x1b[34m",    # debug.        darkblue.
    IN_FMT = "\x1b[32m",    # info.         darkgreen.
    WA_FMT = "\x1b[33m",    # warning.      darkyellow.
    ER_FMT = "\x1b[91m",    # error.        red.
    CR_FMT = "\x1b[31m",    # critical.     darkred.
    NO_FMT = "\x1b[0m",     # reset.        nothing.
    FIZZZZ = "fzz...",      # fun.
    MOD_B1 = "┒",           # module begin.     type 1.
    MSG_B1 = "╾┒",          # message begin.    ^
    MSG_G1 = " ╏",          # message go.       ^
    FNC_B1 = "┠╼",          # function begin.   ^
    FNC_G1 = "┃ ",          # function go.      ^
    FIZ_E1 = "┷",           # fizzzz end.       ^
    WIL_B1 = "🬰",           # wild ball.        ^
    WIL_L1 = "═",           # wild line.        ^
    WIL_C1 = "╧",           # wild connector.   ^
    WIL_D1 = "╤",           # wild connector 2. ^
    MOD_B2 = "\\",          # same as above.    type 2.
    MSG_B2 = "-.",          #                   ^
    MSG_G2 = " |",          #                   you get the point.
    FNC_B2 = "|-",
    FNC_G2 = "| ",
    FIZ_E2 = "=",
    WIL_B2 = "#",
    WIL_L2 = "-",
    WIL_C2 = "^",
    WIL_D2 = "v",
)

# the TreeRoots logger (pat. pending).
#
# there are three "fixes":
# - the "prefix";
# - the "fix";
# - and the "suffix".

class TreeRoots(logging.Formatter):
    def __init__(self, ansi = True, clearance = 12, should_fizz = False):
        super().__init__()
        self.last_mod = "this_string_will_never_be_used"
        self.last_fnc = "this_string_will_never_be_used"

        self.BO_FMT = "" if not ansi else FORMATTING.BO_FMT
        self.BL_FMT = "" if not ansi else FORMATTING.BL_FMT
        self.NS_FMT = "" if not ansi else FORMATTING.NS_FMT + FORMATTING.BO_FMT # will rarely even happen.
        self.DE_FMT = "" if not ansi else FORMATTING.DE_FMT + FORMATTING.BO_FMT
        self.IN_FMT = "" if not ansi else FORMATTING.IN_FMT + FORMATTING.BO_FMT
        self.WA_FMT = "" if not ansi else FORMATTING.WA_FMT + FORMATTING.BO_FMT
        self.ER_FMT = "" if not ansi else FORMATTING.ER_FMT + FORMATTING.BO_FMT
        self.CR_FMT = "" if not ansi else FORMATTING.CR_FMT + FORMATTING.BO_FMT + FORMATTING.BL_FMT
        self.NO_FMT = "" if not ansi else FORMATTING.NO_FMT # still de-assign to prevent potentially printing garbage.
        self.FIZZZZ = "" if not should_fizz else FORMATTING.FIZZZZ  # you lack a soul if you disable this with knowledge.
        self.MOD_BE = FORMATTING.MOD_B1 if ansi else FORMATTING.MOD_B2
        self.MSG_BE = FORMATTING.MSG_B1 if ansi else FORMATTING.MSG_B2
        self.MSG_GO = FORMATTING.MSG_G1 if ansi else FORMATTING.MSG_G2
        self.FNC_BE = FORMATTING.FNC_B1 if ansi else FORMATTING.FNC_B2
        self.FNC_GO = FORMATTING.FNC_G1 if ansi else FORMATTING.FNC_G2
        self.FIZ_EN = FORMATTING.FIZ_E1 if ansi else FORMATTING.FIZ_E2
        self.WIL_BA = FORMATTING.WIL_B1 if ansi else FORMATTING.WIL_B2
        self.WIL_LI = FORMATTING.WIL_L1 if ansi else FORMATTING.WIL_L2
        self.WIL_CO = FORMATTING.WIL_C1 if ansi else FORMATTING.WIL_C2
        self.WIL_DO = FORMATTING.WIL_D1 if ansi else FORMATTING.WIL_D2

        self.cce = clearance

    def format(self, record):
        parts = record.name.split(".")

        mod = parts[1] if len(parts) > 1 else "???"  # mod is the "fix".
        fnc = record.funcName

        new_mod = True if self.last_mod != mod else False
        new_fnc = True if self.last_fnc != fnc else False
        fst_mod = True if self.last_mod == "this_string_will_never_be_used" else False
        if new_mod:
            self.last_mod = mod
        if new_fnc:
            self.last_fnc = fnc

        mod_adorn = "[" + mod + "]"
        fnc_adorn = ("[" + fnc + "]") if new_fnc else ""
        pre_tooth = self.MSG_BE if new_fnc else self.MSG_GO

        # "╘╗".

        #print(new_mod, end=""); print("   ", end="")
        #print(new_fnc, end=""); print("   ", end="")
        #print(fst_mod, end=""); print("   ", end="")
        #print("", flush=True)

        match record.levelname:
            case "NOTSET":
                tooth = self.NS_FMT + "???"
            case "DEBUG":
                tooth = self.DE_FMT + "DBG"
            case "INFO":
                tooth = self.IN_FMT + "INF"
            case "WARNING":
                tooth = self.WA_FMT + "WRN"
            case "ERROR":
                tooth = self.ER_FMT + "ERR"
            case "CRITICAL":
                tooth = self.CR_FMT + "CRT"
            case _:
                tooth = self.NS_FMT + "???"
        tooth = tooth + self.NO_FMT + f"{self.FNC_BE if new_fnc else self.FNC_GO:>{self.cce - 1}}"
        # tooth is the "suffix".
        # fnc_adorn is...another suffix. goes before, though.

        # 2x cce + 2x delimeters.
        total_width = self.cce * 2 + 4

        # helpful variables. (-:
        h1 = self.cce - 1 # half 1.
        h2 = self.cce + 2 # half 2.

        # lambda city.
        mod_beg = lambda: output.append(f"{self.BO_FMT}{mod_adorn:>{self.cce}}{self.NO_FMT}{self.MOD_BE}") # effects cannot be applied before formatting.
        mod_end = lambda: output.append(f"{self.FIZZZZ:^{self.cce}}{self.FIZ_EN}")
        message = lambda: output.append(f"{tooth}{fnc_adorn:^{self.cce}}{pre_tooth} {record.getMessage()}")
        #wild_st = lambda: output.append("A" * total_width)
        wild_st = lambda: output.append(self.WIL_BA + self.WIL_LI*h1 + self.WIL_CO + self.WIL_LI*h2 + self.WIL_CO + self.WIL_BA)
        wild_en = lambda: output.append(self.WIL_BA + self.WIL_LI*h1 + self.WIL_DO + self.WIL_LI*h2 + self.WIL_DO + self.WIL_BA)

        output = []
        if new_mod or fst_mod:
            if not fst_mod:
                mod_end()
            mod_beg()

        # i really want to optimize this...
        if record.getMessage() == "%TRLog%_wild_output_begin":  wild_st()
        elif record.getMessage() == "%TRLog%_wild_output_end":  wild_en()
        else:                                                   message()

        # to those reading this, please trust that this is more efficient.
        return f"[{PREFIX}] " + f"\n[{PREFIX}] ".join(output)

def setup(level = logging.DEBUG, use_TreeRoots = True, **kwargs):
    handler = logging.StreamHandler()
    formatter = TreeRoots(**kwargs) if use_TreeRoots else logging.Formatter()
    handler.setFormatter(formatter)

    variable = logging.getLogger("lib")
    variable.setLevel(level)
    variable.addHandler(handler)
    variable.propagate = False
